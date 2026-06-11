"""
Versioned checkpoint save/load for CINeMA.

A checkpoint is a single ``.pt`` file snapshotting enough state to either:
- Resume training: decoder weights + latents + tfs + optimizer / scheduler /
  scaler state + current epoch + the full ``TrainConfig``.
- Drive inference-only paths (fit / evaluate / atlas): decoder weights plus
  the ``ConditionRegistry`` and ``DatasetSpec`` (carried inside ``TrainConfig``).

Schema is hard-versioned. A mismatch raises immediately — migrations must be
explicit, not implicit. We embed the ``TrainConfig`` object directly (pickle
via ``torch.save``), so checkpoints are tightly coupled to CINeMA's class
layout; bump ``SCHEMA_VERSION`` and write a converter when that layout changes
in a breaking way.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch

from .config import TrainConfig
from .state import TrainState


SCHEMA_VERSION = 1


@dataclass
class Checkpoint:
    version: int
    epoch: int
    config: TrainConfig
    decoder_state_dict: dict
    latents: torch.Tensor
    transformations: Optional[torch.Tensor]
    conditions: Optional[torch.Tensor]
    optimizer_state: Optional[dict]
    scheduler_state: Optional[dict]
    scaler_state: Optional[dict]


def save_checkpoint(
    path: Union[str, Path],
    *,
    state: TrainState,
    config: TrainConfig,
    epoch: int,
) -> Path:
    """Serialise a full snapshot to ``path`` atomically.

    Writes to ``path.tmp`` first then renames on success, so a crash mid-write
    cannot leave a partial file where the previous good one lived.
    """
    payload = {
        "version": SCHEMA_VERSION,
        "epoch": int(epoch),
        "config": config,
        "decoder_state_dict": state.decoder.state_dict(),
        "latents": state.latents.detach().cpu(),
        "transformations": (
            state.transformations.detach().cpu()
            if state.transformations is not None else None
        ),
        "conditions": (
            state.conditions.detach().cpu()
            if state.conditions is not None else None
        ),
        "optimizer_state": state.optimizer.state_dict(),
        "scheduler_state": (
            state.scheduler.state_dict() if state.scheduler is not None else None
        ),
        "scaler_state": (
            state.scaler.state_dict() if state.scaler is not None else None
        ),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return path


def load_checkpoint(
    path: Union[str, Path],
    *,
    map_location: Union[str, torch.device] = "cpu",
) -> Checkpoint:
    """Load a checkpoint and validate its schema version.

    Raises ``ValueError`` if the file is not a CINeMA checkpoint or its schema
    version doesn't match the loader's ``SCHEMA_VERSION``.
    """
    payload = torch.load(
        str(path), map_location=map_location, weights_only=False,
    )
    if not isinstance(payload, dict) or "version" not in payload:
        raise ValueError(
            f"{path} is not a CINeMA checkpoint (missing 'version' key)"
        )
    v = int(payload["version"])
    if v != SCHEMA_VERSION:
        raise ValueError(
            f"checkpoint {path} is schema v{v}, loader expects "
            f"v{SCHEMA_VERSION}. Explicit migration required."
        )
    return Checkpoint(
        version=v,
        epoch=int(payload["epoch"]),
        config=payload["config"],
        decoder_state_dict=payload["decoder_state_dict"],
        latents=payload["latents"],
        transformations=payload.get("transformations"),
        conditions=payload.get("conditions"),
        optimizer_state=payload.get("optimizer_state"),
        scheduler_state=payload.get("scheduler_state"),
        scaler_state=payload.get("scaler_state"),
    )


def apply_to_state(
    checkpoint: Checkpoint,
    state: TrainState,
    *,
    load_optimizer: bool = True,
) -> None:
    """Overlay a loaded checkpoint onto a fresh state in-place.

    The state's latents / transformations / conditions must already be sized
    to match the checkpoint's — build the state for the same dataset size and
    decoder config first, then call this to load the trained weights.

    ``load_optimizer=False`` skips optimizer / scheduler / scaler restore — use
    it when a fit-checkpoint (whose optimizer excludes the decoder + may add a
    learnable-conditions group) is being overlaid onto a fresh train-state for
    pure inference (``infer`` / ``atlas`` / ``evaluate``), where the param-group
    layouts differ and no training step will ever read them.
    """
    state.decoder.load_state_dict(checkpoint.decoder_state_dict)
    with torch.no_grad():
        if state.latents.shape != checkpoint.latents.shape:
            raise ValueError(
                f"latents shape mismatch: state={tuple(state.latents.shape)} "
                f"checkpoint={tuple(checkpoint.latents.shape)}"
            )
        state.latents.copy_(checkpoint.latents.to(state.latents.device))
        if checkpoint.transformations is not None and state.transformations is not None:
            if state.transformations.shape != checkpoint.transformations.shape:
                raise ValueError(
                    f"transformations shape mismatch: "
                    f"state={tuple(state.transformations.shape)} "
                    f"checkpoint={tuple(checkpoint.transformations.shape)}"
                )
            state.transformations.copy_(
                checkpoint.transformations.to(state.transformations.device)
            )
        if checkpoint.conditions is not None and state.conditions is not None:
            if state.conditions.shape != checkpoint.conditions.shape:
                raise ValueError(
                    f"conditions shape mismatch: "
                    f"state={tuple(state.conditions.shape)} "
                    f"checkpoint={tuple(checkpoint.conditions.shape)}"
                )
            state.conditions.copy_(
                checkpoint.conditions.to(state.conditions.device)
            )
    if not load_optimizer:
        return
    if checkpoint.optimizer_state is not None:
        state.optimizer.load_state_dict(checkpoint.optimizer_state)
    if checkpoint.scheduler_state is not None and state.scheduler is not None:
        state.scheduler.load_state_dict(checkpoint.scheduler_state)
    if checkpoint.scaler_state is not None and state.scaler is not None:
        state.scaler.load_state_dict(checkpoint.scaler_state)
