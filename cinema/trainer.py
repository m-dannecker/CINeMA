"""
Pure training loop for CINeMA.

A ``Trainer`` wraps a ``TrainState`` and a ``Criterion`` and runs one epoch or
one batch at a time. It does not touch evaluation, atlas generation, or
checkpoint I/O — those live in separate modules. The validation inner-loop
(fit latents+tfs on val subjects while decoder weights are frozen) is also
separate (stage 7's ``LatentFitter``), even though it reuses the same batch
semantics.

The ``batch`` tuple matches ``Data.collate_fn``'s output:
``(coords, values, conditions, idx_df)``. Each outer DataLoader batch is
chunked into inner mini-batches of size ``n_samples`` (one optimizer step
per chunk), which is what keeps GPU memory bounded when many subjects
contribute millions of coords per outer batch.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np
import torch

from .criterion import Criterion, LossBreakdown
from .state import TrainState


BatchCallback = Callable[[LossBreakdown, int, int], None]
BatchAssembledCallback = Callable[[int, float], None]


def _to_device(batch, device):
    return tuple(x.to(device, non_blocking=True) for x in batch)


class Trainer:
    """Runs one training epoch over a ``TrainState``.

    Parameters
    ----------
    state : TrainState
        The state being trained (decoder + latents + tfs + optimizer).
    criterion : Criterion
        Loss function. Segmentation term is zeroed automatically when the
        dataset has no segmentation modality.
    n_samples : int
        Inner mini-batch size (coords per optimizer step).
    seg_weight : float
        Default segmentation-loss weight. Can be overridden per-batch via
        ``train_batch(..., seg_weight=...)``.
    on_batch_end : callable, optional
        ``(loss_breakdown, epoch, chunk_idx)`` callback after each gradient
        step — useful for wandb logging without coupling the trainer to it.
    on_batch_assembled : callable, optional
        ``(batch_idx, elapsed_secs)`` callback fired after each DataLoader
        batch is fetched but before training starts on it.
    """

    def __init__(
        self,
        state: TrainState,
        criterion: Criterion,
        *,
        n_samples: int,
        seg_weight: float = 1.0,
        on_batch_end: Optional[BatchCallback] = None,
        on_batch_assembled: Optional[BatchAssembledCallback] = None,
        set_decoder_train: bool = True,
    ):
        self.state = state
        self.criterion = criterion
        self.n_samples = int(n_samples)
        self.seg_weight = float(seg_weight)
        self.on_batch_end = on_batch_end
        self.on_batch_assembled = on_batch_assembled
        self.set_decoder_train = bool(set_decoder_train)

    def train_epoch(self, epoch: int) -> float:
        """Iterate the dataloader once, return mean total loss."""
        if self.set_decoder_train:
            self.state.decoder.train()
        losses: list[float] = []
        it = iter(self.state.dataloader)
        batch_idx = 0
        while True:
            t0 = time.perf_counter()
            try:
                batch = next(it)
            except StopIteration:
                break
            elapsed = time.perf_counter() - t0
            if self.on_batch_assembled is not None:
                self.on_batch_assembled(batch_idx, elapsed)
            losses.append(self.train_batch(batch, epoch))
            batch_idx += 1
        return float(np.mean(losses)) if losses else float("nan")

    def train_batch(
        self,
        batch,
        epoch: int,
        *,
        seg_weight: Optional[float] = None,
    ) -> float:
        """Train on one outer batch (chunked into ``n_samples`` inner steps)."""
        if seg_weight is None:
            seg_weight = self.seg_weight
        coords_b, values_b, conds_b, idx_df_b = _to_device(batch, self.state.device)
        n = idx_df_b.shape[0]
        chunk_losses: list[float] = []
        for c, start in enumerate(range(0, n, self.n_samples)):
            end = start + self.n_samples
            coords = coords_b[start:end]
            values = values_b[start:end]
            idx_df = idx_df_b[start:end].squeeze(-1)
            conds = self._conditions_for_chunk(conds_b[start:end], idx_df)

            loss = self._gradient_step(coords, values, conds, idx_df, seg_weight)
            chunk_losses.append(loss.total.detach().item())
            if self.on_batch_end is not None:
                self.on_batch_end(loss, epoch, c)
        return float(np.mean(chunk_losses)) if chunk_losses else float("nan")

    def _conditions_for_chunk(
        self, conds_chunk: torch.Tensor, idx_df: torch.Tensor,
    ) -> torch.Tensor:
        """When learnable per-subject conditions exist, use them instead of
        the per-coord conditions carried in the batch (val/test fit case)."""
        if self.state.conditions is None:
            return conds_chunk
        return self.state.conditions[idx_df]

    def _gradient_step(
        self,
        coords: torch.Tensor,
        values: torch.Tensor,
        conds: torch.Tensor,
        idx_df: torch.Tensor,
        seg_weight: float,
    ) -> LossBreakdown:
        state = self.state
        opt = state.optimizer
        opt.zero_grad(set_to_none=True)
        tfs = (
            state.transformations[idx_df] if state.transformations is not None else None
        )
        with torch.autocast(device_type=state.device, enabled=state.amp):
            pred = state.decoder(
                coords, state.latents, conds, idcs_df=idx_df, tfs=tfs,
            )
            loss = self.criterion(pred, values, tfs, seg_weight=seg_weight)

        if state.scaler is not None:
            state.scaler.scale(loss.total).backward()
            state.scaler.step(opt)
            state.scaler.update()
        else:
            loss.total.backward()
            opt.step()
        return loss
