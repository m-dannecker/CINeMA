"""
Training/validation state container for CINeMA.

A ``TrainState`` bundles the per-split mutable pieces of a training run:
the decoder, the per-subject learnable latent grid, the optional per-subject
rigid/affine transformations, the optimizer, grad scaler and scheduler, plus
the dataset and its dataloader. A ``TrainState`` is created once at the start
of training; stage 7 will reuse the same factories to build a *fitting* state
for val/test subjects where the decoder weights are frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .config import DecoderConfig, OptimizerConfig, TrainConfig
from .data import Data
from .models.inr_decoder import INRDecoder


@dataclass
class TrainState:
    split: str
    device: str
    decoder: INRDecoder
    dataset: Data
    dataloader: DataLoader
    latents: nn.Parameter
    transformations: Optional[nn.Parameter]                  # None if tf_dim == 0
    conditions: Optional[nn.Parameter]                       # set for val/test fit
    optimizer: optim.Optimizer
    scaler: Optional[GradScaler]
    scheduler: Optional[CosineAnnealingLR]
    amp: bool

    def re_init_latents(self, std: float = 0.01) -> None:
        with torch.no_grad():
            self.latents.data.normal_(0, std)
            if self.transformations is not None and self.transformations.requires_grad:
                self.transformations.data.zero_()
        self.optimizer.zero_grad(set_to_none=True)

    def step_scheduler(self) -> None:
        if self.scheduler is not None:
            self.scheduler.step()


def build_latents(
    n_subjects: int,
    latent_dim: list[int],
    *,
    device: str,
    init_tensor: Optional[torch.Tensor] = None,
    std: float = 0.01,
) -> nn.Parameter:
    shape = (n_subjects, *latent_dim)
    if init_tensor is None:
        data = torch.normal(0, std, size=shape, device=device)
    else:
        if tuple(init_tensor.shape) != shape:
            raise ValueError(
                f"init_tensor shape {tuple(init_tensor.shape)} does not match "
                f"expected latent shape {shape}"
            )
        data = init_tensor.to(device=device, dtype=torch.float32)
    return nn.Parameter(data)


def build_transformations(
    n_subjects: int,
    tf_dim: int,
    *,
    device: str,
    init_tensor: Optional[torch.Tensor] = None,
    trainable: bool = True,
) -> Optional[nn.Parameter]:
    if tf_dim <= 0:
        return None
    shape = (n_subjects, max(tf_dim, 6))
    if init_tensor is None:
        data = torch.zeros(shape, device=device)
    else:
        data = init_tensor.to(device=device, dtype=torch.float32)
    tensor = nn.Parameter(data, requires_grad=trainable)
    return tensor


def build_conditions_param(
    n_subjects: int,
    n_cond_dims: int,
    *,
    device: str,
    std: float = 0.01,
) -> nn.Parameter:
    return nn.Parameter(
        torch.normal(0, std, size=(n_subjects, n_cond_dims), device=device)
    )


def _optimizer_param_groups(
    *,
    opt_cfg: OptimizerConfig,
    decoder: INRDecoder,
    latents: nn.Parameter,
    transformations: Optional[nn.Parameter],
    conditions: Optional[nn.Parameter],
    include_decoder: bool,
) -> list[dict]:
    groups: list[dict] = [{
        "name": "latents",
        "params": [latents],
        "lr": opt_cfg.lr_latent,
        "weight_decay": opt_cfg.latent_weight_decay,
    }]
    if transformations is not None and transformations.requires_grad:
        groups.append({
            "name": "transformations",
            "params": [transformations],
            "lr": opt_cfg.lr_tf,
            "weight_decay": opt_cfg.tf_weight_decay,
        })
    if include_decoder:
        groups.append({
            "name": "decoder",
            "params": list(decoder.parameters()),
            "lr": opt_cfg.lr_inr,
            "weight_decay": opt_cfg.inr_weight_decay,
        })
    if conditions is not None:
        groups.append({
            "name": "conditions",
            "params": [conditions],
            "lr": opt_cfg.lr_latent,
            "weight_decay": opt_cfg.latent_weight_decay,
        })
    return groups


def build_optimizer(
    *,
    opt_cfg: OptimizerConfig,
    decoder: INRDecoder,
    latents: nn.Parameter,
    transformations: Optional[nn.Parameter],
    conditions: Optional[nn.Parameter],
    include_decoder: bool,
) -> optim.Optimizer:
    groups = _optimizer_param_groups(
        opt_cfg=opt_cfg,
        decoder=decoder,
        latents=latents,
        transformations=transformations,
        conditions=conditions,
        include_decoder=include_decoder,
    )
    return optim.AdamW(groups)


def build_scheduler(
    optimizer: optim.Optimizer,
    opt_cfg: OptimizerConfig,
    *,
    n_epochs: int,
) -> Optional[CosineAnnealingLR]:
    if opt_cfg.scheduler.type == "cosine":
        return CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=opt_cfg.scheduler.eta_min
        )
    return None


def build_decoder(
    decoder_cfg: DecoderConfig,
    *,
    n_cond_dims: int,
    has_segmentation: bool,
    device: str,
) -> INRDecoder:
    return INRDecoder(
        decoder_cfg, n_cond_dims=n_cond_dims, has_segmentation=has_segmentation,
    ).to(device)


def build_train_state(
    cfg: TrainConfig,
    *,
    dataset: Data,
    decoder: Optional[INRDecoder] = None,
    decoder_state_dict: Optional[dict] = None,
    latents_init: Optional[torch.Tensor] = None,
    transformations_init: Optional[torch.Tensor] = None,
    dataloader_shuffle: Optional[bool] = None,
) -> TrainState:
    """Build a training-side ``TrainState`` from a ``TrainConfig``.

    The decoder is trainable and included in the optimizer; no learnable
    ``conditions`` parameter is built (only fit states have that). If a
    ``decoder`` is passed in, it's reused; otherwise a fresh one is built and
    optionally loaded from ``decoder_state_dict``.
    """
    device = cfg.device
    n_cond_dims = len(cfg.conditions.enabled_specs())

    if decoder is None:
        decoder = build_decoder(
            cfg.decoder,
            n_cond_dims=n_cond_dims,
            has_segmentation=cfg.dataset_spec.has_segmentation,
            device=device,
        )
        if decoder_state_dict is not None:
            decoder.load_state_dict(decoder_state_dict)

    latents = build_latents(
        len(dataset), cfg.decoder.latent_dim, device=device, init_tensor=latents_init,
    )
    tfs = build_transformations(
        len(dataset), cfg.decoder.tf_dim, device=device, init_tensor=transformations_init,
    )

    optimizer = build_optimizer(
        opt_cfg=cfg.optimizer,
        decoder=decoder,
        latents=latents,
        transformations=tfs,
        conditions=None,
        include_decoder=True,
    )
    scheduler = build_scheduler(optimizer, cfg.optimizer, n_epochs=cfg.training.epochs)
    scaler = GradScaler() if cfg.amp else None

    shuffle = dataloader_shuffle if dataloader_shuffle is not None else True
    dataloader = DataLoader(
        dataset,
        batch_size=(cfg.training.batch_size if cfg.training.batch_size > 0 else len(dataset)),
        num_workers=cfg.training.num_workers,
        shuffle=shuffle,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
    )

    return TrainState(
        split="train",
        device=device,
        decoder=decoder,
        dataset=dataset,
        dataloader=dataloader,
        latents=latents,
        transformations=tfs,
        conditions=None,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=scheduler,
        amp=cfg.amp,
    )


def build_fit_state(
    cfg: TrainConfig,
    *,
    dataset: Data,
    decoder: INRDecoder,
    split: str = "val",
    latents_init: Optional[torch.Tensor] = None,
    transformations_init: Optional[torch.Tensor] = None,
    learn_conditions: bool = True,
    n_epochs: Optional[int] = None,
    dataloader_shuffle: Optional[bool] = None,
) -> TrainState:
    """Build a frozen-decoder fitting state for val/test subjects.

    Differs from ``build_train_state`` in three ways:
    - ``decoder`` is required (it's assumed to be a pre-trained model passed in)
      and is *excluded* from the optimizer — its weights won't be updated.
    - A learnable per-subject ``conditions`` parameter is created when
      ``learn_conditions`` is True and at least one condition is enabled,
      so the fitter can predict conditions for val subjects.
    - The scheduler uses ``n_epochs`` (the fit-epoch budget) for ``T_max``,
      falling back to ``cfg.training.epochs`` when not specified.
    """
    device = cfg.device
    n_cond_dims = len(cfg.conditions.enabled_specs())

    latents = build_latents(
        len(dataset), cfg.decoder.latent_dim, device=device, init_tensor=latents_init,
    )
    tfs = build_transformations(
        len(dataset), cfg.decoder.tf_dim, device=device, init_tensor=transformations_init,
    )
    conditions = (
        build_conditions_param(len(dataset), n_cond_dims, device=device)
        if learn_conditions and n_cond_dims > 0
        else None
    )

    optimizer = build_optimizer(
        opt_cfg=cfg.optimizer,
        decoder=decoder,
        latents=latents,
        transformations=tfs,
        conditions=conditions,
        include_decoder=False,
    )
    sched_epochs = n_epochs if n_epochs is not None else cfg.training.epochs
    scheduler = build_scheduler(optimizer, cfg.optimizer, n_epochs=sched_epochs)
    scaler = GradScaler() if cfg.amp else None

    shuffle = dataloader_shuffle if dataloader_shuffle is not None else True
    dataloader = DataLoader(
        dataset,
        batch_size=(cfg.training.batch_size if cfg.training.batch_size > 0 else len(dataset)),
        num_workers=cfg.training.num_workers,
        shuffle=shuffle,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
    )

    return TrainState(
        split=split,
        device=device,
        decoder=decoder,
        dataset=dataset,
        dataloader=dataloader,
        latents=latents,
        transformations=tfs,
        conditions=conditions,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=scheduler,
        amp=cfg.amp,
    )
