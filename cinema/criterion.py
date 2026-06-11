"""
Loss function for the CINeMA decoder.

Separate ``sr`` (intensity), ``seg`` (segmentation) and ``tf`` (transformation
regulariser) terms are returned so they can be logged individually. The total
is ``sr_weight * sr + seg_weight * seg + tf_weight * tf``.

When ``has_segmentation=False`` the ``seg`` term is always zero and the target
is not expected to carry a segmentation column — the caller should pass a
target tensor whose trailing dim matches ``intensity_dims`` only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn


@dataclass
class LossBreakdown:
    sr: torch.Tensor
    seg: torch.Tensor
    tf: torch.Tensor
    total: torch.Tensor

    def item_dict(self) -> dict[str, float]:
        return {
            "sr": float(self.sr.detach().cpu().item()),
            "seg": float(self.seg.detach().cpu().item()),
            "tf": float(self.tf.detach().cpu().item()),
            "total": float(self.total.detach().cpu().item()),
        }


class Criterion(nn.Module):
    def __init__(
        self,
        *,
        intensity_dims: int,
        has_segmentation: bool,
        class_weights: Optional[Sequence[float]] = None,
        loss_metric: str = "l1",
        tf_weight: float = 0.0,
    ):
        super().__init__()
        if loss_metric not in ("l1", "mse"):
            raise ValueError(f"unknown loss_metric: {loss_metric!r}")
        self.intensity_dims = int(intensity_dims)
        self.has_segmentation = bool(has_segmentation)
        self.tf_weight = float(tf_weight)
        self.criterion_sr = nn.MSELoss() if loss_metric == "mse" else nn.L1Loss()
        if has_segmentation and class_weights is not None:
            self.register_buffer(
                "_class_weights",
                torch.tensor(list(class_weights), dtype=torch.float32),
            )
        else:
            self._class_weights = None
        self.criterion_seg = (
            nn.CrossEntropyLoss(weight=self._class_weights)
            if has_segmentation
            else None
        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        tfs: Optional[torch.Tensor] = None,
        *,
        sr_weight: float = 1.0,
        seg_weight: float = 1.0,
    ) -> LossBreakdown:
        device = pred.device
        sr = sr_weight * self.criterion_sr(
            pred[..., : self.intensity_dims], target[..., : self.intensity_dims]
        )

        if self.has_segmentation and seg_weight > 0:
            seg_logits = pred[..., self.intensity_dims :]
            seg_target = target[..., -1].to(torch.int64)
            seg = self.criterion_seg(seg_logits, seg_target)
        else:
            seg = torch.zeros((), device=device)

        if tfs is not None and self.tf_weight > 0:
            tf_rot = tfs[..., :3].pow(2).mean()
            tf_trans = tfs[..., 3:6].pow(2).mean()
            tf_scale = (
                tfs[..., 6:9].pow(2).mean()
                if tfs.shape[-1] >= 9
                else torch.zeros((), device=device)
            )
            tf = tf_rot + tf_trans + tf_scale
        else:
            tf = torch.zeros((), device=device)

        total = sr + seg_weight * seg + self.tf_weight * tf
        return LossBreakdown(sr=sr, seg=seg, tf=tf, total=total)
