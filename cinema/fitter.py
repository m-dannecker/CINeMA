"""
Latent fitter for CINeMA val/test subjects.

Once the decoder is trained, new subjects (val-split at every validation cycle,
or unseen test subjects at inference time) need per-subject latents and
transformations fit to them. The decoder stays frozen — only the latent grid,
the optional rigid/affine tfs, and optionally a learnable conditions vector are
updated.

The fitter reuses ``Trainer.train_batch`` semantics directly: the decoder in
``TrainState.decoder`` has ``requires_grad=False`` on all parameters, and the
optimizer's param groups (built by ``build_fit_state`` with
``include_decoder=False``) never include them. So a single ``train_epoch``
call does the right thing without a separate loop.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from .criterion import Criterion, LossBreakdown
from .state import TrainState
from .trainer import Trainer


EpochCallback = Callable[[float, int], None]


class LatentFitter:
    """Fit per-subject latents (+ optional tfs + conditions) with decoder frozen.

    Parameters mirror ``Trainer``'s, plus an ``n_epochs`` fit budget and an
    ``on_epoch_end(loss, epoch)`` callback. The decoder's parameters are set
    to ``requires_grad=False`` on construction and the module is put into
    ``eval`` mode; the inner ``Trainer`` is configured not to flip it back to
    ``train`` mode at the start of each epoch.
    """

    def __init__(
        self,
        state: TrainState,
        criterion: Criterion,
        *,
        n_samples: int,
        n_epochs: int,
        seg_weight: float = 1.0,
        on_batch_end: Optional[Callable[[LossBreakdown, int, int], None]] = None,
        on_epoch_end: Optional[EpochCallback] = None,
    ):
        self.state = state
        self.n_epochs = int(n_epochs)
        for p in state.decoder.parameters():
            p.requires_grad = False
        state.decoder.eval()
        self._trainer = Trainer(
            state, criterion,
            n_samples=n_samples,
            seg_weight=seg_weight,
            on_batch_end=on_batch_end,
            set_decoder_train=False,
        )
        self.on_epoch_end = on_epoch_end

    def fit(self) -> list[float]:
        """Run ``n_epochs`` fit epochs. Returns the list of per-epoch mean losses."""
        losses: list[float] = []
        for ep in range(self.n_epochs):
            loss = self._trainer.train_epoch(ep)
            self.state.step_scheduler()
            losses.append(loss)
            if self.on_epoch_end is not None:
                self.on_epoch_end(loss, ep)
        return losses
