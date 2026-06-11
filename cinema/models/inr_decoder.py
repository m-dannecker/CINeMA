"""
Implicit Neural Representation decoder for CINeMA.

The decoder is a SIREN MLP whose hidden layers are FiLM-modulated by features
sampled from a small per-subject 3D latent grid (optionally smoothed by a 3D
CNN). On top of coordinates and latent features, a per-coord *condition vector*
(physical-unit values normalized by the ``ConditionRegistry``) is concatenated.

The split between intensity and segmentation channels is explicit and driven by
``DecoderConfig.out_dim`` interpreted *together with* ``has_segmentation``:

- ``has_segmentation=True``  → ``out_dim[:-1]`` are intensity dims (one per
  intensity modality), ``out_dim[-1]`` is the number of segmentation classes.
- ``has_segmentation=False`` → all of ``out_dim`` is intensity. This is the
  test-subject-without-labels case fed to ``cinema fit``.

``forward`` is the per-coord training path. ``predict_volume`` is the
single-subject volume reconstruction path used by evaluator + atlas. They share
the same internals but differ in chunking, post-processing, and what they
return — so we keep them separate rather than overloading one entry point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import label as scipy_label

from ..affine import apply_embedded_affine
from ..config import DecoderConfig
from .siren import Siren


@dataclass
class DecoderOutput:
    """Per-coord raw decoder output split by intensity / segmentation."""

    intensity_logits: torch.Tensor                  # (N, n_intensity_dims)
    seg_logits: Optional[torch.Tensor] = None       # (N, n_seg_classes) or None


@dataclass
class VolumeReconstruction:
    """Volume-shape (X, Y, Z, ...) reconstruction of a single subject."""

    intensities: torch.Tensor                       # (X, Y, Z, n_intensity_dims) in [0,1]
    seg_hard: Optional[torch.Tensor] = None         # (X, Y, Z) int64
    seg_soft: Optional[torch.Tensor] = None         # (X, Y, Z, n_classes)


class Modulator(nn.Module):
    """Optional 3D CNN over the per-subject latent volume.

    ``kernel_size=0`` collapses to identity — common for very small latent
    grids where there is nothing meaningful to convolve over.
    """

    def __init__(self, latent_dims: Sequence[int], kernel_size: int = 3):
        super().__init__()
        if kernel_size > 0:
            self.conv = nn.Conv3d(
                latent_dims[0], latent_dims[0], kernel_size, padding="same"
            )
        else:
            self.conv = nn.Identity()

    def forward(self, latent_stack: torch.Tensor) -> torch.Tensor:
        return self.conv(latent_stack)


class INRDecoder(nn.Module):
    """SIREN-based decoder modulated by per-subject latents and conditions."""

    def __init__(
        self,
        decoder_cfg: DecoderConfig,
        n_cond_dims: int,
        has_segmentation: bool,
    ):
        super().__init__()
        self.cfg = decoder_cfg
        self.n_cond_dims = int(n_cond_dims)
        self.has_segmentation = bool(has_segmentation)

        out_dim = list(decoder_cfg.out_dim)
        if self.has_segmentation:
            if len(out_dim) < 2:
                raise ValueError(
                    "decoder.out_dim must have at least 2 entries when "
                    "has_segmentation=True (intensity dims + seg class count)"
                )
            self.intensity_dims = int(sum(out_dim[:-1]))
            self.seg_classes = int(out_dim[-1])
        else:
            self.intensity_dims = int(sum(out_dim))
            self.seg_classes = 0
        self.total_out = self.intensity_dims + self.seg_classes

        latent_channels = int(decoder_cfg.latent_dim[0])
        latent_size = latent_channels + self.n_cond_dims

        self.modulator = Modulator(
            decoder_cfg.latent_dim, kernel_size=int(decoder_cfg.cnn_kernel_size)
        )
        self.siren = Siren(
            in_size=int(decoder_cfg.in_dim),
            latent_size=latent_size,
            out_size=self.total_out,
            hidden_size=int(decoder_cfg.hidden_size),
            num_hidden_layers=int(decoder_cfg.num_hidden_layers),
            first_omega=float(decoder_cfg.omega[0]),
            hidden_omega=float(decoder_cfg.omega[1]),
            modulated_layers=list(decoder_cfg.modulated_layers),
            outermost_linear=True,
        )

    # === Training path ===

    def forward(
        self,
        coords: torch.Tensor,
        latent_stack: torch.Tensor,
        condition_vecs: torch.Tensor,
        idcs_df: torch.Tensor,
        tfs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Per-coord decoder pass.

        ``latent_stack`` holds *all* subjects' latent grids; ``idcs_df`` selects
        one subject per coord. This matches the mixed-batch sampling pattern
        the trainer uses (``Data.collate_fn`` shuffles voxels across subjects).
        """
        if tfs is not None:
            coords = apply_embedded_affine(coords, tfs)
        modulations = self.modulator(latent_stack)         # (S, C, lx, ly, lz)
        per_coord = modulations[idcs_df.long()]            # (N, C, lx, ly, lz)
        feats = self._spatial_interpolate(coords, per_coord, condition_vecs)
        return self.siren((coords, feats))

    def split_output(self, raw: torch.Tensor) -> DecoderOutput:
        """Slice raw decoder output into its intensity / seg parts."""
        if self.has_segmentation:
            return DecoderOutput(
                intensity_logits=raw[..., : self.intensity_dims],
                seg_logits=raw[..., self.intensity_dims :],
            )
        return DecoderOutput(intensity_logits=raw, seg_logits=None)

    # === Inference path ===

    @torch.no_grad()
    def predict_volume(
        self,
        coords: torch.Tensor,
        latent_vec: torch.Tensor,
        condition_vec: torch.Tensor,
        img_shape: Sequence[int],
        *,
        tfs: Optional[torch.Tensor] = None,
        step_size: int = 100_000,
        renormalize_per_modality: bool = False,
    ) -> VolumeReconstruction:
        """Single-subject volume reconstruction.

        - ``latent_vec``: ``(1, C, lx, ly, lz)`` — single latent grid.
        - ``condition_vec``: ``(n_cond_dims,)`` or ``(1, n_cond_dims)``.
        - ``tfs``: ``(1, 6/9/12)`` or ``None`` if no per-subject affine.
        - Intensities are clamped to ``[0, 1]``. Set ``renormalize_per_modality``
          to additionally rescale each modality channel to span ``[0, 1]`` —
          useful for atlas display where per-modality contrast matters more
          than absolute intensity calibration.
        """
        if condition_vec.ndim == 1:
            condition_vec = condition_vec[None]
        if condition_vec.shape[-1] != self.n_cond_dims:
            raise ValueError(
                f"condition_vec has {condition_vec.shape[-1]} dims but decoder "
                f"expects {self.n_cond_dims}"
            )
        device = coords.device
        n = coords.shape[0]
        out = torch.empty((n, self.total_out), device=device)
        # forward() materialises a per-coord latent gather of shape
        # (chunk, C, lx, ly, lz); cap the chunk so large latent grids don't OOM
        # (purely a memory/throughput knob — does not change the result).
        elem_per_coord = int(np.prod(latent_vec.shape[1:]))
        eff_step = min(step_size, max(2000, 800_000_000 // max(elem_per_coord, 1)))
        for i in range(0, n, eff_step):
            c = coords[i : i + eff_step]
            cv = condition_vec.expand(c.shape[0], -1)
            chunk_idx = torch.zeros(c.shape[0], dtype=torch.long, device=device)
            chunk_tfs = tfs.expand(c.shape[0], -1) if tfs is not None else None
            out[i : i + eff_step] = self.forward(
                c, latent_vec, cv, idcs_df=chunk_idx, tfs=chunk_tfs
            )

        intensity = torch.clamp(out[..., : self.intensity_dims], 0.0, 1.0)
        if renormalize_per_modality and self.intensity_dims > 0:
            mn = intensity.amin(dim=0, keepdim=True)
            mx = intensity.amax(dim=0, keepdim=True)
            denom = torch.where(mx - mn > 0, mx - mn, torch.ones_like(mx))
            intensity = (intensity - mn) / denom
        intensity = intensity.reshape(*img_shape, self.intensity_dims)

        if self.has_segmentation:
            seg_logits = out[..., self.intensity_dims :]
            seg_soft = F.softmax(seg_logits, dim=-1).reshape(
                *img_shape, self.seg_classes
            )
            seg_hard = (
                torch.argmax(seg_logits, dim=-1).reshape(*img_shape).to(torch.int64)
            )
            return VolumeReconstruction(
                intensities=intensity, seg_hard=seg_hard, seg_soft=seg_soft
            )
        return VolumeReconstruction(intensities=intensity)

    # === Internals ===

    @staticmethod
    def _spatial_interpolate(
        coords: torch.Tensor,
        latents: torch.Tensor,
        condition_vecs: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Trilinearly sample a per-coord latent grid; concat with conditions."""
        sample_coords = coords[:, None, None, None, :]
        sampled = F.grid_sample(
            latents, sample_coords, mode="bilinear",
            align_corners=True, padding_mode="border",
        ).squeeze(-1).squeeze(-1).squeeze(-1)
        if condition_vecs is not None:
            sampled = torch.cat((sampled, condition_vecs), dim=-1)
        return sampled


def mask_by_largest_component(
    seg_hard: torch.Tensor,
    label_names: Sequence[str],
    *,
    bg_label_str: str = "BG",
    halo_sigma: float = 1.0,
) -> torch.Tensor:
    """Return a smooth float mask isolating the largest connected component.

    The mask drops everything that isn't either ``bg_label_str`` or part of the
    central connected blob, then smooths the result with a Gaussian and
    thresholds it back to ``{0, 1}``. Used by atlas/evaluator to suppress
    spurious far-from-brain reconstructions.
    """
    if bg_label_str in label_names:
        bg_label = label_names.index(bg_label_str)
    else:
        bg_label = -1

    seg_np = seg_hard.detach().cpu().numpy()
    mask = ((seg_np > 0) & (seg_np != bg_label)).astype(np.uint8)
    if mask.sum() == 0:
        return torch.zeros_like(seg_hard, dtype=torch.float32)

    labeled, _ = scipy_label(mask)
    shp = np.array(mask.shape)
    cp = shp // 2
    ps = np.maximum((shp * 0.1 // 2).astype(int), 1)
    patch = labeled[
        cp[0] - ps[0] : cp[0] + ps[0],
        cp[1] - ps[1] : cp[1] + ps[1],
        cp[2] - ps[2] : cp[2] + ps[2],
    ].ravel()
    patch = patch[patch > 0]
    if patch.size == 0:
        return torch.zeros_like(seg_hard, dtype=torch.float32)
    majority = int(np.bincount(patch).argmax())

    mask = (mask & (labeled == majority)).astype(np.float32)
    smooth = (ndi.gaussian_filter(mask, sigma=halo_sigma) > 1e-3).astype(np.float32)
    return torch.from_numpy(smooth).to(seg_hard.device)


class LatentRegressor(nn.Module):
    """Tiny MLP regressing a scalar (e.g. birth_age) from a flattened latent."""

    def __init__(self, latent_dim: Sequence[int]):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Linear(int(latent_dim[0]), 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.sequence(latent)
