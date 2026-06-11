"""
Evaluation helpers for CINeMA.

Functional module — no class — so each helper takes what it needs explicitly
and has no hidden state. Three concerns:

- Reconstruction: build a world grid for a subject, run ``INRDecoder.predict_volume``
  with that subject's latent / tfs / conditions, write the resulting NIfTI.
- Latent-space analysis: predict a scalar condition (``scan_age`` / ``birth_age``)
  from the learned latent grids via NCA + kNN on rounded labels. Mirrors the
  legacy ``predict_cond_value`` path; the MLP ``LatentRegressor`` variant is
  not ported.
- Metrics (ANTs): PSNR/SSIM/Dice on registered pred↔ref volumes. Gated to an
  opt-in submodule ``cinema.evaluator_ants`` so ``ants`` is only imported when
  the caller actually needs it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import nibabel as nib
import numpy as np
import torch

from .models.inr_decoder import VolumeReconstruction, mask_by_largest_component
from .state import TrainState


def apply_largest_component_mask(
    recon: VolumeReconstruction, label_names: Sequence[str],
) -> VolumeReconstruction:
    """Zero intensities + seg outside the largest connected foreground component.

    Uses ``mask_by_largest_component`` on the hard seg; intensities are masked
    channel-wise and the seg map is re-multiplied by the same mask. Soft seg
    logits are left untouched. No-op if the reconstruction has no seg map.
    """
    if recon.seg_hard is None:
        return recon
    mask = mask_by_largest_component(recon.seg_hard, label_names)
    intensities = recon.intensities * mask.unsqueeze(-1)
    seg_hard = recon.seg_hard * mask.to(recon.seg_hard.dtype)
    return VolumeReconstruction(
        intensities=intensities, seg_hard=seg_hard, seg_soft=recon.seg_soft,
    )


def generate_world_grid(
    world_bbox: Sequence[float],
    spacing: Sequence[float],
    *,
    device: str = "cpu",
    normed: bool = True,
) -> tuple[torch.Tensor, tuple[int, int, int], torch.Tensor]:
    """Build a regular 3D world-space grid of coordinates.

    Returns ``(coords, shape, affine)``:
    - ``coords``: ``(N, 3)`` tensor. Normalised to ``[-1, 1]`` when ``normed``,
      otherwise raw mm.
    - ``shape``: ``(Nx, Ny, Nz)``.
    - ``affine``: diagonal 4x4 with the requested spacing — enough to save a
      reconstructed atlas/volume as a NIfTI in its own frame.
    """
    if len(world_bbox) != 3 or len(spacing) != 3:
        raise ValueError("world_bbox and spacing must each have 3 elements")
    x = torch.arange(0, world_bbox[0], spacing[0], device=device)
    y = torch.arange(0, world_bbox[1], spacing[1], device=device)
    z = torch.arange(0, world_bbox[2], spacing[2], device=device)
    if normed:
        x = x / world_bbox[0] * 2 - 1
        y = y / world_bbox[1] * 2 - 1
        z = z / world_bbox[2] * 2 - 1
    gx, gy, gz = torch.meshgrid(x, y, z, indexing="ij")
    shape = (gx.shape[0], gx.shape[1], gx.shape[2])
    coords = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)
    affine = torch.diag(
        torch.tensor(
            [spacing[0], spacing[1], spacing[2], 1.0],
            dtype=torch.float32, device=device,
        )
    )
    return coords, shape, affine


def reconstruct_subject(
    state: TrainState,
    idx: int,
    *,
    spacing: Sequence[float],
    condition_override: Optional[torch.Tensor] = None,
    step_size: int = 100_000,
    renormalize_per_modality: bool = False,
    mask_reconstruction: bool = False,
) -> tuple[VolumeReconstruction, torch.Tensor]:
    """Reconstruct subject ``idx`` on a regular grid.

    Uses ``state.latents[idx]`` and ``state.transformations[idx]`` (if any),
    and either ``state.conditions[idx]`` (if set, val/test fit case) or the
    per-subject row conditions drawn from the dataset.

    ``mask_reconstruction=True`` zeros everything outside the largest connected
    foreground component of the predicted segmentation — matches the atlas
    masking path so ``infer`` / ``evaluate`` output mirrors atlas output when
    the training config has ``training.mask_reconstruction: true``. No-op when
    the reconstruction has no seg head or when label names are unavailable.

    Returns the ``VolumeReconstruction`` plus the 4x4 grid affine used.
    """
    decoder = state.decoder
    decoder.eval()
    device = state.device
    spec = state.dataset.dataset_spec
    world_bbox = spec.world_bbox

    coords, shape, affine = generate_world_grid(
        world_bbox, spacing, device=device, normed=True,
    )

    latent_vec = state.latents[idx : idx + 1]
    tfs = (
        state.transformations[idx : idx + 1]
        if state.transformations is not None else None
    )
    if condition_override is not None:
        cond = condition_override.to(device=device, dtype=torch.float32)
    elif state.conditions is not None:
        cond = state.conditions[idx].detach()
    else:
        row = state.dataset.df.iloc[idx].to_dict()
        cond = state.dataset.conditions.vector_for_row(row)
        cond = torch.from_numpy(cond).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        recon = decoder.predict_volume(
            coords, latent_vec, cond, shape,
            tfs=tfs,
            step_size=step_size,
            renormalize_per_modality=renormalize_per_modality,
        )
        if (
            mask_reconstruction
            and recon.seg_hard is not None
            and spec.label_names is not None
        ):
            recon = apply_largest_component_mask(recon, spec.label_names)
    return recon, affine


def save_reconstruction(
    recon: VolumeReconstruction,
    affine: Union[torch.Tensor, np.ndarray],
    output_dir: Union[str, Path],
    *,
    subject_id: str,
    intensity_modalities: Sequence[str],
    segmentation_modality: Optional[str] = None,
    epoch: int = 0,
    split: str = "train",
) -> list[Path]:
    """Write a reconstructed volume to disk as one NIfTI per modality.

    Files land at ``output_dir/<split>/<modality>_<subject_id>_ep=<epoch>.nii.gz``.
    Intensities are saved as float32, segmentation as int16. Returns the list
    of files written.
    """
    out_root = Path(output_dir) / split
    out_root.mkdir(parents=True, exist_ok=True)
    if isinstance(affine, torch.Tensor):
        affine_np = affine.detach().cpu().numpy().astype(np.float64)
    else:
        affine_np = np.asarray(affine, dtype=np.float64)

    written: list[Path] = []
    intensities = recon.intensities.detach().cpu().numpy()
    if intensities.ndim != 4 or intensities.shape[-1] != len(intensity_modalities):
        raise ValueError(
            f"intensities shape {intensities.shape} does not match "
            f"intensity_modalities {list(intensity_modalities)}"
        )
    for i, mod in enumerate(intensity_modalities):
        data = intensities[..., i].astype(np.float32)
        path = out_root / f"{mod}_{subject_id}_ep={epoch}.nii.gz"
        nib.save(nib.Nifti1Image(data, affine_np), str(path))
        written.append(path)

    if segmentation_modality is not None and recon.seg_hard is not None:
        seg = recon.seg_hard.detach().cpu().numpy().astype(np.int16)
        path = out_root / f"{segmentation_modality}_{subject_id}_ep={epoch}.nii.gz"
        nib.save(nib.Nifti1Image(seg, affine_np), str(path))
        written.append(path)

    return written


# === Latent-space analysis ===


@dataclass
class LatentAnalysisResult:
    condition_key: str
    method: str                       # 'nca' | 'regressor' | 'pls'
    mae: float
    std: float
    n_train: int
    n_val: int
    k: Optional[int] = None            # neighbours, for kNN-based methods
    n_components: Optional[int] = None # projection dim, for NCA/PLS

    def as_dict(self) -> dict:
        return {
            "condition": self.condition_key,
            "method": self.method,
            "mae": self.mae,
            "std": self.std,
            "k": self.k,
            "n_components": self.n_components,
            "n_train": self.n_train,
            "n_val": self.n_val,
        }


def predict_condition_nca(
    train_latents: torch.Tensor,
    val_latents: torch.Tensor,
    train_labels: torch.Tensor,
    val_labels: torch.Tensor,
    *,
    condition_key: str,
    k: int = 5,
    n_components: int = 2,
    random_state: int = 42,
) -> LatentAnalysisResult:
    """Predict a scalar condition from flattened latents via NCA + kNN classifier.

    Latents are flattened to ``(N, D)``, labels rounded to int, reduced to
    ``n_components`` with Neighborhood Components Analysis, then classified with
    k-nearest neighbours. MAE is reported in rounded-label units.

    Notes
    -----
    NCA optimizes classification accuracy, not MAE/MSE — works well for
    integer-valued or categorical targets, but rounds away half a unit of
    resolution. For continuous targets (e.g. age in weeks), prefer
    ``predict_condition_regressor``.
    """
    from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if len(train_latents) < 1 or len(val_latents) < 1:
        raise ValueError("need at least one train and one val sample")
    train_np = train_latents.detach().cpu().numpy().reshape(len(train_latents), -1)
    val_np = val_latents.detach().cpu().numpy().reshape(len(val_latents), -1)
    train_y = np.round(train_labels.detach().cpu().numpy()).astype(int)
    val_y = np.round(val_labels.detach().cpu().numpy()).astype(int)

    k_eff = min(k, len(train_np), len(val_np))
    if k_eff < 1:
        raise ValueError("need at least one train and one val sample")

    nca = make_pipeline(
        StandardScaler(),
        NeighborhoodComponentsAnalysis(n_components=n_components, random_state=random_state),
    )
    knn = KNeighborsClassifier(n_neighbors=k_eff)
    nca.fit(train_np, train_y)
    knn.fit(nca.transform(train_np), train_y)
    preds = knn.predict(nca.transform(val_np))
    err = np.abs(preds - val_y)
    return LatentAnalysisResult(
        condition_key=condition_key,
        method="nca",
        mae=float(err.mean()),
        std=float(err.std()),
        k=k_eff,
        n_components=n_components,
        n_train=len(train_np),
        n_val=len(val_np),
    )


def predict_condition_regressor(
    train_latents: torch.Tensor,
    val_latents: torch.Tensor,
    train_labels: torch.Tensor,
    val_labels: torch.Tensor,
    *,
    condition_key: str,
    k: int = 5,
) -> LatentAnalysisResult:
    """Predict a continuous condition via standardized + distance-weighted kNN regression.

    Unlike ``predict_condition_nca`` this keeps labels as floats (no rounding),
    averages neighbour values weighted by inverse distance, and reports MAE in
    the target's native units (e.g. weeks for ``scan_age``).
    """
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if len(train_latents) < 1 or len(val_latents) < 1:
        raise ValueError("need at least one train and one val sample")
    train_np = train_latents.detach().cpu().numpy().reshape(len(train_latents), -1)
    val_np = val_latents.detach().cpu().numpy().reshape(len(val_latents), -1)
    train_y = train_labels.detach().cpu().numpy().astype(np.float64)
    val_y = val_labels.detach().cpu().numpy().astype(np.float64)

    k_eff = min(k, len(train_np))
    if k_eff < 1:
        raise ValueError("need at least one train sample")

    pipe = make_pipeline(
        StandardScaler(),
        KNeighborsRegressor(n_neighbors=k_eff, weights="distance"),
    )
    pipe.fit(train_np, train_y)
    preds = pipe.predict(val_np)
    err = np.abs(preds - val_y)
    return LatentAnalysisResult(
        condition_key=condition_key,
        method="regressor",
        mae=float(err.mean()),
        std=float(err.std()),
        k=k_eff,
        n_train=len(train_np),
        n_val=len(val_np),
    )


def predict_condition_pls(
    train_latents: torch.Tensor,
    val_latents: torch.Tensor,
    train_labels: torch.Tensor,
    val_labels: torch.Tensor,
    *,
    condition_key: str,
    n_components: int = 5,
) -> LatentAnalysisResult:
    """Predict a continuous condition via Partial Least Squares regression.

    PLS finds latent components that maximize covariance between the features
    and the target — a supervised analogue of PCA. Usually a strong baseline
    for high-dim features with small N and a continuous target, and the natural
    upgrade if unsupervised PCA underperformed.

    ``n_components`` is clamped to ``min(n_components, n_train - 1, n_features)``.
    """
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if len(train_latents) < 2 or len(val_latents) < 1:
        raise ValueError("need at least two train and one val sample")
    train_np = train_latents.detach().cpu().numpy().reshape(len(train_latents), -1)
    val_np = val_latents.detach().cpu().numpy().reshape(len(val_latents), -1)
    train_y = train_labels.detach().cpu().numpy().astype(np.float64)
    val_y = val_labels.detach().cpu().numpy().astype(np.float64)

    n_comp_eff = min(n_components, len(train_np) - 1, train_np.shape[1])
    if n_comp_eff < 1:
        raise ValueError("need at least 2 train samples to fit PLS")

    pipe = make_pipeline(
        StandardScaler(),
        PLSRegression(n_components=n_comp_eff),
    )
    pipe.fit(train_np, train_y)
    preds = pipe.predict(val_np).reshape(-1)
    err = np.abs(preds - val_y)
    return LatentAnalysisResult(
        condition_key=condition_key,
        method="pls",
        mae=float(err.mean()),
        std=float(err.std()),
        n_components=n_comp_eff,
        n_train=len(train_np),
        n_val=len(val_np),
    )
