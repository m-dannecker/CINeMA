"""
Optional ANTs-based metrics for CINeMA.

Registers a predicted volume to each ground-truth modality with ANTs, crops to
the ground-truth bounding box, and reports PSNR/SSIM for intensities and Dice
for segmentation. Kept in a separate module so ``cinema.evaluator`` stays
import-cheap — ``ants`` only loads when the caller imports this file.

The PSNR/SSIM implementations deliberately follow the original scikit-image
conventions (``data_range=1`` after min-max normalisation) so numbers line up
with the legacy pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import nibabel as nib
import numpy as np


@dataclass
class SubjectMetrics:
    subject_id: str
    psnr: list[float] = field(default_factory=list)          # one per intensity modality
    ssim: list[float] = field(default_factory=list)
    dice: Optional[dict[str, float]] = None                   # per-label + "mean"

    def as_dict(self) -> dict:
        return {
            "subject": self.subject_id,
            "psnr": self.psnr,
            "ssim": self.ssim,
            "dice": self.dice,
        }


def compute_subject_metrics(
    subject_id: str,
    pred_intensities: np.ndarray,           # (X, Y, Z, n_intensity)
    pred_seg: Optional[np.ndarray],         # (X, Y, Z) int labels, or None
    pred_affine: np.ndarray,
    ref_paths: dict[str, str],              # modality_name -> NIfTI path
    *,
    intensity_modalities: Sequence[str],
    segmentation_modality: Optional[str] = None,
    label_names: Optional[Sequence[str]] = None,
    reg_type: str = "Rigid",
    bg_label_str: str = "BG",
) -> SubjectMetrics:
    """Register prediction to each reference modality and score it.

    A single transform is estimated from the first intensity modality and reused
    for the rest (matching the legacy ``reg_imgs`` behaviour). The segmentation
    reference is used as a mask during registration when available.
    """
    import ants

    metrics = SubjectMetrics(subject_id=subject_id)
    mask_ref_nii = None
    if segmentation_modality is not None and segmentation_modality in ref_paths:
        mask_ref_nii = nib.load(ref_paths[segmentation_modality])

    mytx = None
    for i, mod in enumerate(intensity_modalities):
        pred_nii = nib.Nifti1Image(pred_intensities[..., i].astype(np.float32), pred_affine)
        ref_nii = nib.load(ref_paths[mod])
        pred_reg, ref_reg, mytx = _register(
            fix_nii=pred_nii, mov_nii=ref_nii, mask_mov_nii=mask_ref_nii,
            mytx=mytx, reg_type=reg_type, is_seg=False,
        )
        pred_crop, ref_crop = _crop_to_ref_bbox(pred_reg, ref_reg)
        metrics.psnr.append(_psnr(pred_crop, ref_crop, data_range=1.0))
        metrics.ssim.append(_ssim(pred_crop, ref_crop, data_range=1.0))

    if pred_seg is not None and segmentation_modality is not None:
        pred_seg_nii = nib.Nifti1Image(pred_seg.astype(np.int16), pred_affine)
        ref_seg_nii = nib.load(ref_paths[segmentation_modality])
        pred_seg_reg, ref_seg_reg, _ = _register(
            fix_nii=pred_seg_nii, mov_nii=ref_seg_nii, mask_mov_nii=mask_ref_nii,
            mytx=mytx, reg_type=reg_type, is_seg=True,
        )
        pred_crop, ref_crop = _crop_to_ref_bbox(pred_seg_reg, ref_seg_reg)
        bg_label = _bg_label_id(label_names, bg_label_str)
        metrics.dice = _dice(
            pred_crop, ref_crop,
            bg_label=bg_label,
            label_names=label_names,
        )

    return metrics


# === Internals ===


def _register(
    *,
    fix_nii: nib.Nifti1Image,
    mov_nii: nib.Nifti1Image,
    mask_mov_nii: Optional[nib.Nifti1Image],
    mytx,
    reg_type: str,
    is_seg: bool,
):
    import ants

    fix = ants.from_nibabel(fix_nii)
    mov = ants.from_nibabel(mov_nii)
    mask_mov = ants.from_nibabel(mask_mov_nii) > 0 if mask_mov_nii is not None else None
    interp = "genericLabel" if is_seg else "linear"

    if mytx is None:
        mov_eff = mov * mask_mov if mask_mov is not None else mov
        mytx = ants.registration(fix, mov_eff, type_of_transform=reg_type)
        mov_reg = mytx["warpedmovout"]
    else:
        mov_reg = ants.apply_transforms(
            fix, mov, mytx["fwdtransforms"], interpolator=interp,
        )
    if mask_mov is not None:
        mask_reg = ants.apply_transforms(
            fix, mask_mov, mytx["fwdtransforms"], interpolator="genericLabel",
        )
        fix = fix * mask_reg
        mov_reg = mov_reg * mask_reg
    if not is_seg:
        mov_reg = _minmax(mov_reg)
        fix = _minmax(fix)
    return ants.to_nibabel(fix), ants.to_nibabel(mov_reg), mytx


def _minmax(img):
    return (img - img.min()) / (img.max() - img.min() + 1e-12)


def _crop_to_ref_bbox(
    pred_nii: nib.Nifti1Image, ref_nii: nib.Nifti1Image,
) -> tuple[np.ndarray, np.ndarray]:
    ref = ref_nii.get_fdata()
    nz = np.argwhere(ref != 0)
    if nz.size == 0:
        return pred_nii.get_fdata(), ref
    mn = nz.min(axis=0)
    mx = nz.max(axis=0) + 1
    s = tuple(slice(mn[d], mx[d]) for d in range(3))
    return pred_nii.get_fdata()[s], ref[s]


def _bg_label_id(label_names: Optional[Sequence[str]], bg_label_str: str) -> int:
    if label_names is None:
        return 0
    return label_names.index(bg_label_str) if bg_label_str in label_names else 0


def _psnr(pred: np.ndarray, ref: np.ndarray, *, data_range: float) -> float:
    from skimage.metrics import peak_signal_noise_ratio
    return float(peak_signal_noise_ratio(ref, pred, data_range=data_range))


def _ssim(pred: np.ndarray, ref: np.ndarray, *, data_range: float) -> float:
    from skimage.metrics import structural_similarity
    return float(structural_similarity(ref, pred, data_range=data_range))


def _dice(
    pred: np.ndarray,
    ref: np.ndarray,
    *,
    bg_label: int,
    label_names: Optional[Sequence[str]] = None,
) -> dict[str, float]:
    if pred.shape != ref.shape:
        raise ValueError(f"pred shape {pred.shape} != ref shape {ref.shape}")
    # Cast to int to avoid float comparison issues after ANTs round-trip
    pred = np.round(pred).astype(np.int32)
    ref = np.round(ref).astype(np.int32)
    labels = np.union1d(np.unique(pred), np.unique(ref))
    labels = labels[(labels != 0) & (labels != bg_label)]
    if labels.size == 0:
        return {"mean": 1.0}
    scores: dict[str, float] = {}
    for lab in labels:
        pm = pred == lab
        rm = ref == lab
        inter = np.count_nonzero(pm & rm)
        denom = np.count_nonzero(pm) + np.count_nonzero(rm)
        score = 1.0 if denom == 0 else 2.0 * inter / denom
        key = label_names[lab] if label_names is not None and lab < len(label_names) else str(lab)
        scores[key] = float(score)
    scores["mean"] = float(np.mean(list(scores.values())))
    return scores
