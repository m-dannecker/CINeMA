"""
Atlas generation for CINeMA.

An atlas is a synthetic reconstruction at a specific (age × condition-combo)
target. Given a trained ``TrainState`` (decoder + per-subject latents + the
training dataset's condition registry), the steps per atlas point are:

1. Compute a "mean latent" for the target age by Gaussian-weighting training
   subjects' latents by distance to the target age in *normalised* condition
   space. Width of the Gaussian is derived from the recipe's ``gaussian_span``
   (in raw units, e.g. weeks of scan_age) by sampling the condition's
   normalisation around the target — this handles non-linear normalisations
   (age-relative, etc.) correctly.
2. For each condition combination in the recipe, build a decoder condition
   vector (temporal value + recipe-supplied values for the other enabled
   conditions), then decode the mean latent on a regular world grid with
   ``tfs=None`` — atlases are not pose-anchored to any individual subject.
3. Save one 4D NIfTI per (modality × condition-combo), with the temporal
   axis stacked along the last dim so the atlas can be scrubbed through in
   ITK-SNAP / fsleyes without loading many files. An ``ages.json`` sidecar
   in the atlas output directory records the ordered age list.

Functional module, no class. Reuses ``evaluator.generate_world_grid`` for grid
construction and ``INRDecoder.predict_volume`` for reconstruction.
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Optional, Sequence, Union

import nibabel as nib
import numpy as np
import torch

from .conditioning import AgeRelativeNormalization
from .config import AtlasRecipe
from .evaluator import (
    apply_intensity_floor,
    apply_largest_component_mask,
    generate_world_grid,
)
from .state import TrainState


def mean_latent_for(
    state: TrainState,
    *,
    condition_key: str,
    target_value_raw: float,
    gaussian_span: float,
    n_max: int = 100,
    context: Optional[dict] = None,
) -> torch.Tensor:
    """Gaussian-weighted mean of training latents around a target condition value.

    ``gaussian_span`` is in the condition's raw units (e.g. weeks of scan_age)
    and is taken to cover ≈ ±2 standard deviations. Sigma is computed in
    normalised condition space by sampling ``target ± span/2`` through the
    condition's normalisation — this handles non-linear mappings correctly.

    Only the top ``n_max`` subjects by weight are kept; the rest are zeroed.
    Weights are then L1-normalised before the weighted sum.
    """
    registry = state.dataset.conditions
    spec = registry[condition_key]
    target_normed = float(spec.normalize(target_value_raw, context=context))
    lo = float(spec.normalize(target_value_raw - gaussian_span / 2, context=context))
    hi = float(spec.normalize(target_value_raw + gaussian_span / 2, context=context))
    normed_span = abs(hi - lo)
    if normed_span <= 0:
        raise ValueError(
            f"normalisation collapses {condition_key} span {gaussian_span} to 0 — "
            "can't compute a mean latent"
        )
    sigma = 0.5 * normed_span

    latents = state.latents.detach()
    values, _ = state.dataset.get_condition_values(
        condition_key, normed=True, device=latents.device,
    )
    if values.shape[0] != latents.shape[0]:
        raise RuntimeError(
            f"latents has {latents.shape[0]} rows but dataset yields "
            f"{values.shape[0]} condition values"
        )

    diff = values - target_normed
    weights = torch.exp(-(diff ** 2) / (2 * sigma ** 2))
    k = min(n_max, weights.numel())
    if k < weights.numel():
        top_idx = torch.topk(weights, k, largest=True).indices
        mask = torch.zeros_like(weights, dtype=torch.bool)
        mask[top_idx] = True
        weights = torch.where(mask, weights, torch.zeros_like(weights))
    w_sum = weights.sum()
    if w_sum <= 0:
        raise RuntimeError(
            "all weights are zero — gaussian_span may be too narrow for this target"
        )
    weights = weights / w_sum
    extra_dims = (1,) * (latents.ndim - 1)
    return (latents * weights.reshape(-1, *extra_dims)).sum(dim=0, keepdim=True)


def generate_atlas(
    state: TrainState,
    recipe: AtlasRecipe,
    *,
    output_dir: Union[str, Path],
    temporal_condition: str = "scan_age",
    epoch: int = 0,
    renormalize_per_modality: bool = False,
    step_size: int = 100_000,
    mask_open_radius: int = 0,
    intensity_floor: float = 0.0,
) -> list[Path]:
    """Generate all atlases in ``recipe`` and write them to ``output_dir/atlas/``.

    One 4D NIfTI per ``(modality × condition-combo)`` is written, with
    ``recipe.ages`` stacked along the last axis so the temporal progression
    can be scrubbed through as a single file in ITK-SNAP / fsleyes. Files are
    named ``<modality>_cond=<i>_<temporal>=<min>-<max>_ep=<epoch>.nii.gz``
    (``cond=i`` indexes into the cartesian product of ``recipe.conditions``,
    defaulting to ``cond=0`` when none are declared beyond the temporal axis).
    Segmentation (int16) is emitted alongside intensities when available,
    using the largest-connected-component mask when
    ``recipe.mask_reconstruction`` is on. An ``ages.json`` sidecar records the
    ordered age list so the 4th axis is interpretable later.
    """
    decoder = state.decoder
    decoder.eval()
    device = state.device
    registry = state.dataset.conditions
    enabled = registry.enabled_specs()
    spec = state.dataset.dataset_spec

    if temporal_condition not in registry:
        raise ValueError(
            f"temporal_condition {temporal_condition!r} not in condition registry"
        )

    coords, shape, grid_affine = generate_world_grid(
        spec.world_bbox, recipe.spacing, device=device, normed=True,
    )
    combos = _condition_combinations(recipe.conditions)

    atlas_root = Path(output_dir) / "atlas"
    atlas_root.mkdir(parents=True, exist_ok=True)

    # Per-combo buffers: list of (X, Y, Z, C_int) arrays and (X, Y, Z) seg arrays
    # per age, stacked later along the new trailing time axis.
    intensity_frames: list[list[np.ndarray]] = [[] for _ in combos]
    seg_frames: list[list[np.ndarray]] = [[] for _ in combos]

    with torch.no_grad():
        for age in recipe.ages:
            mean_lat = mean_latent_for(
                state,
                condition_key=temporal_condition,
                target_value_raw=age,
                gaussian_span=recipe.gaussian_span,
                n_max=recipe.n_max,
                context={temporal_condition: age},
            )
            for combo_idx, combo in enumerate(combos):
                ctx = {temporal_condition: age, **combo}
                cond_vec = _build_cond_vector(
                    enabled, temporal_condition, age, combo, ctx, device,
                )
                recon = decoder.predict_volume(
                    coords, mean_lat, cond_vec, shape,
                    tfs=None,
                    step_size=step_size,
                    renormalize_per_modality=renormalize_per_modality,
                )
                recon = apply_intensity_floor(recon, intensity_floor)
                if (
                    recipe.mask_reconstruction
                    and recon.seg_hard is not None
                    and spec.label_names is not None
                ):
                    recon = apply_largest_component_mask(
                        recon, spec.label_names, open_radius=mask_open_radius,
                    )
                intensity_frames[combo_idx].append(
                    recon.intensities.detach().cpu().numpy().astype(np.float32)
                )
                if recon.seg_hard is not None:
                    seg_frames[combo_idx].append(
                        recon.seg_hard.detach().cpu().numpy().astype(np.int16)
                    )

    written: list[Path] = []
    for combo_idx in range(len(combos)):
        written.extend(_save_atlas_combo(
            intensity_frames[combo_idx],
            seg_frames[combo_idx],
            grid_affine,
            atlas_root,
            ages=list(recipe.ages),
            combo=combos[combo_idx],
            combo_idx=combo_idx,
            epoch=epoch,
            intensity_modalities=spec.intensity_modalities,
            segmentation_modality=spec.segmentation_modality,
            temporal_condition=temporal_condition,
        ))

    ages_sidecar = atlas_root / "ages.json"
    ages_sidecar.write_text(json.dumps(list(recipe.ages), indent=2))

    # Optional tissue growth-curve overlay (validation against training data).
    # The atlas files are already written above, so a plotting hiccup here must
    # not lose them — log and continue.
    gc = getattr(recipe, "growth_curves", None)
    if gc is not None and gc.enabled:
        import traceback

        from .growth_curves import generate_growth_curves
        try:
            written.extend(generate_growth_curves(
                state, recipe,
                temporal_condition=temporal_condition,
                output_dir=output_dir,
                combos=combos,
                seg_frames=seg_frames,
                epoch=epoch,
            ))
        except Exception:
            print("[growth_curves] generation failed:\n" + traceback.format_exc())

    return written


# === Internals ===


def _condition_combinations(
    raw_conditions: dict[str, list[float]],
) -> list[dict[str, float]]:
    """Cartesian product of ``{key: [values...]}`` → list of dicts.

    Returns ``[{}]`` (one empty combo) when no extra conditions are declared,
    so the age-only atlas path still runs through the same loop.
    """
    if not raw_conditions:
        return [{}]
    keys = list(raw_conditions.keys())
    lists = [raw_conditions[k] for k in keys]
    return [dict(zip(keys, vals)) for vals in product(*lists)]


def combo_tag(combo: dict, combo_idx: int = 0, sep: str = "_") -> str:
    """Readable tag for a condition combination, e.g. ``ExamType_num=-1.5``.

    Used for atlas filenames (``sep="_"``) and growth-curve marker labels
    (``sep=", "``). Falls back to ``cond=<i>`` for an empty (age-only) combo so
    files stay distinguishable.
    """
    if not combo:
        return f"cond={combo_idx}"
    return sep.join(f"{k}={v:g}" for k, v in combo.items())


def _build_cond_vector(
    enabled_specs: Sequence, temporal_condition: str,
    age: float, combo: dict, context: dict, device: str,
) -> torch.Tensor:
    vec: list[float] = []
    for spec in enabled_specs:
        if spec.name == temporal_condition:
            vec.append(float(spec.normalize(age, context=context)))
        elif spec.name in combo:
            value = combo[spec.name]
            # For age-relative conditions the recipe supplies a z-score (sigma
            # units, age-invariant) rather than a raw physical value, since the
            # raw value's meaning drifts with age. Everything else is raw units.
            if isinstance(spec.normalization, AgeRelativeNormalization):
                code = float(spec.normalization.from_z(value)) * spec.cond_scale
            else:
                code = float(spec.normalize(value, context=context))
            vec.append(code)
        else:
            raise ValueError(
                f"atlas recipe does not supply a value for decoder condition "
                f"{spec.name!r} — add it under recipe.conditions"
            )
    return torch.tensor(vec, dtype=torch.float32, device=device)


def _save_atlas_combo(
    intensity_per_age: list[np.ndarray],
    seg_per_age: list[np.ndarray],
    affine: Union[torch.Tensor, np.ndarray],
    atlas_root: Path,
    *,
    ages: list[float],
    combo: dict,
    combo_idx: int,
    epoch: int,
    intensity_modalities: Sequence[str],
    segmentation_modality: Optional[str],
    temporal_condition: str,
) -> list[Path]:
    if isinstance(affine, torch.Tensor):
        affine_np = affine.detach().cpu().numpy().astype(np.float64)
    else:
        affine_np = np.asarray(affine, dtype=np.float64)
    written: list[Path] = []
    age_range = f"{min(ages):g}-{max(ages):g}" if len(ages) > 1 else f"{ages[0]:g}"
    tag = combo_tag(combo, combo_idx, sep="_")  # e.g. ExamType_num=-1.5

    # intensity_per_age[t] has shape (X, Y, Z, C_int). Stack along new trailing
    # time axis → (X, Y, Z, C_int, T), then per-modality → (X, Y, Z, T).
    stacked_int = np.stack(intensity_per_age, axis=-1)  # (X, Y, Z, C, T)
    for c, mod in enumerate(intensity_modalities):
        vol_4d = np.ascontiguousarray(stacked_int[..., c, :])
        path = atlas_root / (
            f"{mod}_{tag}_{temporal_condition}={age_range}_ep={epoch}.nii.gz"
        )
        nib.save(nib.Nifti1Image(vol_4d, affine_np), str(path))
        written.append(path)

    if segmentation_modality is not None and seg_per_age:
        stacked_seg = np.stack(seg_per_age, axis=-1)  # (X, Y, Z, T)
        path = atlas_root / (
            f"{segmentation_modality}_{tag}"
            f"_{temporal_condition}={age_range}_ep={epoch}.nii.gz"
        )
        nib.save(nib.Nifti1Image(stacked_seg, affine_np), str(path))
        written.append(path)

    return written
