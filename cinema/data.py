"""
INR training/validation/test dataset for CINeMA.

The legacy ``args`` dict is gone — configuration is passed as typed objects
(``DatasetSpec``, ``ConstraintSet``, ``ConditionRegistry``). Condition
normalization is delegated wholesale to the registry, so the data module knows
nothing about ``cond_scale`` or ``min/max`` of any column.

``skip_segmentation=True`` opts out of the segmentation modality entirely:
- the seg path column is not loaded,
- coord sampling falls back to the union of non-zero voxels across intensity
  modalities,
- the ``values`` tensor returned by ``__getitem__`` has no segmentation column.

Use this for test subjects that lack ground-truth labels. The decoder still
emits its segmentation head; the fitter just drops the segmentation loss term.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torchio as tio
from torch.utils.data import Dataset

from .conditioning import ConditionRegistry
from .config import (
    ConstraintSet,
    DataAugmentationConfig,
    DatasetSpec,
)


def _normalize_intensities(values: np.ndarray, norm_type: str) -> np.ndarray:
    if values.shape[-1] == 0:
        return values
    values = np.clip(values, 0, None)
    if norm_type == "minmax":
        v_min = values.min(axis=0)
        v_max = values.max(axis=0)
        denom = np.where((v_max - v_min) > 0, v_max - v_min, 1.0)
        return (values - v_min) / denom
    if norm_type == "zscore":
        v_mean = values.mean(axis=0)
        v_std = values.std(axis=0)
        denom = np.where(v_std > 0, v_std, 1.0)
        return (values - v_mean) / denom
    raise ValueError(f"unknown intensity normalization: {norm_type!r}")


def _assert_coords_in_unit_bbox(coords: np.ndarray) -> None:
    if coords.size == 0:
        return
    mn, mx = coords.min(axis=0), coords.max(axis=0)
    if not ((mn >= -1.0).all() and (mx <= 1.0).all()):
        raise AssertionError(
            f"coords outside [-1, 1]: min={mn}, max={mx}; world_bbox is too small"
        )


def _squeeze_4d(nii: nib.Nifti1Image) -> nib.Nifti1Image:
    if nii.ndim == 4:
        return nib.Nifti1Image(np.squeeze(nii.get_fdata()), nii.affine)
    return nii


def _add_background_halo(
    seg: np.ndarray,
    label_names: list[str],
    halo_width: float = 1.5,
    bg_label_str: str = "BG",
) -> tuple[np.ndarray, np.ndarray]:
    """Add a thin background "halo" ring around the segmentation foreground.

    The ring is ``dilate(foreground) \\ foreground`` — the whole foreground
    enlarged, minus the original foreground. Ring voxels are relabelled to a
    background class so the decoder gets "just outside the brain = background +
    (masked) zero intensity" supervision, which is what yields a clean boundary
    at largest-connected-component masking time.

    The ring label is the explicit ``"BG"`` class if the scheme defines one,
    otherwise label index 0 (the conventional background slot). Works either way.

    Returns ``(seg_with_ring, sampling_mask)`` where ``sampling_mask`` is the
    foreground *plus* the ring (the set of voxels to sample), so the ring is
    sampled regardless of whether its label happens to be 0.
    """
    import scipy.ndimage as ndi

    fg = seg > 0
    dilated = ndi.gaussian_filter(fg.astype(np.float32), sigma=halo_width) > 0.001
    ring = dilated & ~fg
    bg_label = label_names.index(bg_label_str) if bg_label_str in label_names else 0
    out = seg.copy()
    out[ring] = bg_label
    return out, dilated


class MaskSubjectTransform(tio.Transform):
    """torchio transform: mask intensity images by the segmentation map."""

    def __init__(self, segmentation_key: str, **kwargs):
        super().__init__(**kwargs)
        self.segmentation_key = segmentation_key

    def apply_transform(self, subject):
        seg = subject[self.segmentation_key].data
        for k in subject.get_images_names():
            if k == self.segmentation_key:
                continue
            img = subject[k]
            img.data = img.data * (seg > 0)
        return subject


class Data(Dataset):
    """INR dataset over per-subject NIfTI files."""

    def __init__(
        self,
        dataset_spec: DatasetSpec,
        constraints: ConstraintSet,
        conditions: ConditionRegistry,
        tsv_file: Union[pd.DataFrame, str, Path],
        split: str,
        n_subjects: int,
        *,
        subject_ids: Optional[list[str]] = None,
        seed: int = 42,
        df_loaded: Optional[pd.DataFrame] = None,
        skip_segmentation: bool = False,
        mask_intensities_by_segmentation: bool = False,
        augmentation: Optional[DataAugmentationConfig] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        self.dataset_spec = dataset_spec
        self.constraints = constraints
        self.conditions = conditions
        self.split = split
        self.seed = seed
        self.n_subjects_target = n_subjects
        self.subject_ids = subject_ids
        self.skip_segmentation = skip_segmentation
        self.mask_intensities_by_segmentation = mask_intensities_by_segmentation
        self.output_dir = Path(output_dir) if output_dir is not None else None

        self._intensity_modalities = list(dataset_spec.intensity_modalities)
        self._seg_modality = (
            None if skip_segmentation else dataset_spec.segmentation_modality
        )
        self._modality_keys = list(self._intensity_modalities)
        if self._seg_modality is not None:
            self._modality_keys.append(self._seg_modality)
        self._world_bbox = np.asarray(dataset_spec.world_bbox, dtype=np.float64)

        self.tsv_file = (
            pd.read_csv(tsv_file, sep="\t")
            if isinstance(tsv_file, (str, Path))
            else tsv_file
        )
        self.df = (
            df_loaded.reset_index(drop=True)
            if df_loaded is not None
            else self._filter_dataframe(self.tsv_file)
        )
        self._validate_modality_consistency()
        self._init_data_augmentation(augmentation)

    def __len__(self) -> int:
        return len(self.df)

    def set_world_bbox(self, world_bbox) -> None:
        """Update the coordinate-normalisation box (used by `world_bbox: auto`,
        which resolves the box from this dataset after construction)."""
        self._world_bbox = np.asarray(world_bbox, dtype=np.float64)

    def __getitem__(self, idx: int):
        row_dict = self.df.iloc[idx].to_dict()
        modalities = self._load_modalities(row_dict)
        coords, values = self._load_coords_and_values(modalities)
        coords_t = torch.from_numpy(coords.astype(np.float32))
        values_t = torch.from_numpy(values.astype(np.float32))
        cond_vec = self._load_conditions(row_dict)
        conditions_t = cond_vec.unsqueeze(0).expand(coords_t.shape[0], -1)
        idx_df_t = torch.full(
            (coords_t.shape[0], 1), int(idx), dtype=torch.int32
        )
        return coords_t, values_t, conditions_t, idx_df_t

    def collate_fn(self, batch, shuffle: bool = True):
        coords = torch.cat([b[0] for b in batch], dim=0)
        values = torch.cat([b[1] for b in batch], dim=0)
        conditions = torch.cat([b[2] for b in batch], dim=0)
        idx_df = torch.cat([b[3] for b in batch], dim=0)
        if shuffle:
            perm = torch.randperm(coords.shape[0])
            coords = coords[perm]
            values = values[perm]
            conditions = conditions[perm]
            idx_df = idx_df[perm]
        return coords, values, conditions, idx_df

    # === Filtering / sampling ===

    def _filter_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._sample_subject_ids(df)
        df = self._remove_missing_modalities(df)
        df = self._check_constraints(df)
        df = self._sample_subjects(df)
        return df.reset_index(drop=True)

    def _sample_subject_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.subject_ids:
            df = df[df["subject_id"].isin(self.subject_ids)]
        return df.reset_index(drop=True)

    def _remove_missing_modalities(self, df: pd.DataFrame) -> pd.DataFrame:
        keep = np.ones(len(df), dtype=bool)
        for mk in self._modality_keys:
            if mk not in df.columns:
                continue
            keep &= (df[mk].notnull() & (df[mk] != "")).to_numpy()
        return df[keep].reset_index(drop=True)

    def _check_constraints(self, df: pd.DataFrame) -> pd.DataFrame:
        keep = np.ones(len(df), dtype=bool)
        for cs in self.constraints:
            if cs.name not in df.columns:
                continue
            if cs.is_numeric:
                col = df[cs.name].to_numpy()
                keep &= (col >= cs.min) & (col <= cs.max)
            else:
                keep &= df[cs.name].isin(cs.values).to_numpy()
        return df[keep].reset_index(drop=True)

    def _sample_subjects(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.n_subjects_target > len(df):
            self.n_subjects_target = len(df)
        sampling_constraints = self.constraints.with_sampling()
        if not sampling_constraints:
            if self.n_subjects_target == 0:
                return df.iloc[:0].copy()
            return df.sample(
                n=self.n_subjects_target, random_state=self.seed
            ).reset_index(drop=True)
        df_sampled = self._shallow_sampling(
            df, sampling_constraints[0], self.n_subjects_target
        )
        if len(df_sampled) > self.n_subjects_target:
            df_sampled = df_sampled.sample(
                n=self.n_subjects_target, random_state=self.seed
            ).reset_index(drop=True)
        if self.output_dir is not None:
            for cs in sampling_constraints:
                self._save_histogram(df_sampled, cs)
        return df_sampled

    def _shallow_sampling(
        self, df: pd.DataFrame, cs, max_n: int
    ) -> pd.DataFrame:
        c_min, c_max = cs.min, cs.max
        bins = cs.sampling.bins or max(1, int(round(c_max - c_min)))
        edges = np.linspace(c_min, c_max, bins + 1)
        edges[-1] += 1e-6
        values = df[cs.name].to_numpy()
        bin_idx = np.digitize(values, edges) - 1
        drawn = [0] * bins
        drained: set[int] = set()
        # bins outside the df are immediately drained
        for i in range(bins):
            if (bin_idx == i).sum() == 0:
                drained.add(i)
        while sum(drawn) < max_n and len(drained) < bins:
            remaining = bins - len(drained)
            need = max_n - sum(drawn)
            per_bin = math.ceil(need / remaining)
            progressed = False
            for i in range(bins):
                if i in drained:
                    continue
                bin_size = int((bin_idx == i).sum())
                avail = bin_size - drawn[i]
                take = min(per_bin, avail)
                if take > 0:
                    drawn[i] += take
                    progressed = True
                if drawn[i] >= bin_size:
                    drained.add(i)
            if not progressed:
                break
        out = []
        for i, n in enumerate(drawn):
            if n > 0:
                bin_df = df[bin_idx == i]
                out.append(bin_df.sample(n=n, random_state=self.seed))
        if not out:
            return df.iloc[:0].copy()
        return pd.concat(out, ignore_index=True)

    def _save_histogram(self, df: pd.DataFrame, cs) -> None:
        col = df[cs.name].dropna().to_numpy()
        if col.size == 0:
            return
        c_min, c_max = cs.min, cs.max
        bins = cs.sampling.bins or max(1, int(round(c_max - c_min)))
        edges = np.linspace(c_min, c_max, bins)
        edges[-1] += 1e-9
        counts, edges = np.histogram(col, bins=edges)
        plt.figure()
        centers = 0.5 * (edges[:-1] + edges[1:])
        plt.bar(centers, counts, width=(edges[1] - edges[0]) * 0.9)
        plt.title(f"hist {cs.name} ({cs.sampling.type}) – {self.split}")
        plt.xlabel(cs.name)
        plt.ylabel("count")
        out = self.output_dir / f"hist_{cs.name}_{cs.sampling.type}_{self.split}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=100, bbox_inches="tight")
        plt.close()

    # === Per-subject loading ===

    def _validate_modality_consistency(self) -> None:
        for _, row in self.df.iterrows():
            sid = row.get("subject_id", "<unknown>")
            shapes, affines = [], []
            for mk in self._modality_keys:
                nii = nib.load(row[mk])
                shapes.append(nii.shape[:3] if nii.ndim == 4 else nii.shape)
                affines.append(nii.affine)
            if len(set(shapes)) != 1:
                raise ValueError(
                    f"subject {sid}: modalities have different shapes: "
                    f"{dict(zip(self._modality_keys, shapes))}"
                )
            ref = affines[0]
            for mk, aff in zip(self._modality_keys[1:], affines[1:]):
                if not np.allclose(aff, ref, atol=1e-6):
                    raise ValueError(
                        f"subject {sid}: modality {mk} affine differs from "
                        f"{self._modality_keys[0]} affine."
                    )

    def _load_modalities(self, row_dict: dict) -> dict:
        mods: dict[str, nib.Nifti1Image] = {}
        for mk in self._modality_keys:
            if mk not in row_dict:
                raise ValueError(f"modality {mk!r} not in dataframe row")
            path = row_dict[mk]
            if not path:
                raise ValueError(f"modality {mk!r} path is empty")
            mods[mk] = _squeeze_4d(nib.load(path))
        if (
            self.mask_intensities_by_segmentation
            and self._seg_modality is not None
        ):
            seg_data = mods[self._seg_modality].get_fdata()
            mask = (seg_data > 0).astype(np.float32)
            for mk in self._intensity_modalities:
                d = mods[mk].get_fdata() * mask
                mods[mk] = nib.Nifti1Image(d, mods[mk].affine)
            # The background halo is added later, on the augmented seg, in
            # _load_coords_and_values (so the ring follows the augmented
            # foreground and is sampled via an explicit dilated mask).
        return mods

    def _load_coords_and_values(
        self, modalities: dict
    ) -> tuple[np.ndarray, np.ndarray]:
        modalities_data = self._augment(modalities)
        affine = modalities[self._modality_keys[0]].affine

        if self._seg_modality is not None:
            seg = modalities_data[self._seg_modality]
            if (
                self.mask_intensities_by_segmentation
                and self.dataset_spec.label_names is not None
            ):
                # Add the background halo and sample foreground + ring.
                seg, mask = _add_background_halo(seg, self.dataset_spec.label_names)
                modalities_data[self._seg_modality] = seg
            else:
                mask = seg > 0
        else:
            mask = np.zeros_like(
                modalities_data[self._intensity_modalities[0]], dtype=bool
            )
            for mk in self._intensity_modalities:
                mask |= modalities_data[mk] > 0
        c_nz = np.argwhere(mask)
        values = np.stack(
            [
                modalities_data[mk][c_nz[:, 0], c_nz[:, 1], c_nz[:, 2]].ravel()
                for mk in self._modality_keys
            ],
            axis=-1,
        )
        c_world = nib.affines.apply_affine(affine, c_nz)
        center = np.mean(c_world, axis=0)
        coords = (c_world - center) / (self._world_bbox / 2.0)
        _assert_coords_in_unit_bbox(coords)

        n_int = len(self._intensity_modalities)
        if n_int > 0:
            int_vals = values[:, :n_int].astype(np.float32)
            int_vals = _normalize_intensities(
                int_vals, self.dataset_spec.normalize_intensities
            )
            values = values.astype(np.float32, copy=True)
            values[:, :n_int] = int_vals
        return coords, values

    def _load_conditions(self, row_dict: dict) -> torch.Tensor:
        vec = self.conditions.vector_for_row(row_dict)
        return torch.from_numpy(vec)

    def get_condition_values(
        self,
        condition_key: str,
        normed: bool = True,
        device=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """All subjects' values for one condition (used by atlas / latent analysis)."""
        spec = self.conditions[condition_key]
        raw = self.df[condition_key].to_numpy()
        if normed:
            out = np.empty(len(raw), dtype=np.float32)
            for i, v in enumerate(raw):
                out[i] = spec.normalize(v, context=self.df.iloc[i].to_dict())
        else:
            out = raw.astype(np.float32)
        return (
            torch.tensor(out, device=device),
            torch.arange(len(self.df), device=device),
        )

    # === Augmentation ===

    def _init_data_augmentation(
        self, aug: Optional[DataAugmentationConfig]
    ) -> None:
        self._augment_active = bool(
            aug is not None and aug.activate and self.split == "train"
        )
        self._tio_transform = None
        if not self._augment_active:
            return
        d = aug.raw
        transforms = []
        if d.get("augment_deformation", {}).get("p", 0) > 0:
            a = d["augment_deformation"]
            transforms.append(
                tio.RandomElasticDeformation(
                    p=a["p"],
                    num_control_points=a["num_control_points"],
                    max_displacement=a["max_displacement"],
                )
            )
        if d.get("augment_motion", {}).get("p", 0) > 0:
            a = d["augment_motion"]
            transforms.append(
                tio.RandomMotion(
                    p=a["p"],
                    degrees=a["degrees"],
                    translation=a["translation"],
                    num_transforms=a["num_transforms"],
                )
            )
        if d.get("augment_noise", {}).get("p", 0) > 0:
            a = d["augment_noise"]
            transforms.append(tio.RandomNoise(p=a["p"], mean=a["mean"], std=a["std"]))
        if d.get("augment_biasfield", {}).get("p", 0) > 0:
            a = d["augment_biasfield"]
            transforms.append(tio.RandomBiasField(p=a["p"], coefficients=a["coeff"]))
        if d.get("augment_gamma", {}).get("p", 0) > 0:
            a = d["augment_gamma"]
            transforms.append(tio.RandomGamma(p=a["p"], log_gamma=a["log_gamma"]))
        if self._seg_modality is not None:
            transforms.append(MaskSubjectTransform(self._seg_modality))
        self._tio_transform = tio.Compose(transforms) if transforms else None

    def _augment(self, modalities: dict) -> dict:
        if not self._augment_active or self._tio_transform is None:
            return {mk: m.get_fdata() for mk, m in modalities.items()}
        sub = {}
        for mk, m in modalities.items():
            data = torch.from_numpy(m.get_fdata()).type(torch.float32).unsqueeze(0)
            if mk == self._seg_modality:
                sub[mk] = tio.LabelMap(tensor=data, affine=m.affine)
            else:
                sub[mk] = tio.ScalarImage(tensor=data, affine=m.affine)
        sub = tio.Subject(sub)
        sub = self._tio_transform(sub)
        return {mk: sub[mk].data.squeeze().numpy() for mk in modalities}
