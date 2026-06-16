#!/usr/bin/env python3
"""Tissue-volume growth curves vs age, grouped by a categorical variable.

Reads per-subject segmentation NIfTIs listed in a CINeMA subjects table, computes
per-tissue volumes (mm^3), and fits smooth mean +/- SD growth curves of each
tissue vs age, one curve per group (e.g. ExamType = PRE_OP / POST_OP). The fitted
curves + per-subject volumes are saved as artifacts, and one plot per tissue is
written for visual inspection.

Intended use: an empirical, data-derived reference to sanity-check generated
atlases (e.g. does the atlas total brain volume track the real cohort at each
age?). The optional ``--atlas-dir`` mode measures per-tissue volumes from the 4D
atlas segmentation NIfTIs and overlays them on the empirical bands.

The tissue -> label-index mapping is taken from the dataset YAML's
``segmentation_classes.label_names`` (index i == label value i), which is
cohort-specific (different label schemes may assign tissues to different indices).

Example:
    python scripts/tissue_growth_curves.py \
        --dataset configs/datasets/example.yaml \
        --age-col GA_MRI --group-col ExamType --groups PRE_OP POST_OP \
        --rating-col rating_svr --rating-min 1 \
        --output output/growth_curves/example

With atlas overlay (markers labelled by their condition value, read from the
atlas filenames):
    python scripts/tissue_growth_curves.py ... --atlas-dir output/<run>/atlas
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root for `cinema`
from cinema.config import load_dataset_config
from cinema.data import read_subjects_table

TOTAL_BRAIN = "TotalBrain"


# --- segmentation -> per-label volumes ---------------------------------------

def _label_volumes(path: str, n_labels: int) -> np.ndarray:
    """Per-label volume (mm^3) for one segmentation file, length ``n_labels``.

    Uses ``np.asarray(img.dataobj)`` (integer labels, no float64 upcast) and a
    bincount over voxel labels times the voxel volume — same loading choice as
    scripts/extract_lv_condition.py.
    """
    img = nib.load(path)
    data = np.asarray(img.dataobj)
    voxvol = float(np.prod(img.header.get_zooms()[:3]))
    counts = np.bincount(data.astype(np.int64).ravel(), minlength=n_labels)
    return counts[:n_labels].astype(np.float64) * voxvol


def _worker(args: tuple[int, str, int]) -> tuple[int, Optional[np.ndarray]]:
    idx, path, n_labels = args
    try:
        return idx, _label_volumes(path, n_labels)
    except Exception as e:  # unreadable/corrupt -> drop the row, keep going
        print(f"  [warn] row {idx}: failed to read {path}: {e}")
        return idx, None


# --- growth-curve fit --------------------------------------------------------

def fit_curve(ages: np.ndarray, vals: np.ndarray, kernel_sigma: float,
              n_grid: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kernel-regress mu(age) and sigma(age) over an age grid (Gaussian kernel).

    Replicated from scripts/compute_lv_zscore.py (which mirrors
    cinema.conditioning.AgeRelativeNormalization.fit) so the curve math matches
    the rest of the repo.
    """
    grid = np.linspace(ages.min(), ages.max(), n_grid)
    diffs = (ages[None, :] - grid[:, None]) / kernel_sigma
    w = np.exp(-0.5 * diffs * diffs)
    w = w / np.clip(w.sum(axis=1, keepdims=True), 1e-12, None)
    mu = (w * vals[None, :]).sum(axis=1)
    var = (w * (vals[None, :] - mu[:, None]) ** 2).sum(axis=1)
    sd = np.sqrt(var + 1e-8)
    return grid, mu, sd


# --- atlas overlay -----------------------------------------------------------

def _atlas_frame_volumes(path: str, n_labels: int) -> np.ndarray:
    """Per-tissue volumes for each time frame of a 4D atlas seg (X,Y,Z,T).

    Returns an array of shape (T, n_labels) in mm^3.
    """
    img = nib.load(path)
    data = np.asarray(img.dataobj)
    voxvol = float(np.prod(img.header.get_zooms()[:3]))
    if data.ndim == 3:
        data = data[..., None]
    n_t = data.shape[-1]
    out = np.zeros((n_t, n_labels), dtype=np.float64)
    for t in range(n_t):
        counts = np.bincount(data[..., t].astype(np.int64).ravel(), minlength=n_labels)
        out[t] = counts[:n_labels] * voxvol
    return out


# --- plotting ----------------------------------------------------------------

def _plot_tissue(tissue: str, per_group: dict, raw: dict,
                 atlas: dict, out_dir: Path) -> Path:
    """One figure per tissue: mu(age) line + +/-SD band + raw scatter per group,
    with optional atlas-measured points overlaid."""
    plt.figure(figsize=(7, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for gi, (group, curve) in enumerate(per_group.items()):
        c = colors[gi % len(colors)]
        grid, mu, sd = curve["age_grid"], curve["mu_grid"], curve["sigma_grid"]
        plt.plot(grid, mu, color=c, label=f"{group} (n={curve['n']})")
        plt.fill_between(grid, mu - sd, mu + sd, color=c, alpha=0.18)
        rg = raw.get(group)
        if rg is not None:
            plt.scatter(rg[0], rg[1], color=c, s=10, alpha=0.4, edgecolors="none")
    # Atlas markers: one distinctly-coloured, labelled series per condition value.
    atlas_cmap = plt.get_cmap("coolwarm")
    a_keys = list(atlas.keys())
    for ai, tag in enumerate(a_keys):
        xs, ys = atlas[tag]
        plt.scatter(xs, ys, color=atlas_cmap(ai / max(1, len(a_keys) - 1)),
                    marker="X", s=80, edgecolors="black", linewidths=0.6,
                    zorder=5, label=f"atlas {tag}")
    plt.title(f"{tissue} volume vs age")
    plt.xlabel("age (weeks)")
    plt.ylabel("volume (mm$^3$)")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.2)
    out = out_dir / f"curve_{tissue}.png"
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close()
    return out


# --- main --------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=Path("configs/datasets/example.yaml"),
                    help="dataset YAML; supplies the seg column + label_names + default table")
    ap.add_argument("--table", type=Path, default=None,
                    help="subjects table (default: the dataset YAML's tsv_file)")
    ap.add_argument("--age-col", default="GA_MRI")
    ap.add_argument("--group-col", default="ExamType")
    ap.add_argument("--groups", nargs="+", default=["PRE_OP", "POST_OP"],
                    help="group values to keep, in plot order")
    ap.add_argument("--rating-col", default="rating_svr")
    ap.add_argument("--rating-min", type=float, default=1.0)
    ap.add_argument("--kernel-sigma-weeks", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--output", type=Path, default=Path("output/growth_curves"))
    ap.add_argument("--atlas-dir", type=Path, default=None,
                    help="atlas/ dir with {seg}_<cond>=<val>_*.nii.gz + ages.json "
                         "to overlay (markers labelled by condition value)")
    args = ap.parse_args()

    spec, _, _ = load_dataset_config(args.dataset)
    seg_col = spec.segmentation_modality
    label_names = list(spec.label_names)
    n_labels = len(label_names)
    tissues = [n for i, n in enumerate(label_names) if i != 0]  # skip background
    table = args.table or spec.tsv_file
    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"dataset={spec.name} seg_col={seg_col} labels={label_names}")
    print(f"table={table}")

    # 1. load + filter
    df = read_subjects_table(table)
    for c in (seg_col, args.age_col, args.group_col, args.rating_col):
        if c not in df.columns:
            raise SystemExit(f"column {c!r} not in {table}")
    df = df[df[args.rating_col] >= args.rating_min]
    df = df[df[args.group_col].isin(args.groups)]
    df = df[df[seg_col].notna() & (df[seg_col] != "") & df[args.age_col].notna()]
    df = df.reset_index(drop=True)
    print(f"rows after filter (rating>={args.rating_min}, group in {args.groups}): {len(df)}")
    if len(df) == 0:
        raise SystemExit("no rows left after filtering")

    # 2. per-seg per-label volumes (parallel)
    jobs = [(i, p, n_labels) for i, p in df[seg_col].items() if Path(p).exists()]
    missing = len(df) - len(jobs)
    if missing:
        print(f"  [warn] {missing} seg paths do not exist on disk and are skipped")
    vols = np.full((len(df), n_labels), np.nan)
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for idx, v in ex.map(_worker, jobs, chunksize=4):
            if v is not None:
                vols[idx] = v
            done += 1
            if done % 50 == 0:
                print(f"  processed {done}/{len(jobs)}")

    vol_df = df[["subject_id", args.age_col, args.group_col, args.rating_col]].copy()
    for i, name in enumerate(label_names):
        if i == 0:
            continue
        vol_df[name] = vols[:, i]
    vol_df[TOTAL_BRAIN] = vols[:, 1:].sum(axis=1)
    vol_df = vol_df[vol_df[TOTAL_BRAIN] > 0].reset_index(drop=True)
    vol_csv = out_dir / "tissue_volumes.csv"
    vol_df.to_csv(vol_csv, index=False)
    print(f"wrote {vol_csv}: {len(vol_df)} subjects x {len(tissues)} tissues")

    # 3. fit curves per (tissue x group)
    curve_tissues = tissues + [TOTAL_BRAIN]
    reference: dict = {
        "age_col": args.age_col, "group_col": args.group_col,
        "rating_col": args.rating_col, "rating_min": args.rating_min,
        "kernel_sigma_weeks": args.kernel_sigma_weeks, "tissues": {},
    }
    curves_by_tissue: dict[str, dict] = {}
    raw_by_tissue: dict[str, dict] = {}
    for tissue in curve_tissues:
        per_group: dict = {}
        raw: dict = {}
        for group in args.groups:
            g = vol_df[vol_df[args.group_col] == group]
            ages = g[args.age_col].to_numpy(dtype=np.float64)
            vv = g[tissue].to_numpy(dtype=np.float64)
            m = ~(np.isnan(ages) | np.isnan(vv))
            ages, vv = ages[m], vv[m]
            if ages.size < 2:
                print(f"  [warn] {tissue}/{group}: <2 rows, skipped")
                continue
            grid, mu, sd = fit_curve(ages, vv, args.kernel_sigma_weeks)
            per_group[group] = {"n": int(ages.size), "age_grid": grid,
                                "mu_grid": mu, "sigma_grid": sd}
            raw[group] = (ages, vv)
        curves_by_tissue[tissue] = per_group
        raw_by_tissue[tissue] = raw
        reference["tissues"][tissue] = {
            grp: {"n": c["n"], "age_grid": c["age_grid"].tolist(),
                  "mu_grid": c["mu_grid"].tolist(), "sigma_grid": c["sigma_grid"].tolist()}
            for grp, c in per_group.items()
        }
    ref_path = out_dir / "growth_curves_reference.json"
    ref_path.write_text(json.dumps(reference, indent=2))
    print(f"wrote {ref_path}")

    # 4 + 5. atlas overlay (optional) then plot
    atlas_by_tissue: dict[str, dict] = {t: {} for t in curve_tissues}
    if args.atlas_dir is not None:
        atlas_by_tissue = _collect_atlas_volumes(
            args.atlas_dir, seg_col, label_names, tissues, out_dir,
        )

    for tissue in curve_tissues:
        _plot_tissue(tissue, curves_by_tissue[tissue], raw_by_tissue[tissue],
                     atlas_by_tissue.get(tissue, {}), out_dir)
    print(f"wrote {len(curve_tissues)} plots to {out_dir}")
    return 0


def _collect_atlas_volumes(atlas_dir: Path, seg_col: str, label_names: list[str],
                           tissues: list[str], out_dir: Path) -> dict[str, dict]:
    """Measure per-tissue volumes from 4D atlas seg NIfTIs and index them by
    (tissue -> condition-tag -> (ages, volumes)). The atlas filenames carry the
    actual condition values (e.g. ``Seg_ExamType_num=-1.5_GA_MRI=25-26_ep=5``),
    so each marker series is keyed/labelled by that value. Writes
    atlas_volumes.csv."""
    n_labels = len(label_names)
    ages_json = atlas_dir / "ages.json"
    if not ages_json.exists():
        raise SystemExit(f"atlas overlay needs {ages_json}")
    ages = np.asarray(json.loads(ages_json.read_text()), dtype=np.float64)
    # {seg}_<tag>_<temporal>=<age-range>_ep=<n>.nii.gz ; capture <tag> (the
    # condition value(s)); the trailing _ep= anchor disambiguates the temporal part.
    pattern = re.compile(
        rf"^{re.escape(seg_col)}_(?P<tag>.+?)_[A-Za-z_]\w*=[-\d.]+(?:-[-\d.]+)?_ep=\d+\.nii\.gz$"
    )
    rows = []
    result: dict[str, dict] = {t: {} for t in tissues + [TOTAL_BRAIN]}
    for f in sorted(atlas_dir.glob(f"{seg_col}_*.nii.gz")):
        m = pattern.match(f.name)
        if not m:
            continue
        tag = m.group("tag")          # e.g. "ExamType_num=-1.5" or "cond=0"
        frame_vols = _atlas_frame_volumes(str(f), n_labels)  # (T, n_labels)
        if frame_vols.shape[0] != ages.size:
            print(f"  [warn] {f.name}: {frame_vols.shape[0]} frames != {ages.size} ages; skipped")
            continue
        total = frame_vols[:, 1:].sum(axis=1)
        for ti, name in enumerate(label_names):
            if ti == 0:
                continue
            result[name].setdefault(tag, ([], []))
            result[name][tag][0].extend(ages.tolist())
            result[name][tag][1].extend(frame_vols[:, ti].tolist())
        result[TOTAL_BRAIN].setdefault(tag, ([], []))
        result[TOTAL_BRAIN][tag][0].extend(ages.tolist())
        result[TOTAL_BRAIN][tag][1].extend(total.tolist())
        for t, age in enumerate(ages):
            row = {"condition": tag, "age": age, TOTAL_BRAIN: total[t]}
            for ti, name in enumerate(label_names):
                if ti != 0:
                    row[name] = frame_vols[t, ti]
            rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "atlas_volumes.csv", index=False)
        print(f"wrote {out_dir / 'atlas_volumes.csv'}: {len(rows)} atlas frames")
    # convert ([ages],[vols]) lists to arrays for plotting
    return {t: {g: (np.asarray(a), np.asarray(v)) for g, (a, v) in gd.items()}
            for t, gd in result.items()}


if __name__ == "__main__":
    raise SystemExit(main())
