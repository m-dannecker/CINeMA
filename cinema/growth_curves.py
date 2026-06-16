"""Tissue-volume growth curves for atlas validation.

Measures per-tissue volumes from the *training* segmentations (the same cohort
the atlas is regressed from), fits mean +/- SD curves vs the temporal axis, and
overlays the generated atlas's own measured tissue volumes. Driven by the
``growth_curves`` block of the atlas recipe (``GrowthCurvesConfig``); produced
automatically at the end of atlas generation when enabled.

The tissue -> label-index map comes from the dataset's
``segmentation_classes.label_names`` (index i == label value i), and is
cohort-specific. ``TotalBrain`` is the sum of all foreground labels (> 0).

Because ``generate_atlas`` already holds the per-(age x combo) hard segmentations
in memory and knows each combo's condition values, the atlas overlay is exact and
auto-routed to the matching empirical curve — no manual cond->group mapping.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np

from .atlas import combo_tag
from .conditioning import kernel_regress
from .state import TrainState

TOTAL_BRAIN = "TotalBrain"
MAX_GROUPS = 12  # above this, group_by is treated as continuous (single curve)


def _label_volumes(path: str, n_labels: int) -> np.ndarray:
    """Per-label volume (mm^3) for one segmentation file, length ``n_labels``."""
    img = nib.load(path)
    data = np.asarray(img.dataobj)
    voxvol = float(np.prod(img.header.get_zooms()[:3]))
    counts = np.bincount(data.astype(np.int64).ravel(), minlength=n_labels)
    return counts[:n_labels].astype(np.float64) * voxvol


def _training_volumes(df, seg_col: str, n_labels: int) -> np.ndarray:
    """(N, n_labels) per-subject label volumes from the training df's seg paths.

    Serial on purpose — the atlas runs inside the (CUDA-initialised) training
    process, where forking a process pool is unsafe.
    """
    vols = np.full((len(df), n_labels), np.nan)
    for i, p in enumerate(df[seg_col].to_numpy()):
        if isinstance(p, str) and p and Path(p).exists():
            try:
                vols[i] = _label_volumes(p, n_labels)
            except Exception as e:  # corrupt/unreadable -> leave NaN
                print(f"[growth_curves] warn: failed to read {p}: {e}")
    return vols


def _atlas_volumes(combos, seg_frames, ages, spacing, label_names,
                   tissue_idx: dict[str, int], group_by: Optional[str], grouped: bool):
    """Per-tissue volumes of the generated atlas, keyed tissue -> list of
    (age, volume, group_value, combo_label). Also returns flat rows for CSV."""
    voxvol = float(np.prod(spacing))
    n_labels = len(label_names)
    points: dict[str, list] = {t: [] for t in list(tissue_idx) + [TOTAL_BRAIN]}
    rows: list[dict] = []
    for ci, combo in enumerate(combos):
        frames = seg_frames[ci] if ci < len(seg_frames) else []
        if not frames:
            continue
        gval = combo.get(group_by) if grouped else None
        clabel = combo_tag(combo, ci, sep=", ")   # e.g. "ExamType_num=-1.5"
        for ai, age in enumerate(ages):
            if ai >= len(frames):
                break
            counts = np.bincount(frames[ai].astype(np.int64).ravel(), minlength=n_labels)
            total = float(counts[1:n_labels].sum()) * voxvol
            row = {"cond": ci, "combo": clabel, "age": float(age), TOTAL_BRAIN: total}
            for t, idx in tissue_idx.items():
                vol = float(counts[idx]) * voxvol
                points[t].append((age, vol, gval, clabel))
                row[t] = vol
            points[TOTAL_BRAIN].append((age, total, gval, clabel))
            rows.append(row)
    return points, rows


def generate_growth_curves(
    state: TrainState,
    recipe,
    *,
    temporal_condition: str,
    output_dir,
    combos: list[dict],
    seg_frames: list,
    epoch: int = 0,
) -> list[Path]:
    """Fit + plot tissue growth curves with the atlas overlaid. Returns paths written."""
    import json

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    cfg = recipe.growth_curves
    spec = state.dataset.dataset_spec
    label_names = list(spec.label_names or [])
    seg_col = spec.segmentation_modality
    df = state.dataset.df
    if not label_names or not seg_col:
        print("[growth_curves] dataset has no segmentation label scheme; skipping")
        return []
    if not any(seg_frames):
        print("[growth_curves] atlas produced no segmentation frames; skipping")
        return []
    if temporal_condition not in df.columns:
        print(f"[growth_curves] age column {temporal_condition!r} not in training df; skipping")
        return []

    n_labels = len(label_names)
    out_dir = Path(output_dir) / "atlas" / "growth_curves"
    out_dir.mkdir(parents=True, exist_ok=True)

    # tissue -> label index (TotalBrain is special: sum of foreground)
    requested = cfg.tissues or [n for i, n in enumerate(label_names) if i != 0]
    tissue_idx: dict[str, int] = {}
    for t in requested:
        if t == TOTAL_BRAIN:
            continue
        if t in label_names and label_names.index(t) != 0:
            tissue_idx[t] = label_names.index(t)
        else:
            print(f"[growth_curves] tissue {t!r} not a foreground label; skipping")
    plot_tissues = list(tissue_idx) + [TOTAL_BRAIN]

    # training-subject volumes
    vols = _training_volumes(df, seg_col, n_labels)
    total = vols[:, 1:].sum(axis=1)
    ages_all = df[temporal_condition].to_numpy(dtype=np.float64)

    def series(tissue: str) -> np.ndarray:
        return total if tissue == TOTAL_BRAIN else vols[:, tissue_idx[tissue]]

    # grouping
    group_by = cfg.group_by
    grouped = bool(
        group_by and group_by in df.columns
        and df[group_by].nunique(dropna=True) <= MAX_GROUPS
    )
    if group_by and not grouped:
        print(f"[growth_curves] group_by={group_by!r} unusable (missing/too many "
              f"values); fitting a single curve")
    group_vals = (sorted(df[group_by].dropna().unique().tolist()) if grouped else [None])
    masks = {
        g: (np.ones(len(df), bool) if g is None else (df[group_by].to_numpy() == g))
        for g in group_vals
    }
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    gcolor = {g: colors[i % len(colors)] for i, g in enumerate(group_vals)}

    def glabel(g) -> str:
        return "all" if g is None else cfg.group_labels.get(g, str(g))

    # atlas-measured volumes (exact, from in-memory hard segs)
    atlas_pts, atlas_rows = _atlas_volumes(
        combos, seg_frames, recipe.ages, recipe.spacing, label_names,
        tissue_idx, group_by, grouped,
    )
    # One colour + legend entry per distinct atlas condition value, along a
    # gradient (so e.g. a -1.5..1.5 sweep reads cool->warm).
    atlas_labels = list(dict.fromkeys(
        combo_tag(c, i, sep=", ") for i, c in enumerate(combos)
    ))
    cmap = plt.get_cmap("coolwarm")
    n_a = max(1, len(atlas_labels))
    atlas_colors = {
        lbl: cmap(i / max(1, n_a - 1)) for i, lbl in enumerate(atlas_labels)
    }

    written: list[Path] = []
    reference: dict = {
        "age_col": temporal_condition,
        "group_by": group_by if grouped else None,
        "kernel_sigma": cfg.kernel_sigma,
        "tissues": {},
    }

    for tissue in plot_tissues:
        s = series(tissue)
        plt.figure(figsize=(7, 5))
        ref_t: dict = {}
        for g in group_vals:
            m = masks[g] & np.isfinite(s) & np.isfinite(ages_all)
            ag, vv = ages_all[m], s[m]
            if ag.size < 2:
                continue
            grid, mu, sd = kernel_regress(ag, vv, cfg.kernel_sigma)
            c = gcolor[g]
            plt.plot(grid, mu, color=c, label=f"{glabel(g)} (n={ag.size})")
            plt.fill_between(grid, mu - sd, mu + sd, color=c, alpha=0.18)
            plt.scatter(ag, vv, color=c, s=10, alpha=0.35, edgecolors="none")
            ref_t[glabel(g)] = {"n": int(ag.size), "age_grid": grid.tolist(),
                                "mu_grid": mu.tolist(), "sigma_grid": sd.tolist()}
        # atlas overlay: group points by their condition value so each value is
        # one labelled, distinctly-coloured marker series (X markers).
        by_label: dict[str, tuple[list, list]] = {}
        for (age, vol, gval, clabel) in atlas_pts.get(tissue, []):
            xs, ys = by_label.setdefault(clabel, ([], []))
            xs.append(age)
            ys.append(vol)
        for clabel in atlas_labels:
            if clabel not in by_label:
                continue
            xs, ys = by_label[clabel]
            plt.scatter(xs, ys, color=atlas_colors[clabel], marker="X", s=90,
                        zorder=5, edgecolors="black", linewidths=0.6,
                        label=f"atlas {clabel}")
        plt.title(f"{tissue} volume vs {temporal_condition}")
        plt.xlabel(temporal_condition)
        plt.ylabel("volume (mm$^3$)")
        plt.legend(fontsize=8)
        plt.grid(alpha=0.2)
        p = out_dir / f"curve_{tissue}_ep={epoch}.png"
        plt.savefig(p, dpi=100, bbox_inches="tight")
        plt.close()
        reference["tissues"][tissue] = ref_t
        written.append(p)

    # artifacts: training volumes, fitted curves, atlas volumes
    cols = (["subject_id"] if "subject_id" in df.columns else []) + [temporal_condition]
    vol_df = df[cols].copy()
    if grouped:
        vol_df[group_by] = df[group_by].to_numpy()
    for t, idx in tissue_idx.items():
        vol_df[t] = vols[:, idx]
    vol_df[TOTAL_BRAIN] = total
    vol_csv = out_dir / "tissue_volumes.csv"
    vol_df.to_csv(vol_csv, index=False)
    written.append(vol_csv)

    ref_path = out_dir / "growth_curves_reference.json"
    ref_path.write_text(json.dumps(reference, indent=2))
    written.append(ref_path)

    if atlas_rows:
        atlas_csv = out_dir / "atlas_volumes.csv"
        pd.DataFrame(atlas_rows).to_csv(atlas_csv, index=False)
        written.append(atlas_csv)

    print(f"[growth_curves] wrote {len(plot_tissues)} curves to {out_dir}")
    return written
