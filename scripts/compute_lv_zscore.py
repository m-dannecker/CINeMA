#!/usr/bin/env python3
"""Compute an age-normalized lateral-ventricle z-score column for a subjects table.

This is the *normative-reference* step that follows scripts/extract_lv_condition.py
(which writes ``lv_ratio``). It fits, per cohort, a smooth age-conditional mean
``mu(age)`` and std ``sigma(age)`` of ``lv_ratio`` over the **full cohort** (max
statistical power, not the training split), then writes a per-subject z-score:

    lv_z = (lv_ratio - mu(age)) / sigma(age)

Fetal and neonatal are fit **separately** (grouped by ``--group-col``): they are
different developmental regimes / acquisition pipelines, so a shared curve would
be ill-posed at the cohort boundary.

The fitted curves are saved to a JSON reference artifact so the *same* z-score
mapping can be applied to future subjects (and so the reference is inspectable
and version-controlled, instead of being a per-split side effect of training).

CINeMA consumes ``lv_z`` directly via the parameter-free ``zscore``
normalization (tanh(z / clip_sigmas)); the clip is applied in-model, so this
script stores the *raw* (unclipped) z-score.

Usage:
    python scripts/compute_lv_zscore.py \
        --table configs/GenericSubjectsTable.csv \
        --value-col lv_ratio --age-col scan_age --group-col dataset_id \
        --kernel-sigma-weeks 2.0 \
        --reference configs/lv_zscore_reference.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def fit_curve(ages: np.ndarray, vals: np.ndarray, kernel_sigma: float,
              n_grid: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kernel-regress mu(age) and sigma(age) over an age grid (Gaussian kernel).

    Mirrors cinema.conditioning.AgeRelativeNormalization.fit so the offline
    reference matches the in-code math exactly.
    """
    grid = np.linspace(ages.min(), ages.max(), n_grid)
    diffs = (ages[None, :] - grid[:, None]) / kernel_sigma
    w = np.exp(-0.5 * diffs * diffs)
    w = w / np.clip(w.sum(axis=1, keepdims=True), 1e-12, None)
    mu = (w * vals[None, :]).sum(axis=1)
    var = (w * (vals[None, :] - mu[:, None]) ** 2).sum(axis=1)
    sd = np.sqrt(var + 1e-8)
    return grid, mu, sd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", type=Path,
                    default=Path("configs/GenericSubjectsTable.csv"))
    ap.add_argument("--value-col", default="lv_ratio")
    ap.add_argument("--age-col", default="scan_age")
    ap.add_argument("--group-col", default="dataset_id",
                    help="fit a separate reference curve per value of this column")
    ap.add_argument("--out-col", default="lv_z")
    ap.add_argument("--kernel-sigma-weeks", type=float, default=2.0)
    ap.add_argument("--reference", type=Path,
                    default=Path("configs/lv_zscore_reference.json"))
    ap.add_argument("--output", type=Path, default=None,
                    help="defaults to overwriting --table in place")
    args = ap.parse_args()

    df = pd.read_csv(args.table)
    for c in (args.value_col, args.age_col, args.group_col):
        if c not in df.columns:
            raise SystemExit(f"column {c!r} not in {args.table}")

    z = np.full(len(df), np.nan)
    reference: dict = {
        "value_col": args.value_col,
        "age_col": args.age_col,
        "group_col": args.group_col,
        "kernel_sigma_weeks": float(args.kernel_sigma_weeks),
        "groups": {},
    }

    for group, gdf in df.groupby(args.group_col):
        valid = gdf[[args.value_col, args.age_col]].dropna()
        ages = valid[args.age_col].to_numpy(dtype=np.float64)
        vals = valid[args.value_col].to_numpy(dtype=np.float64)
        if ages.size < 2:
            print(f"  [warn] group {group!r}: <2 valid rows, skipped")
            continue
        grid, mu, sd = fit_curve(ages, vals, args.kernel_sigma_weeks)
        mu_at = np.interp(ages, grid, mu)
        sd_at = np.interp(ages, grid, sd)
        z[valid.index] = (vals - mu_at) / sd_at
        reference["groups"][str(group)] = {
            "n": int(ages.size),
            "age_grid": grid.tolist(),
            "mu_grid": mu.tolist(),
            "sigma_grid": sd.tolist(),
        }
        zz = z[valid.index]
        print(f"  {group}: n={ages.size} "
              f"age=[{ages.min():.1f},{ages.max():.1f}] "
              f"z=[{zz.min():.2f},{zz.max():.2f}] mean={zz.mean():.3f}")

    df[args.out_col] = z

    out = args.output or args.table
    df.to_csv(out, index=False)
    args.reference.write_text(json.dumps(reference, indent=2))
    n_ok = int(df[args.out_col].notna().sum())
    print(f"wrote {out}: {n_ok}/{len(df)} rows have {args.out_col}")
    print(f"wrote reference curves: {args.reference}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
