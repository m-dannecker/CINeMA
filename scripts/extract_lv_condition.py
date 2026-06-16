#!/usr/bin/env python3
"""Extract lateral-ventricle (LV) condition columns into a CINeMA subjects table.

For every row whose segmentation path resolves on disk, compute:

- ``lv_volume_mm3``    — volume of the LV label (default id 5) in mm^3
- ``brain_volume_mm3`` — volume of all foreground labels (seg > 0) in mm^3
- ``lv_ratio``         — ``lv_volume_mm3 / brain_volume_mm3``

The ``lv_ratio`` column is the physical-unit quantity meant to be conditioned on
via ``age_relative`` normalization (see configs/datasets/example.yaml). Rows with
a missing/empty seg path (or a path that does not resolve) get NaN, which the
age-relative fit masks out automatically.

Usage:
    python scripts/extract_lv_condition.py \
        --table configs/GenericSubjectsTable.csv \
        --seg-col SegEM9 --lv-label 5
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


def _volumes_for_seg(path: str, lv_label: int) -> tuple[float, float]:
    """Return (lv_volume_mm3, brain_volume_mm3) for one segmentation file."""
    img = nib.load(path)
    # Integer label volume — avoid the float64 upcast of get_fdata().
    data = np.asarray(img.dataobj)
    voxvol = float(np.prod(img.header.get_zooms()[:3]))
    lv_vox = int((data == lv_label).sum())
    brain_vox = int((data > 0).sum())
    return lv_vox * voxvol, brain_vox * voxvol


def _worker(args: tuple[int, str, int]) -> tuple[int, float, float]:
    idx, path, lv_label = args
    try:
        lv, brain = _volumes_for_seg(path, lv_label)
    except Exception as e:  # unreadable / corrupt file -> NaN, keep going
        print(f"  [warn] row {idx}: failed to read {path}: {e}")
        return idx, float("nan"), float("nan")
    return idx, lv, brain


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", type=Path,
                    default=Path("configs/GenericSubjectsTable.csv"))
    ap.add_argument("--seg-col", default="SegEM9")
    ap.add_argument("--lv-label", type=int, default=5)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--output", type=Path, default=None,
                    help="defaults to overwriting --table in place")
    args = ap.parse_args()

    df = pd.read_csv(args.table)
    if args.seg_col not in df.columns:
        raise SystemExit(f"seg column {args.seg_col!r} not in {args.table}")

    jobs = []
    for idx, p in df[args.seg_col].items():
        if isinstance(p, str) and p and os.path.exists(p):
            jobs.append((idx, p, args.lv_label))
    print(f"rows={len(df)} seg-present-on-disk={len(jobs)} "
          f"workers={args.workers} lv_label={args.lv_label}")

    lv_col = np.full(len(df), np.nan)
    brain_col = np.full(len(df), np.nan)
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for idx, lv, brain in ex.map(_worker, jobs, chunksize=4):
            lv_col[idx] = lv
            brain_col[idx] = brain
            done += 1
            if done % 100 == 0:
                print(f"  processed {done}/{len(jobs)}")

    df["lv_volume_mm3"] = lv_col
    df["brain_volume_mm3"] = brain_col
    with np.errstate(invalid="ignore", divide="ignore"):
        df["lv_ratio"] = df["lv_volume_mm3"] / df["brain_volume_mm3"]

    out = args.output or args.table
    df.to_csv(out, index=False)
    n_ok = int(df["lv_ratio"].notna().sum())
    print(f"wrote {out}: {n_ok}/{len(df)} rows have lv_ratio")
    valid = df["lv_ratio"].dropna()
    if len(valid):
        print(f"lv_ratio: min={valid.min():.5f} med={valid.median():.5f} "
              f"max={valid.max():.5f} mean={valid.mean():.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
