"""Brain-age prediction sweep against an existing train checkpoint.

Loads the train checkpoint, builds the val split (30 subjects by default),
fits val latents with the frozen decoder, then reports:

1. Age distribution stats for train + val (sanity: not pathologically narrow).
2. Naive baselines: predict median / mean of train ages.
3. kNN regressor at a few k.
4. PLS regression sweep over n_components.

No CLI surface — invoke directly: ``python scripts/sweep_age_prediction.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from cinema.checkpoint import apply_to_state, load_checkpoint
from cinema.cli import _build_dataset
from cinema.criterion import Criterion
from cinema.evaluator import predict_condition_pls, predict_condition_regressor
from cinema.fitter import LatentFitter
from cinema.state import build_fit_state, build_train_state


TRAIN_CKPT = "output/checkpoint_final.pt"
N_VAL = 30
FIT_EPOCHS = 5
PLS_COMPONENTS = [2, 5, 10, 20, 40]
KNN_K_VALUES = [3, 5, 10]


def _ascii_hist(values: np.ndarray, bins: int = 10, width: int = 40) -> str:
    counts, edges = np.histogram(values, bins=bins)
    peak = max(counts.max(), 1)
    lines = []
    for i in range(bins):
        bar = "█" * int(round(width * counts[i] / peak))
        lines.append(f"    [{edges[i]:5.2f}, {edges[i + 1]:5.2f})  {counts[i]:3d}  {bar}")
    return "\n".join(lines)


def main() -> int:
    print(f"loading train checkpoint: {TRAIN_CKPT}")
    ckpt = load_checkpoint(TRAIN_CKPT, map_location="cuda")
    cfg = ckpt.config
    cfg.device = "cuda"

    print(f"building train dataset ({ckpt.latents.shape[0]} subjects) ...")
    train_dataset = _build_dataset(cfg, split="train", n_subjects=ckpt.latents.shape[0])
    train_state = build_train_state(cfg, dataset=train_dataset)
    apply_to_state(ckpt, train_state, load_optimizer=False)

    print(f"building val dataset ({N_VAL} subjects) ...")
    val_dataset = _build_dataset(cfg, split="val", n_subjects=N_VAL)
    print(f"val: {len(val_dataset)} subjects loaded")

    train_ages = train_dataset.df["scan_age"].astype(float).to_numpy()
    val_ages = val_dataset.df["scan_age"].astype(float).to_numpy()

    print("\n=== Age distribution (weeks) ===")
    for name, ages in [("train", train_ages), ("val", val_ages)]:
        print(
            f"  {name}: n={len(ages):3d}  min={ages.min():.2f}  max={ages.max():.2f}  "
            f"mean={ages.mean():.2f}  median={np.median(ages):.2f}  std={ages.std():.2f}"
        )
    print("\n  val histogram:")
    print(_ascii_hist(val_ages, bins=10))

    print(f"\nfitting val latents ({FIT_EPOCHS} epochs) ...")
    criterion = Criterion(
        intensity_dims=sum(cfg.decoder.out_dim[:-1]),
        has_segmentation=cfg.dataset_spec.has_segmentation,
        class_weights=cfg.dataset_spec.class_weights,
        loss_metric=cfg.optimizer.loss_metric,
        tf_weight=cfg.optimizer.tf_weight,
    ).to(cfg.device)
    fit_state = build_fit_state(
        cfg, dataset=val_dataset, decoder=train_state.decoder,
        split="val", learn_conditions=True, n_epochs=FIT_EPOCHS,
    )
    fitter = LatentFitter(
        fit_state, criterion,
        n_samples=cfg.training.n_samples,
        n_epochs=FIT_EPOCHS,
        seg_weight=cfg.optimizer.seg_weight,
        on_epoch_end=lambda loss, ep: print(f"  fit {ep + 1}/{FIT_EPOCHS}: loss={loss:.4f}"),
    )
    fitter.fit()

    train_latents = train_state.latents.detach()
    val_latents = fit_state.latents.detach()
    train_ages_t = torch.from_numpy(train_ages)
    val_ages_t = torch.from_numpy(val_ages)

    print("\n=== Naive baselines (no latents used) ===")
    median_train = float(np.median(train_ages))
    mean_train = float(np.mean(train_ages))
    naive_median = float(np.mean(np.abs(val_ages - median_train)))
    naive_mean = float(np.mean(np.abs(val_ages - mean_train)))
    print(f"  predict median(train)={median_train:.2f}: MAE = {naive_median:.3f}")
    print(f"  predict mean(train)  ={mean_train:.2f}: MAE = {naive_mean:.3f}")

    print("\n=== kNN regressor ===")
    for k in KNN_K_VALUES:
        r = predict_condition_regressor(
            train_latents, val_latents, train_ages_t, val_ages_t,
            condition_key="scan_age", k=k,
        )
        print(f"  k={k:2d}: mae={r.mae:.3f} ± {r.std:.3f}")

    print("\n=== PLS regression sweep ===")
    for n in PLS_COMPONENTS:
        r = predict_condition_pls(
            train_latents, val_latents, train_ages_t, val_ages_t,
            condition_key="scan_age", n_components=n,
        )
        print(f"  n_components={n:2d}: mae={r.mae:.3f} ± {r.std:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
