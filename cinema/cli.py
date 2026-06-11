"""
Command-line interface for CINeMA.

Five subcommands, each a thin orchestrator over the pure modules in this
package:

- ``train`` — full training loop (trainer + periodic checkpoint + optional
  periodic validation + optional final atlas). Validation is controlled by
  the ``validation:`` block in the train config.
- ``fit`` — point a trained checkpoint at a new dataset, fit per-subject
  latents (and optional tfs / conditions) with the decoder frozen, save the
  result as a new checkpoint.
- ``infer`` — reconstruct every subject in a checkpoint's dataset to NIfTI.
- ``atlas`` — generate atlases from a checkpoint according to a recipe YAML.
- ``evaluate`` — reconstruct + ANTs-register + compute PSNR/SSIM/Dice per
  subject; write a JSON report.

Invocation: ``python -m cinema <cmd> ...``. The module is importable (tests
call ``main(argv=[...])`` directly) rather than exclusively script-form.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

from .atlas import generate_atlas
from .checkpoint import Checkpoint, apply_to_state, load_checkpoint, save_checkpoint
from .config import (
    AtlasRecipe,
    TrainConfig,
    load_atlas_recipe,
    load_dataset_config,
    load_split_subject_ids,
    load_train_config,
)
from .criterion import Criterion
from .data import Data
from .evaluator import (
    LatentAnalysisResult,
    predict_condition_nca,
    predict_condition_pls,
    predict_condition_regressor,
    reconstruct_subject,
    save_reconstruction,
)
from .fitter import LatentFitter
from .state import build_fit_state, build_train_state
from .trainer import Trainer


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _log(msg: str) -> None:
    print(f"[CINeMA] {msg}", flush=True)


def _fmt_secs(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{int(m)}m{s:04.1f}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h{int(m):02d}m{s:04.1f}s"


def _build_criterion(
    cfg: TrainConfig, *, has_segmentation: Optional[bool] = None
) -> Criterion:
    intensity_dims = sum(cfg.decoder.out_dim[:-1])
    if has_segmentation is None:
        has_segmentation = cfg.dataset_spec.has_segmentation
    return Criterion(
        intensity_dims=intensity_dims,
        has_segmentation=has_segmentation,
        class_weights=cfg.dataset_spec.class_weights,
        loss_metric=cfg.optimizer.loss_metric,
        tf_weight=cfg.optimizer.tf_weight,
    )


def _config_to_dict(cfg: TrainConfig, *, drop_paths: bool = False) -> dict:
    """Full effective training config as a nested, serialisable dict (reflects
    --set overrides). With ``drop_paths=True`` the dataset tsv + subject_ids file
    paths are omitted (used for the wandb config)."""
    dataset = asdict(cfg.dataset_spec)
    if drop_paths:
        dataset.pop("tsv_file", None)
        dataset.pop("subject_ids", None)
    return {
        "seed": cfg.seed,
        "device": cfg.device,
        "amp": cfg.amp,
        "output_dir": cfg.output_dir,
        "dataset": dataset,
        "conditions": cfg.conditions.to_dict(),
        "constraints": {s.name: asdict(s) for s in cfg.constraints.specs},
        "decoder": asdict(cfg.decoder),
        "optimizer": asdict(cfg.optimizer),
        "training": asdict(cfg.training),
        "validation": asdict(cfg.validation),
        "atlas": asdict(cfg.atlas),
        "augmentation": asdict(cfg.augmentation),
        "logging": asdict(cfg.logging),
    }


def _centroid_latents_init(
    train_latents: torch.Tensor, n_subjects: int
) -> torch.Tensor:
    """Per-subject latent init for val/test fitting: the mean training latent,
    repeated for every new subject.

    Auto-decoder test-time fitting starts from a fixed init and optimises the
    latent through the frozen decoder. With no latent regularisation the trained
    latents drift away from the origin, so a zero/near-zero init lands off the
    decoder's learned manifold and fits poorly (the symptom: val-fit start-loss
    that worsens as training proceeds). Seeding every new subject at the
    training-latent centroid starts the optimisation inside that manifold.
    """
    centroid = train_latents.detach().mean(dim=0, keepdim=True)  # (1, C, lx, ly, lz)
    return centroid.repeat(n_subjects, *([1] * (centroid.ndim - 1))).contiguous()


def _build_dataset(
    cfg: TrainConfig, *, split: str, n_subjects: int,
    skip_segmentation: bool = False,
    disable_augmentation: bool = False,
) -> Data:
    mask_intensities = bool(cfg.training.mask_reconstruction) and not skip_segmentation
    augmentation = None if disable_augmentation else cfg.augmentation
    subject_ids = load_split_subject_ids(cfg.dataset_spec, split)
    return Data(
        dataset_spec=cfg.dataset_spec,
        constraints=cfg.constraints,
        conditions=cfg.conditions,
        tsv_file=cfg.dataset_spec.tsv_file,
        split=split,
        n_subjects=n_subjects,
        subject_ids=subject_ids,
        seed=cfg.seed,
        skip_segmentation=skip_segmentation,
        mask_intensities_by_segmentation=mask_intensities,
        augmentation=augmentation,
    )


def _resolve_auto_bbox(dataset: Data) -> list[float]:
    """Compute a tight ``world_bbox`` from the ACTUAL training subjects.

    Uses ``dataset.df`` — i.e. the subjects actually used for training (after
    subject-id filtering, constraints and ``n_subjects`` sampling), not the full
    subject-id list. For each, the binding quantity is the farthest foreground
    voxel from that subject's foreground centroid (the data pipeline centres
    coords on that centroid and divides by ``world_bbox/2``); the per-axis cohort
    maximum, doubled and inflated by ``world_bbox_margin``, is the tightest box
    that keeps every training subject inside ``[-1, 1]`` (margin gives val/test/
    fit headroom).
    """
    import nibabel as nib

    spec = dataset.dataset_spec
    key = spec.segmentation_modality or spec.intensity_modalities[0]

    max_rel = np.zeros(3)
    n = 0
    for path in dataset.df[key].tolist():
        if not isinstance(path, str) or not path.endswith((".nii", ".nii.gz")):
            continue
        try:
            nii = nib.load(path)
            data = np.asarray(nii.get_fdata())
        except Exception:
            continue
        fg = np.argwhere(data > 0)
        if len(fg) == 0:
            continue
        world = nib.affines.apply_affine(nii.affine, fg)
        max_rel = np.maximum(max_rel, np.abs(world - world.mean(0)).max(0))
        n += 1
    if n == 0:
        raise ValueError(
            f"world_bbox: auto found no loadable foreground for dataset {spec.name!r}"
        )
    bbox = np.ceil(2.0 * max_rel * (1.0 + spec.world_bbox_margin))
    _log(
        f"auto world_bbox from {n} actual train subjects "
        f"(+{spec.world_bbox_margin:.0%} margin): {[float(x) for x in bbox]} mm"
    )
    return [float(x) for x in bbox]


def _expand_latent_dim(latent_dim: list[int], world_bbox: list[float]) -> list[int]:
    """Resolve a ``[channels, max_size]`` latent grid to an anisotropic
    ``[channels, lx, ly, lz]`` whose spatial cells follow the ``world_bbox``
    aspect ratio (largest axis gets ``max_size`` cells, others scale down, min 2).

    A ``[channels, lx, ly, lz]`` (length-4) latent_dim is returned unchanged.
    """
    if len(latent_dim) == 4:
        return list(latent_dim)
    channels, max_size = int(latent_dim[0]), int(latent_dim[1])
    bb = np.asarray(world_bbox, dtype=float)
    dims = [max(2, int(round(max_size * b / bb.max()))) for b in bb]
    return [channels, *dims]


def _swap_dataset(
    cfg: TrainConfig, dataset_yaml: Path,
) -> TrainConfig:
    """Return a shallow-copied config whose dataset-side comes from a new YAML."""
    ds_spec, conditions, constraints = load_dataset_config(dataset_yaml)
    cfg.dataset_spec = ds_spec
    cfg.conditions = conditions
    cfg.constraints = constraints
    return cfg


# === validation ===


def _run_latent_analysis(
    cfg: TrainConfig,
    train_dataset: Data,
    val_dataset: Data,
    train_latents: torch.Tensor,
    val_latents: torch.Tensor,
) -> list[LatentAnalysisResult]:
    """Run every step in ``cfg.validation.latent_analysis`` and return results.

    Raw (un-normalized) condition values are used so the reported MAE is in the
    target's native units (e.g. weeks for ``scan_age``).
    """
    results: list[LatentAnalysisResult] = []
    for step in cfg.validation.latent_analysis:
        train_labels, _ = train_dataset.get_condition_values(step.target, normed=False)
        val_labels, _ = val_dataset.get_condition_values(step.target, normed=False)
        k = int(step.extras.get("k", 5))
        if step.type == "nca":
            r = predict_condition_nca(
                train_latents, val_latents, train_labels, val_labels,
                condition_key=step.target, k=k,
                n_components=int(step.extras.get("n_components", 2)),
            )
        elif step.type == "regressor":
            r = predict_condition_regressor(
                train_latents, val_latents, train_labels, val_labels,
                condition_key=step.target, k=k,
            )
        elif step.type == "pls":
            r = predict_condition_pls(
                train_latents, val_latents, train_labels, val_labels,
                condition_key=step.target,
                n_components=int(step.extras.get("n_components", 5)),
            )
        else:
            _log(f"  warning: unknown latent_analysis type {step.type!r}, skipping")
            continue
        hparam = (
            f"k={r.k}" if r.k is not None
            else f"n_components={r.n_components}"
        )
        _log(
            f"  latent {r.method}/{r.condition_key}: mae={r.mae:.3f} ± {r.std:.3f} "
            f"({hparam}, n_train={r.n_train}, n_val={r.n_val})"
        )
        results.append(r)
    return results


def _run_validation(
    cfg: TrainConfig,
    state,
    criterion: Criterion,
    val_dataset: Data,
    epoch: int,
    out: Path,
    *,
    use_wandb: bool,
) -> None:
    """Fit val-set latents, optionally reconstruct, optionally score with ANTs,
    optionally run latent-space analysis steps.

    The shared decoder is frozen for the duration and fully restored
    (grad flags + train/eval mode) on exit via a finally block.
    """
    val_cfg = cfg.validation
    decoder = state.decoder
    n_val = len(val_dataset)
    _log(f"validation @ epoch {epoch}: {n_val} subjects, {val_cfg.fit_epochs} fit epoch(s) ...")
    val_start = time.perf_counter()

    # Snapshot decoder state so training can resume cleanly afterwards.
    param_grad = {id(p): p.requires_grad for p in decoder.parameters()}
    was_training = decoder.training

    try:
        fit_state = build_fit_state(
            cfg, dataset=val_dataset, decoder=decoder,
            split="val", learn_conditions=True,
            n_epochs=val_cfg.fit_epochs,
            latents_init=_centroid_latents_init(state.latents, len(val_dataset)),
        )
        fitter = LatentFitter(
            fit_state, criterion,
            n_samples=cfg.training.n_samples,
            n_epochs=val_cfg.fit_epochs,
            # Val subjects are scored as if segmentation labels are unavailable
            # (test-time conditions): never optimise the seg term during fitting.
            seg_weight=0.0,
            on_epoch_end=lambda loss, ep: _log(
                f"  val fit {ep + 1}/{val_cfg.fit_epochs}: loss={loss:.4f}"
            ),
        )
        fitter.fit()

        # Latent-space analysis runs whether or not we reconstruct: it only
        # needs fitted latents + ground-truth labels.
        latent_results: list[LatentAnalysisResult] = []
        if val_cfg.latent_analysis:
            latent_results = _run_latent_analysis(
                cfg,
                train_dataset=state.dataset,
                val_dataset=val_dataset,
                train_latents=state.latents.detach(),
                val_latents=fit_state.latents.detach(),
            )
            if use_wandb and latent_results:
                import wandb
                wandb.log(
                    {"epoch": epoch, **{
                        f"val/latent_{r.method}_mae_{r.condition_key}": r.mae
                        for r in latent_results
                    }}
                )

        all_metrics: list[dict] = []
        if val_cfg.reconstruct:
            # Check ANTs availability once so we don't re-import per subject.
            compute_fn = None
            if val_cfg.compute_metrics:
                try:
                    from .evaluator_ants import compute_subject_metrics as _csm
                    compute_fn = _csm
                except ImportError:
                    _log("  warning: compute_metrics=true but ants is not installed — skipping metrics")

            spacing = [0.5, 0.5, 0.5]
            for i in range(n_val):
                sid = str(val_dataset.df.iloc[i]["subject_id"])
                recon, affine = reconstruct_subject(
                    fit_state, i, spacing=spacing,
                    mask_reconstruction=cfg.training.mask_reconstruction,
                )
                if val_cfg.save_imgs:
                    save_reconstruction(
                        recon, affine, out,
                        subject_id=sid,
                        intensity_modalities=cfg.dataset_spec.intensity_modalities,
                        segmentation_modality=cfg.dataset_spec.segmentation_modality,
                        epoch=epoch,
                        split="val",
                    )
                if compute_fn is not None:
                    ref_paths = {
                        mod: str(val_dataset.df.iloc[i][mod])
                        for mod in cfg.dataset_spec.all_modality_keys
                        if mod in val_dataset.df.columns
                    }
                    m = compute_fn(
                        subject_id=sid,
                        pred_intensities=recon.intensities.detach().cpu().numpy(),
                        pred_seg=(
                            recon.seg_hard.detach().cpu().numpy()
                            if recon.seg_hard is not None else None
                        ),
                        pred_affine=affine.detach().cpu().numpy(),
                        ref_paths=ref_paths,
                        intensity_modalities=cfg.dataset_spec.intensity_modalities,
                        segmentation_modality=cfg.dataset_spec.segmentation_modality,
                        label_names=cfg.dataset_spec.label_names,
                    )
                    all_metrics.append(m.as_dict())
                    psnr_str = ", ".join(
                        f"{mod}={v:.2f}"
                        for mod, v in zip(cfg.dataset_spec.intensity_modalities, m.psnr)
                    )
                    _log(f"  val [{i + 1}/{n_val}] {sid}: psnr=[{psnr_str}]")

            if all_metrics and use_wandb:
                import wandb
                log_dict: dict = {"epoch": epoch}
                for mi, mod in enumerate(cfg.dataset_spec.intensity_modalities):
                    log_dict[f"val/psnr_{mod}"] = float(
                        np.mean([m["psnr"][mi] for m in all_metrics])
                    )
                    log_dict[f"val/ssim_{mod}"] = float(
                        np.mean([m["ssim"][mi] for m in all_metrics])
                    )
                dice_entries = [m["dice"] for m in all_metrics if m["dice"] is not None]
                if dice_entries:
                    for label_key in dice_entries[0]:
                        vals = [d[label_key] for d in dice_entries if label_key in d]
                        log_dict[f"val/dice_{label_key}"] = float(np.mean(vals))
                wandb.log(log_dict)

        # Dump per-epoch metrics next to the reconstructions so they can be
        # parsed offline (wandb has the time-series, this is the on-disk snapshot).
        if all_metrics or latent_results:
            metrics_dir = out / "val"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = metrics_dir / f"metrics_epoch_{epoch}.json"
            payload = {
                "epoch": epoch,
                "subjects": all_metrics,
                "latent_analysis": [r.as_dict() for r in latent_results],
            }
            with open(metrics_path, "w") as f:
                json.dump(payload, f, indent=2)
            _log(f"  wrote val metrics to {metrics_path}")

        _log(f"validation finished in {_fmt_secs(time.perf_counter() - val_start)}")

    finally:
        for p in decoder.parameters():
            p.requires_grad = param_grad[id(p)]
        decoder.train(was_training)


# === train ===


def _cmd_train(args: argparse.Namespace) -> int:
    _log(f"loading train config: {args.config}")
    cfg = load_train_config(args.config, args.set or [])
    if args.output_dir is not None:
        cfg.output_dir = str(args.output_dir)
    _seed_everything(cfg.seed)

    out = Path(cfg.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    _log(f"seed={cfg.seed} device={cfg.device} amp={cfg.amp}")
    _log(f"output dir: {out}")

    _log(
        f"loading dataset (tsv={cfg.dataset_spec.tsv_file}, "
        f"target n_subjects={cfg.training.n_subjects.train}) ..."
    )
    t0 = time.perf_counter()
    dataset = _build_dataset(cfg, split="train", n_subjects=cfg.training.n_subjects.train)
    n_loaded = len(dataset)
    _log(f"dataset: {n_loaded} train subjects loaded in {_fmt_secs(time.perf_counter() - t0)}")

    # Resolve `world_bbox: auto` from the ACTUAL training subjects (post-constraint,
    # post-sampling — i.e. dataset.df, not the full subject-id list), update the
    # dataset's frame, and freeze the concrete box into the checkpoint.
    if cfg.dataset_spec.auto_bbox:
        cfg.dataset_spec.world_bbox = _resolve_auto_bbox(dataset)
        cfg.dataset_spec.auto_bbox = False
        dataset.set_world_bbox(cfg.dataset_spec.world_bbox)
    _log(f"world_bbox: {cfg.dataset_spec.world_bbox} mm")

    # Resolve a [channels, max_size] latent grid to an anisotropic grid that
    # follows the (resolved) world_bbox aspect ratio, then freeze it.
    if len(cfg.decoder.latent_dim) == 2:
        cfg.decoder.latent_dim = _expand_latent_dim(
            cfg.decoder.latent_dim, cfg.dataset_spec.world_bbox
        )
        _log(f"latent grid expanded to {cfg.decoder.latent_dim} (anisotropic, from world_bbox)")

    # Record the full effective config (incl. --set overrides + resolved bbox/grid).
    config_path = out / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(_config_to_dict(cfg), f, sort_keys=False, default_flow_style=False)
    _log(f"saved resolved config to {config_path}")

    # Full-batch training (batch_size >= n_subjects) is strongly recommended:
    # smaller subject-batches sample each gradient step from only a subset of
    # subjects, which measurably degrades reconstruction quality. batch_size <= 0
    # means full batch. Reduce training.n_samples (coords/step) for GPU memory,
    # not batch_size.
    bs = cfg.training.batch_size
    if 0 < bs < n_loaded:
        _log(
            f"WARNING: batch_size={bs} < {n_loaded} train subjects (partial batch). "
            f"This degrades quality — prefer full batch (set training.batch_size: 0) "
            f"and reduce training.n_samples if GPU memory is the limit."
        )

    state = build_train_state(cfg, dataset=dataset)
    criterion = _build_criterion(cfg).to(cfg.device)

    val_dataset: Optional[Data] = None
    if cfg.validation.fit_epochs > 0 and cfg.training.n_subjects.val > 0:
        _log(
            f"loading val dataset (target n_subjects={cfg.training.n_subjects.val}) ..."
        )
        t0 = time.perf_counter()
        val_dataset = _build_dataset(
            cfg, split="val", n_subjects=cfg.training.n_subjects.val,
        )
        _log(
            f"val dataset: {len(val_dataset)} subjects loaded in "
            f"{_fmt_secs(time.perf_counter() - t0)}"
        )

    if args.resume is not None:
        _log(f"resuming from checkpoint: {args.resume}")
        ckpt = load_checkpoint(args.resume, map_location=cfg.device)
        apply_to_state(ckpt, state)

    use_wandb = cfg.logging.enabled
    if use_wandb:
        import wandb
        wandb.init(
            entity=cfg.logging.wandb_entity or None,
            project=cfg.logging.project or None,
            config=_config_to_dict(cfg, drop_paths=True),
            dir=str(out),
        )
        _log(f"wandb run: {wandb.run.name}")

    _iter_state = {"count": 0, "t": time.perf_counter()}

    def _on_batch_assembled(batch_idx: int, elapsed: float) -> None:
        if elapsed > 1.0:
            _log(
                f"  batch {batch_idx}: assembled in {_fmt_secs(elapsed)} "
                f"(DataLoader workers starting up)"
            )

    def _on_batch_end(loss, epoch: int, chunk_idx: int) -> None:
        _iter_state["count"] += 1
        n = _iter_state["count"]
        if use_wandb:
            wandb.log({"batch/loss_" + k: v for k, v in loss.item_dict().items()})
        if n % 100 == 0:
            elapsed = time.perf_counter() - _iter_state["t"]
            _log(
                f"  iter {n}: loss={loss.total.detach().item():.4f} "
                f"(last 100 iters: {_fmt_secs(elapsed)})"
            )
            _iter_state["t"] = time.perf_counter()

    trainer = Trainer(
        state, criterion,
        n_samples=cfg.training.n_samples,
        seg_weight=cfg.optimizer.seg_weight,
        on_batch_end=_on_batch_end,
        on_batch_assembled=_on_batch_assembled,
    )

    n_epochs = cfg.training.epochs
    _log(
        f"starting training: {n_epochs} epochs, batch_size="
        f"{cfg.training.batch_size}, n_samples={cfg.training.n_samples}"
    )
    save_every = cfg.training.save_checkpoint_every
    train_start = time.perf_counter()
    for epoch in range(n_epochs):
        ep_start = time.perf_counter()
        loss = trainer.train_epoch(epoch)
        state.step_scheduler()
        ep_elapsed = time.perf_counter() - ep_start
        _log(
            f"epoch {epoch + 1}/{n_epochs}: loss={loss:.4f} "
            f"({_fmt_secs(ep_elapsed)})"
        )
        if use_wandb:
            wandb.log({"epoch/loss": loss, "epoch": epoch + 1})
        if (
            val_dataset is not None
            and (epoch + 1) % cfg.training.validate_every == 0
        ):
            _run_validation(
                cfg, state, criterion, val_dataset,
                epoch=epoch + 1, out=out, use_wandb=use_wandb,
            )
        if save_every and (epoch + 1) % save_every == 0:
            ckpt_path = save_checkpoint(
                out / f"checkpoint_epoch_{epoch + 1}.pt",
                state=state, config=cfg, epoch=epoch + 1,
            )
            _log(f"saved intermediate checkpoint to {ckpt_path}")
    total = time.perf_counter() - train_start
    _log(f"training finished in {_fmt_secs(total)} ({_fmt_secs(total / max(n_epochs, 1))}/epoch)")

    # Latent-refresh pass: re-fit the per-subject latents against the now-final
    # decoder with augmentation off, then fold them back into the state so the
    # *saved* checkpoint holds clean, decoder-consistent latents. Downstream
    # (atlas / evaluate) then needs no refresh of its own.
    if cfg.training.refresh_epochs > 0:
        _log(
            f"latent-refresh pass: re-fitting {n_loaded} training latents against "
            f"the frozen decoder for {cfg.training.refresh_epochs} epoch(s) "
            f"(augmentation disabled) before final save ..."
        )
        refresh_start = time.perf_counter()
        refresh_dataset = _build_dataset(
            cfg, split="train", n_subjects=cfg.training.n_subjects.train,
            disable_augmentation=True,
        )
        refreshed = _refresh_train_latents(
            cfg, refresh_dataset, state.decoder,
            latents_init=state.latents.detach(),
            transformations_init=(
                state.transformations.detach()
                if state.transformations is not None else None
            ),
            fit_epochs=cfg.training.refresh_epochs,
        )
        with torch.no_grad():
            state.latents.copy_(refreshed.latents)
            if state.transformations is not None and refreshed.transformations is not None:
                state.transformations.copy_(refreshed.transformations)
        _log(
            f"latent-refresh done in {_fmt_secs(time.perf_counter() - refresh_start)}"
        )

    if use_wandb:
        wandb.finish()

    final_path = save_checkpoint(
        out / "checkpoint_final.pt",
        state=state, config=cfg, epoch=n_epochs,
    )
    _log(f"saved final checkpoint to {final_path}")

    if cfg.atlas.enabled and cfg.atlas.recipe is not None:
        _log(
            f"generating atlas: {len(cfg.atlas.recipe.ages)} ages × "
            f"{max(1, len(cfg.atlas.recipe.conditions) or 0) or 1} combo(s) ..."
        )
        atlas_start = time.perf_counter()
        written = generate_atlas(
            state, cfg.atlas.recipe,
            output_dir=out, epoch=n_epochs,
            temporal_condition=cfg.atlas.recipe.temporal_condition,
        )
        _log(
            f"wrote {len(written)} atlas file(s) to {out / 'atlas'} "
            f"in {_fmt_secs(time.perf_counter() - atlas_start)}"
        )
    return 0


# === fit ===


def _cmd_fit(args: argparse.Namespace) -> int:
    _log(f"loading checkpoint: {args.checkpoint}")
    ckpt = load_checkpoint(args.checkpoint, map_location=args.device or "cpu")
    cfg = ckpt.config
    if args.device:
        cfg.device = args.device
    trained_bbox = list(cfg.dataset_spec.world_bbox)  # frame the latents were trained in
    _log(f"swapping dataset to: {args.dataset}")
    cfg = _swap_dataset(cfg, Path(args.dataset))
    # Keep the trained coordinate frame: the latent grid was learned in the
    # checkpoint's world_bbox, so test/fit coords must use the same box (ignore
    # any world_bbox / `auto` in the new dataset YAML).
    cfg.dataset_spec.world_bbox = trained_bbox
    cfg.dataset_spec.auto_bbox = False
    _seed_everything(cfg.seed)

    out = (
        Path(args.output_dir).resolve() if args.output_dir
        else (Path(cfg.output_dir) / "fit").resolve()
    )
    out.mkdir(parents=True, exist_ok=True)
    cfg.output_dir = str(out)
    _log(f"output dir: {out}")

    _log(f"loading fit dataset (split={args.split}) ...")
    t0 = time.perf_counter()
    dataset = _build_dataset(
        cfg, split=args.split, n_subjects=args.n_subjects or (
        cfg.training.n_subjects.val if args.split == "val"
        else cfg.training.n_subjects.train
    ),
        skip_segmentation=args.skip_segmentation,
    )
    _log(f"fit dataset: {len(dataset)} subjects loaded in {_fmt_secs(time.perf_counter() - t0)}")

    # Rehydrate the decoder from the checkpoint.
    from .state import build_decoder
    n_cond_dims = len(cfg.conditions.enabled_specs())
    decoder = build_decoder(
        cfg.decoder,
        n_cond_dims=n_cond_dims,
        has_segmentation=cfg.dataset_spec.has_segmentation,
        device=cfg.device,
    )
    decoder.load_state_dict(ckpt.decoder_state_dict)

    state = build_fit_state(
        cfg, dataset=dataset, decoder=decoder,
        split=args.split,
        learn_conditions=not args.fix_conditions,
        n_epochs=args.epochs,
        latents_init=_centroid_latents_init(ckpt.latents, len(dataset)),
    )
    # Val/test fitting acts as if segmentation labels are unavailable: drop the
    # seg modality from the dataset (--skip_segmentation) AND from the criterion,
    # so the frozen decoder's seg head is never supervised here.
    criterion = _build_criterion(
        cfg,
        has_segmentation=cfg.dataset_spec.has_segmentation and not args.skip_segmentation,
    ).to(cfg.device)

    def _on_epoch_end(loss: float, ep: int) -> None:
        _log(f"fit epoch {ep + 1}/{args.epochs}: loss={loss:.4f}")

    fitter = LatentFitter(
        state, criterion,
        n_samples=cfg.training.n_samples,
        n_epochs=args.epochs,
        # Never optimise the seg term when adapting val/test latents.
        seg_weight=0.0,
        on_epoch_end=_on_epoch_end,
    )
    _log(
        f"starting fit: {args.epochs} epochs, "
        f"learn_conditions={not args.fix_conditions}, "
        f"skip_segmentation={args.skip_segmentation}"
    )
    fit_start = time.perf_counter()
    fitter.fit()
    _log(f"fit finished in {_fmt_secs(time.perf_counter() - fit_start)}")

    fit_path = save_checkpoint(
        out / "checkpoint_fit.pt",
        state=state, config=cfg, epoch=args.epochs,
    )
    _log(f"saved fit checkpoint to {fit_path}")
    return 0


def _read_tsv(cfg: TrainConfig) -> "pd.DataFrame":
    import pandas as pd
    return pd.read_csv(cfg.dataset_spec.tsv_file, sep="\t")


# === infer ===


def _cmd_infer(args: argparse.Namespace) -> int:
    _log(f"loading checkpoint: {args.checkpoint}")
    ckpt = load_checkpoint(args.checkpoint, map_location=args.device or "cpu")
    cfg = ckpt.config
    if args.device:
        cfg.device = args.device
    _seed_everything(cfg.seed)

    out = (
        Path(args.output_dir).resolve() if args.output_dir
        else (Path(cfg.output_dir) / "infer").resolve()
    )
    out.mkdir(parents=True, exist_ok=True)
    _log(f"output dir: {out}")

    _log(f"loading dataset (split={args.split}) ...")
    t0 = time.perf_counter()
    dataset = _build_dataset(
        cfg, split=args.split,
        n_subjects=ckpt.latents.shape[0],
    )
    _log(f"dataset: {len(dataset)} subjects loaded in {_fmt_secs(time.perf_counter() - t0)}")
    state = build_train_state(cfg, dataset=dataset)
    apply_to_state(ckpt, state, load_optimizer=False)

    spacing = args.spacing or [0.5, 0.5, 0.5]
    mask_reconstruction = (
        False if args.no_mask else bool(cfg.training.mask_reconstruction)
    )
    _log(
        f"reconstructing {len(dataset)} subject(s) at spacing {spacing} "
        f"(mask_reconstruction={mask_reconstruction}) ..."
    )
    infer_start = time.perf_counter()
    written: list[Path] = []
    for i in range(len(dataset)):
        s_start = time.perf_counter()
        recon, affine = reconstruct_subject(
            state, i, spacing=spacing,
            renormalize_per_modality=args.renormalize_per_modality,
            mask_reconstruction=mask_reconstruction,
        )
        sid = str(dataset.df.iloc[i]["subject_id"])
        files = save_reconstruction(
            recon, affine, out,
            subject_id=sid,
            intensity_modalities=cfg.dataset_spec.intensity_modalities,
            segmentation_modality=cfg.dataset_spec.segmentation_modality,
            epoch=ckpt.epoch,
            split=args.split,
        )
        written.extend(files)
        _log(
            f"  [{i + 1}/{len(dataset)}] {sid}: "
            f"{len(files)} file(s) in {_fmt_secs(time.perf_counter() - s_start)}"
        )
    _log(
        f"wrote {len(written)} files to {out} "
        f"in {_fmt_secs(time.perf_counter() - infer_start)}"
    )
    return 0


# === atlas ===


def _refresh_train_latents(
    cfg: TrainConfig,
    dataset: Data,
    decoder,
    *,
    latents_init: torch.Tensor,
    transformations_init: Optional[torch.Tensor],
    fit_epochs: int,
):
    """Re-fit training-subject latents against the frozen final decoder.

    This is the "latent-refresh pass". After training (optionally with data
    augmentation), the per-subject latents are both (a) *stale* w.r.t. the
    final decoder — under mini-batched training each latent was last updated
    against an older decoder state — and (b) *polluted* by any augmentation
    artifacts that got baked into them. Both hurt atlas quality, since the
    atlas is a Gaussian-weighted mean of training latents read against the
    final decoder.

    We freeze the decoder and re-optimise only the latents (+ transformations),
    warm-started from the trained values, over an augmentation-free dataset.
    Conditions are *not* learned: training subjects have known condition values,
    so the dataset's per-coord conditions are used as-is. The returned
    ``TrainState`` carries the refreshed latents; the train command folds them
    back into the saved checkpoint (see ``training.refresh_epochs``).

    Note: this scrubs the *per-subject* (latent-level) augmentation pollution,
    but cannot remove any *systematic* augmentation bias that lives in the
    frozen decoder weights (e.g. directional motion blur). Keep augmentations
    zero-centered (see the train configs) to keep that residual small.
    """
    fit_state = build_fit_state(
        cfg, dataset=dataset, decoder=decoder, split="train",
        latents_init=latents_init.clone(),
        transformations_init=(
            transformations_init.clone() if transformations_init is not None else None
        ),
        learn_conditions=False,
        n_epochs=fit_epochs,
    )
    criterion = _build_criterion(cfg)
    fitter = LatentFitter(
        fit_state, criterion,
        n_samples=cfg.training.n_samples,
        n_epochs=fit_epochs,
        seg_weight=cfg.optimizer.seg_weight,
        on_epoch_end=lambda loss, ep: _log(
            f"  latent-refresh {ep + 1}/{fit_epochs}: loss={loss:.4f}"
        ),
    )
    fitter.fit()
    return fit_state


def _cmd_atlas(args: argparse.Namespace) -> int:
    _log(f"loading checkpoint: {args.checkpoint}")
    ckpt = load_checkpoint(args.checkpoint, map_location=args.device or "cpu")
    cfg = ckpt.config
    if args.device:
        cfg.device = args.device

    recipe = (
        load_atlas_recipe(args.recipe) if args.recipe
        else cfg.atlas.recipe
    )
    if recipe is None:
        print("no atlas recipe: pass --recipe or bake one into the train config", file=sys.stderr)
        return 2

    out = (
        Path(args.output_dir).resolve() if args.output_dir
        else Path(cfg.output_dir).resolve()
    )
    out.mkdir(parents=True, exist_ok=True)
    _log(f"output dir: {out}")

    # The checkpoint's latents are already decoder-consistent and augmentation-
    # free (the train command runs the latent-refresh pass before saving, see
    # training.refresh_epochs), so the atlas trusts them as-is. Augmentation is
    # disabled here only as a belt-and-braces guard.
    _log(f"loading dataset (split=train) ...")
    t0 = time.perf_counter()
    dataset = _build_dataset(
        cfg, split="train", n_subjects=ckpt.latents.shape[0],
        disable_augmentation=True,
    )
    _log(f"dataset: {len(dataset)} subjects loaded in {_fmt_secs(time.perf_counter() - t0)}")
    state = build_train_state(cfg, dataset=dataset)
    apply_to_state(ckpt, state, load_optimizer=False)

    n_combos = 1
    if recipe.conditions:
        from functools import reduce
        n_combos = reduce(
            lambda a, b: a * b, (len(v) for v in recipe.conditions.values()), 1,
        )
    _log(
        f"generating atlas: {len(recipe.ages)} ages × {n_combos} combo(s), "
        f"spacing={recipe.spacing} ..."
    )
    atlas_start = time.perf_counter()
    written = generate_atlas(
        state, recipe,
        output_dir=out,
        temporal_condition=args.temporal_condition or recipe.temporal_condition,
        epoch=ckpt.epoch,
        renormalize_per_modality=args.renormalize_per_modality,
    )
    _log(
        f"wrote {len(written)} atlas file(s) to {out / 'atlas'} "
        f"in {_fmt_secs(time.perf_counter() - atlas_start)}"
    )
    return 0


# === evaluate ===


def _cmd_evaluate(args: argparse.Namespace) -> int:
    try:
        from .evaluator_ants import compute_subject_metrics
    except ImportError as e:
        print(f"evaluate requires the ants and skimage extras: {e}", file=sys.stderr)
        return 2
    import pandas as pd

    _log(f"loading checkpoint: {args.checkpoint}")
    ckpt = load_checkpoint(args.checkpoint, map_location=args.device or "cpu")
    cfg = ckpt.config
    if args.device:
        cfg.device = args.device

    refs_tsv = args.refs_tsv or cfg.dataset_spec.tsv_file
    refs = pd.read_csv(refs_tsv, sep="\t")
    refs = refs.set_index("subject_id")
    _log(f"loaded refs tsv: {len(refs)} subject rows from {refs_tsv}")

    out = (
        Path(args.output_dir).resolve() if args.output_dir
        else (Path(cfg.output_dir) / "evaluate").resolve()
    )
    out.mkdir(parents=True, exist_ok=True)
    _log(f"output dir: {out}")

    _log(f"loading dataset (split={args.split}) ...")
    t0 = time.perf_counter()
    dataset = _build_dataset(
        cfg, split=args.split, n_subjects=ckpt.latents.shape[0],
    )
    _log(f"dataset: {len(dataset)} subjects loaded in {_fmt_secs(time.perf_counter() - t0)}")
    state = build_train_state(cfg, dataset=dataset)
    apply_to_state(ckpt, state, load_optimizer=False)

    spacing = args.spacing or [0.5, 0.5, 0.5]
    mask_reconstruction = (
        False if args.no_mask else bool(cfg.training.mask_reconstruction)
    )
    _log(
        f"evaluating at spacing {spacing} "
        f"(mask_reconstruction={mask_reconstruction}) ..."
    )
    eval_start = time.perf_counter()
    reports: list[dict] = []
    for i in range(len(dataset)):
        sid = str(dataset.df.iloc[i]["subject_id"])
        if sid not in refs.index:
            print(f"no refs for {sid}; skipping", file=sys.stderr)
            continue
        s_start = time.perf_counter()
        recon, affine = reconstruct_subject(
            state, i, spacing=spacing,
            mask_reconstruction=mask_reconstruction,
        )
        ref_paths = {
            mod: str(refs.loc[sid, mod])
            for mod in cfg.dataset_spec.all_modality_keys
            if mod in refs.columns
        }
        metrics = compute_subject_metrics(
            subject_id=sid,
            pred_intensities=recon.intensities.detach().cpu().numpy(),
            pred_seg=(recon.seg_hard.detach().cpu().numpy()
                      if recon.seg_hard is not None else None),
            pred_affine=affine.detach().cpu().numpy(),
            ref_paths=ref_paths,
            intensity_modalities=cfg.dataset_spec.intensity_modalities,
            segmentation_modality=cfg.dataset_spec.segmentation_modality,
            label_names=cfg.dataset_spec.label_names,
        )
        reports.append(metrics.as_dict())
        _log(
            f"  [{i + 1}/{len(dataset)}] {sid}: "
            f"metrics computed in {_fmt_secs(time.perf_counter() - s_start)}"
        )

    report_path = out / "metrics.json"
    with open(report_path, "w") as f:
        json.dump(reports, f, indent=2)
    _log(
        f"wrote metrics for {len(reports)} subject(s) to {report_path} "
        f"in {_fmt_secs(time.perf_counter() - eval_start)}"
    )

    if args.train_checkpoint and cfg.validation.latent_analysis:
        _log(f"loading train checkpoint for latent analysis: {args.train_checkpoint}")
        train_ckpt = load_checkpoint(
            args.train_checkpoint, map_location=args.device or "cpu",
        )
        train_cfg = train_ckpt.config
        if args.device:
            train_cfg.device = args.device
        _log("loading train dataset for latent analysis ...")
        t0 = time.perf_counter()
        train_dataset = _build_dataset(
            train_cfg, split="train", n_subjects=train_ckpt.latents.shape[0],
        )
        _log(
            f"train dataset: {len(train_dataset)} subjects loaded in "
            f"{_fmt_secs(time.perf_counter() - t0)}"
        )
        train_state = build_train_state(train_cfg, dataset=train_dataset)
        apply_to_state(train_ckpt, train_state, load_optimizer=False)

        latent_results = _run_latent_analysis(
            cfg,
            train_dataset=train_dataset,
            val_dataset=dataset,
            train_latents=train_state.latents.detach(),
            val_latents=state.latents.detach(),
        )
        if latent_results:
            la_path = out / "latent_analysis.json"
            with open(la_path, "w") as f:
                json.dump([r.as_dict() for r in latent_results], f, indent=2)
            _log(f"wrote latent analysis to {la_path}")
    elif cfg.validation.latent_analysis:
        _log(
            "skipping latent analysis: --train-checkpoint not provided "
            "(needed to access train latents)"
        )
    return 0


# === arg parsing ===


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cinema", description="CINeMA command-line interface.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train", help="Train a decoder + latents from a train config.")
    pt.add_argument("config", type=Path, help="Train config YAML.")
    pt.add_argument("--set", action="append", default=[],
                    help="Override a config key, e.g. training.epochs=50.")
    pt.add_argument("--output-dir", type=Path, default=None)
    pt.add_argument("--resume", type=Path, default=None,
                    help="Checkpoint to resume from.")
    pt.set_defaults(func=_cmd_train)

    pf = sub.add_parser("fit", help="Fit latents for a new dataset with a frozen decoder.")
    pf.add_argument("checkpoint", type=Path)
    pf.add_argument("--dataset", type=Path, required=True,
                    help="Dataset YAML the new subjects come from.")
    pf.add_argument("--epochs", type=int, required=True)
    pf.add_argument("--split", default="test")
    pf.add_argument("--n-subjects", type=int, default=None,
                    help="Subject count; defaults to all rows in the new TSV.")
    pf.add_argument("--skip-segmentation", action="store_true",
                    help="Fit even though the new subjects have no seg ground truth.")
    pf.add_argument("--fix-conditions", action="store_true",
                    help="Use TSV-provided conditions instead of learning them.")
    pf.add_argument("--output-dir", type=Path, default=None)
    pf.add_argument("--device", default=None)
    pf.set_defaults(func=_cmd_fit)

    pi = sub.add_parser("infer", help="Reconstruct every subject in the checkpoint's dataset.")
    pi.add_argument("checkpoint", type=Path)
    pi.add_argument("--output-dir", type=Path, default=None)
    pi.add_argument("--spacing", type=float, nargs=3, default=None,
                    metavar=("SX", "SY", "SZ"))
    pi.add_argument("--split", default="train")
    pi.add_argument("--renormalize-per-modality", action="store_true")
    pi.add_argument("--no-mask", action="store_true",
                    help="Disable largest-component masking (default: follow "
                         "training.mask_reconstruction from the checkpoint).")
    pi.add_argument("--device", default=None)
    pi.set_defaults(func=_cmd_infer)

    pa = sub.add_parser("atlas", help="Generate atlas NIfTIs from a checkpoint.")
    pa.add_argument("checkpoint", type=Path)
    pa.add_argument("--recipe", type=Path, default=None,
                    help="Atlas recipe YAML; falls back to the train config's recipe.")
    pa.add_argument("--output-dir", type=Path, default=None)
    pa.add_argument("--temporal-condition", default=None,
                    help="condition the atlas ages sweep; defaults to the recipe's "
                         "temporal_condition (itself defaulting to scan_age)")
    pa.add_argument("--renormalize-per-modality", action="store_true")
    pa.add_argument("--device", default=None)
    pa.set_defaults(func=_cmd_atlas)

    pe = sub.add_parser("evaluate", help="PSNR/SSIM/Dice against reference NIfTIs.")
    pe.add_argument("checkpoint", type=Path)
    pe.add_argument("--refs-tsv", type=Path, default=None,
                    help="TSV with subject_id + modality columns pointing at ref NIfTIs. "
                         "Defaults to the dataset TSV embedded in the checkpoint.")
    pe.add_argument("--train-checkpoint", type=Path, default=None,
                    help="Train checkpoint to source train latents from for "
                         "latent-space analysis steps in cfg.validation.latent_analysis. "
                         "Skipped if not provided.")
    pe.add_argument("--output-dir", type=Path, default=None)
    pe.add_argument("--spacing", type=float, nargs=3, default=None,
                    metavar=("SX", "SY", "SZ"))
    pe.add_argument("--split", default="train")
    pe.add_argument("--no-mask", action="store_true",
                    help="Disable largest-component masking (default: follow "
                         "training.mask_reconstruction from the checkpoint).")
    pe.add_argument("--device", default=None)
    pe.set_defaults(func=_cmd_evaluate)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
