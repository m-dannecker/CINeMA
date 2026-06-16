# CINeMA

**C**onditional **I**mplicit **Ne**ural **M**ulti-Modal **A**tlas for a Spatio-Temporal Representation of the Perinatal Brain

CINeMA is a deep learning framework for building conditional implicit neural multi-modal atlases that provide spatio-temporal representations of the developing brain. The framework uses Implicit Neural Representations (INRs) to create continuous, smooth atlases that can be conditioned on developmental factors such as scan age, birth age, lateral ventricular volume (to model ventriculomegaly), and agenesis of the corpus callosum.

## Features

- **Multi-modal reconstruction**: joint T1w / T2w / segmentation learning from a single decoder
- **Conditional atlas generation**: continuous atlases conditioned on age and arbitrary numeric covariates
- **Atlas validation via growth curves**: overlay the generated atlas's own per-tissue volumes on empirical volume-vs-age curves fit from the training cohort, to check anatomical plausibility (e.g. total brain size) at a glance
- **Resolution-agnostic INR decoder**: SIREN-based MLP + FiLM modulation from per-subject latent grids
- **Train once, re-fit later**: load a trained checkpoint and fit per-subject latents for new (possibly seg-free) test subjects with the decoder frozen
- **Typed, versioned configs**: dataclass-backed YAML configs with explicit dataset / train / atlas separation and `--set` overrides
- **First-class CLI**: `python -m cinema <subcommand>` drives training, fitting, inference, atlas generation, and evaluation

## Precomputed Atlases
Pre-computed temporal atlases modeling neurotypical fetal and neonatal brain development — as well as ventriculomegaly (VM) and agenesis of the corpus callosum (ACC) — are available on the [Zenodo repository](https://zenodo.org/records/17023473).

## Table of Contents

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Output](#output)
- [Troubleshooting](#troubleshooting)
- [Known limitations](#known-limitations)
- [Citation](#citation)

## Installation

### Prerequisites

- Python 3.10
- CUDA-compatible GPU (recommended)
- Conda or Miniconda

### Option 1: Conda (recommended)

```bash
git clone <repository-url>
cd CINeMA_2
conda env create -f environment.yml
conda activate cinema
```

### Option 2: pip

```bash
git clone <repository-url>
cd CINeMA_2
pip install -r requirements.txt
```

## Data Preparation

### Supported datasets

CINeMA has been validated on the dHCP (developing Human Connectome Project) neonatal and fetal cohorts, and on an in-house fetal cohort with pathology labels (e.g. ventriculomegaly, agenesis of the corpus callosum). Any new dataset needs:

- **Subjects table (CSV or TSV)** — metadata + NIfTI paths per subject. Must include every value you want to either condition on or constrain sampling by. `.csv` is read as comma-separated (Excel-friendly), `.tsv` as tab-separated.
- **Subject-IDs YAML** — maps split names (`train`, `val`, optional `test`) to lists of subject IDs referenced in the subjects table.
- **Dataset YAML** — declares modalities, conditions, constraints, and normalisation spans. Templates live under [configs/datasets/](configs/datasets/).

### Preprocessing expectations

- All volumes in NIfTI format, roughly in the same orientation.
- No resampling required — INRs operate directly on the subject's voxel grid, so heterogeneous resolutions and spacings are fine.
- Conditionable properties (e.g. ventricular volume) must be pre-extracted and written into the subjects table.
- Fine rigid / rigid+scale registration is learned by the model (`decoder.tf_dim: 6` or `9`); no prior atlas is needed.

## Configuration

CINeMA 2 splits configs into three layered YAMLs — this replaces the legacy monolithic `config_atlas.yaml` + `config_data.yaml` pair.

```
configs/
├── datasets/        # dataset specs (modalities, conditions, constraints, normalisation)
│   └── example.yaml
├── train/           # training configs (decoder + optimizer + training + pointers to dataset/atlas)
│   └── example.yaml
├── atlas/           # standalone atlas recipes (ages × condition combos)
│   └── example.yaml
├── GenericSubjectsTable.csv   # subjects-table template (header only — add your own rows)
└── example_subject_ids.yaml   # split definition template (train/val/test ID lists)
```

The `example` configs are a self-contained, commented template — copy and adapt them to your own dataset. They are syntactically valid but ship with a header-only table, so they document the workflow rather than run as-is.

### Dataset YAML

- `dataset.modalities.intensity` is the list of reconstructed intensity channels; `dataset.modalities.segmentation` is a single optional seg modality (`null` for test subjects that lack GT seg).
- `conditions` are values consumed by the decoder. Each entry declares a normalisation (`minmax`, `age_relative`, or `identity`) and a `cond_scale`. `use_in_decoder: false` stores a condition in the latent bookkeeping without concatenating it onto the decoder input (typical for `scan_age`, which is handled via latent regression at atlas time).
- `constraints` filter which subjects are drawn. `sampling: {type: uniform_fillup, priority: K}` yields uniformly-binned sampling along that axis. A condition must also be constrained so its normalisation range is well-defined.

### Train YAML

Holds `decoder`, `optimizer`, `training`, `validation`, `atlas`, and `logging` blocks, plus a `dataset_config:` pointer (relative to the train YAML) and an optional `atlas.recipe:` pointer. See [configs/train/example.yaml](configs/train/example.yaml) for a commented reference.

### Atlas recipe

A standalone YAML holding `ages`, `conditions` (Cartesian product across all condition axes), `spacing`, `gaussian_span`, `n_max`, and `mask_reconstruction`. One atlas volume is emitted per `(age × condition-combo)`, named by the **actual condition values** (e.g. `T2w_ExamType_num=-1.0_GA_MRI=25-26_ep=5.nii.gz`; age-only atlases fall back to `cond=0`). `gaussian_span` is in raw condition units (weeks for `scan_age`). `n_max` caps how many nearest training subjects contribute to `mean_latent`. An optional `growth_curves:` block enables atlas validation (see [Atlas validation](#atlas-validation-growth-curves)).

## Usage

All commands are invoked through the `cinema` package module:

```bash
python -m cinema <subcommand> [args]
```

### Train

```bash
python -m cinema train configs/train/example.yaml
```

Override any nested key with repeatable `--set section.key=value` flags. Values are YAML-parsed, so lists, nulls, and numerics just work:

```bash
python -m cinema train configs/train/example.yaml \
  --set training.epochs=50 \
  --set decoder.hidden_size=1024 \
  --set optimizer.lr_inr=2e-4 \
  --output-dir ./output/example_run1
```

The resolved config (including any `--set` overrides) is written to `<output_dir>/config.yaml` at the start of every run.

#### Recommended settings

- **Train full-batch.** Keep `training.batch_size: 0` (= full batch, all training subjects per step). A `batch_size` smaller than the training set samples each gradient step from only a subset of subjects and **measurably degrades reconstruction quality**. If GPU memory is the limit, lower `training.n_samples` (coords per optimizer step) rather than `batch_size`.
- **Latent-grid resolution drives validation quality.** A larger spatial latent grid — e.g. `decoder.latent_dim: [256, 7, 7, 7]` instead of the `[256, 3, 3, 3]` default — **markedly improves reconstruction of unseen (validation/test) subjects, for both intensity structure and segmentation labels**, at the cost of GPU memory.
- **Fit the latent grid to the brain.** Set `dataset.world_bbox: auto` to compute a tight (anisotropic) bounding box from the actual training subjects at run start — a fixed 155 mm box wastes ~75–85% of the grid on empty space around the brain. The resolved box is logged and frozen into the checkpoint. Combine with a length-2 grid `decoder.latent_dim: [channels, max_size]` (e.g. `[256, 9]`): it auto-expands to an anisotropic `[channels, lx, ly, lz]` matching the box's aspect ratio (largest axis → `max_size` cells), so the grid concentrates resolution where the anatomy is.

Resume from a checkpoint:
```bash
python -m cinema train configs/train/example.yaml --resume ./output/<run>/checkpoint_final.pt
```

### Fit (test-time latent fitting)

Point a trained checkpoint at a new dataset YAML and fit per-subject latents (+ optional transformations / conditions) with the decoder frozen. Useful for unseen subjects that may not have segmentation ground truth.

```bash
python -m cinema fit ./output/<run>/checkpoint_final.pt \
  --dataset configs/datasets/example.yaml \
  --epochs 200 \
  --split test \
  --skip-segmentation          # new subjects without seg GT
  # --fix-conditions           # use TSV-provided conditions instead of learning them
```

### Inference

Reconstruct every subject from the checkpoint's own dataset:

```bash
python -m cinema infer ./output/<run>/checkpoint_final.pt \
  --spacing 0.5 0.5 0.5 \
  --renormalize-per-modality
```

### Atlas generation

Generate atlases using the train-config's recipe, or supply an explicit recipe:

```bash
python -m cinema atlas ./output/<run>/checkpoint_final.pt
python -m cinema atlas ./output/<run>/checkpoint_final.pt --recipe configs/atlas/example.yaml
```

### Atlas validation (growth curves)

To sanity-check that an atlas is anatomically plausible (does its total brain volume, CSF, WM, … track the real cohort at each age?), add a `growth_curves:` block to the atlas recipe. During atlas generation CINeMA then measures per-tissue volumes from the **training** segmentations, fits mean ± SD volume-vs-age curves, and overlays the atlas's **own** measured volumes (one marker series per condition value):

```yaml
# in configs/atlas/<dataset>.yaml
growth_curves:
  enabled: true
  group_by: ExamType_num                       # optional: one curve per value of a
                                               # categorical condition (omit for a
                                               # single curve, e.g. a continuous sweep)
  group_labels: {-1.0: PRE_OP, 1.0: POST_OP}   # optional display names
  tissues: [TotalBrain, CSF, WM, cGM]          # optional; default = all foreground labels + TotalBrain
  kernel_sigma: 2.0                            # smoothing width along the age axis (weeks)
```

This runs automatically at the end of `train` (when `atlas.enabled`) and `atlas`, writing to `<output_dir>/atlas/growth_curves/`:

- `curve_<tissue>.png` — empirical mean ± SD band + per-subject scatter, with atlas markers labelled by condition value
- `tissue_volumes.csv` — per-training-subject volumes (mm³); `atlas_volumes.csv` — per atlas frame
- `growth_curves_reference.json` — the fitted curves (age grid, μ, σ) per tissue/group

The volumes use the dataset's `segmentation_classes.label_names` as the tissue→label map, and `TotalBrain` is the sum of all foreground labels.

The same plots can be produced ad-hoc (and against an external reference cohort) with `scripts/tissue_growth_curves.py`; the two-step `scripts/extract_lv_condition.py` → `scripts/compute_lv_zscore.py` pipeline shows how to derive a conditioning column (lateral-ventricle z-score) from segmentations.

### Evaluation

Reconstruct subjects in the checkpoint's dataset, register to reference NIfTIs with ANTs, and write PSNR / SSIM / Dice into `metrics.json`:

```bash
python -m cinema evaluate ./output/<run>/checkpoint_final.pt \
  --refs-tsv ./references.tsv
```

## Output

Each `train` run writes to `output_dir` (defaults to `./output/<config_stem>_<timestamp>/`):

```
output_dir/
├── config.yaml                   # merged + normalised TrainConfig (dataset spec inlined)
├── checkpoint_final.pt           # final checkpoint (TrainConfig + state dicts, pickled)
├── checkpoint_epoch_{N}.pt       # periodic checkpoints when training.save_checkpoint_every is set
└── atlas/                        # created when atlas.enabled: true
    ├── {modality}_{cond}={val}_{temporal}={age-range}_ep={epoch}.nii.gz
    ├── ages.json                  # ordered age list for the stacked 4D time axis
    └── growth_curves/             # created when recipe growth_curves.enabled: true
        ├── curve_{tissue}.png
        ├── tissue_volumes.csv / atlas_volumes.csv
        └── growth_curves_reference.json
```

One 4D NIfTI is written per `(modality × condition-combo)`, with `ages` stacked along the trailing axis (scrub through it in ITK-SNAP / fsleyes). Downstream subcommands write into their own output directories:

- `fit` → `fit_checkpoint.pt`
- `infer` → one NIfTI per subject-modality
- `atlas` → `{modality}_{cond}={val}_{temporal}={age-range}_ep={epoch}.nii.gz` (+ optional `growth_curves/`)
- `evaluate` → `metrics.json`

### Logging

Set `logging.enabled: true` and provide `wandb_entity` / `project` in the train YAML to stream losses and metrics to Weights & Biases.

## Troubleshooting

### Common issues

1. **CUDA out of memory**
   - Reduce `training.n_samples` (coords per optimizer step) — this is the primary memory lever and does not change batch composition
   - **Keep full batch** (`training.batch_size: 0`); a smaller batch degrades reconstruction quality, so prefer lowering `n_samples` (and, only if needed, `decoder.latent_dim` / `decoder.hidden_size`) over shrinking the batch
   - During validation/inference, `predict_volume` auto-scales its chunk size to the latent-grid size, so larger grids do not OOM the reconstruction path
   - For atlases, reduce the number of ages × condition combos in the recipe or coarsen `spacing`

2. **Configuration errors**
   - Every condition referenced by the decoder must also appear as a constraint (so min/max bounds are defined)
   - Conditions referenced in the atlas recipe must be declared as `conditions` in the dataset YAML
   - Subject IDs in the subject-IDs YAML must exist in the dataset's subjects table

3. **Data loading**
   - All NIfTI volumes must be 3D and share an affine per subject across modalities
   - Modalities are declared as `intensity: [...]` plus a single optional `segmentation: <name|null>` — there is no "last entry is segmentation" convention; follow the provided templates

4. **`fit` against a seg-free dataset**
   - Use `--skip-segmentation`; the decoder still predicts segmentation logits but they are not used in the loss

5. **Hallucinations / unclean background outside the brain**
   - The reconstruction is masked to the largest connected segmentation component. For this to work the decoder must learn a background label — supervised by a thin background "halo" ring the data pipeline paints around the brain. That halo class **must have a non-zero `class_weights` entry**, or the decoder never learns background and predicts tissue everywhere outside (CINeMA prints a loud warning at dataset build if this is misconfigured). dHCP uses a dedicated `BG` class; datasets without one (the halo falls back to label 0) must give label 0 a non-zero weight.
   - For lower-quality segmentations where thin "bridges" keep hallucinations attached to the brain, raise `training.mask_open_radius` (a morphological opening, in voxels, applied before component selection) and/or `training.mask_halo_width` (wider background supervision; needs retraining).
   - `training.intensity_floor` zeros a faint background haze (and strips the corresponding seg) below a threshold before masking — keep it under the darkest real tissue (e.g. `0.05`).

## Known limitations

Open modelling limitations include validation/test reconstruction sharpness on
unseen subjects and the full-batch training requirement (gradient accumulation
for very large cohorts within bounded host memory is currently unimplemented).

## Citation

If you use CINeMA in your research, please cite:

```bibtex
@article{dannecker2025cinema,
  title={CINeMA: Conditional Implicit Neural Multi-Modal Atlas for a Spatio-Temporal Representation of the Perinatal Brain},
  author={Dannecker, Maik and Sideri-Lampretsa, Vasiliki and Starck, Sophie and Mihailov, Angeline and Milh, Mathieu and Girard, Nadine and Auzias, Guillaume and Rueckert, Daniel},
  journal={IEEE Transactions on Medical Imaging},
  year={2025},
  publisher={IEEE}
}
```

## License

This project is licensed under the terms specified in the LICENSE file.

For questions and support, please open an issue on the GitHub repository.
