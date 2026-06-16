"""
Typed configuration objects and YAML loaders for CINeMA.

Two YAML schemas are supported:

- **Dataset config** (``configs/datasets/<name>.yaml``): describes the data
  source — modalities, world bbox, conditions (with their own normalization
  strategies and ``cond_scale``), and constraints (filtering + balanced
  sampling). This is what tells CINeMA *what* to learn from.

- **Train config** (``configs/train/<name>.yaml``): references a dataset
  config and adds training/optim/decoder/validation/atlas blocks. This is what
  tells CINeMA *how* to learn it.

A standalone **atlas recipe** YAML (``configs/atlas/<name>.yaml``) can also be
loaded for the ``cinema atlas`` CLI command — same schema as ``train.atlas.recipe``.

Constraints and conditions are deliberately decoupled:

- *Conditions* own their normalization (so inference uses physical units with
  no help from the constraints block).
- *Constraints* purely filter rows and drive balanced sampling (``priority`` /
  ``bins`` / ``uniform_fillup``). They never set normalization bounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional, Union

import yaml

from .conditioning import (
    AgeRelativeNormalization,
    ConditionRegistry,
    ConditionSpec,
)


# === Dataset side ===


@dataclass
class SamplingSpec:
    """Per-constraint sampling rule used by ``Data.sample_subjects``."""

    type: str
    priority: int
    bins: Optional[int] = None

    @classmethod
    def from_dict(cls, d: dict) -> "SamplingSpec":
        return cls(
            type=str(d["type"]),
            priority=int(d["priority"]),
            bins=int(d["bins"]) if d.get("bins") is not None else None,
        )


@dataclass
class ConstraintSpec:
    """One filtering constraint. Numeric ``[min, max]`` or categoric ``values``."""

    name: str
    min: Optional[float] = None
    max: Optional[float] = None
    values: Optional[list] = None
    sampling: Optional[SamplingSpec] = None

    @property
    def is_categoric(self) -> bool:
        return self.values is not None

    @property
    def is_numeric(self) -> bool:
        return not self.is_categoric

    @classmethod
    def from_dict(cls, name: str, d: dict) -> "ConstraintSpec":
        spec = cls(name=name)
        if "values" in d:
            spec.values = list(d["values"])
        if "min" in d:
            spec.min = float(d["min"])
        if "max" in d:
            spec.max = float(d["max"])
        if "sampling" in d:
            spec.sampling = SamplingSpec.from_dict(d["sampling"])
        if spec.is_categoric and (spec.min is not None or spec.max is not None):
            raise ValueError(
                f"constraint {name!r} has both 'values' and 'min'/'max'; pick one"
            )
        if spec.is_numeric and (spec.min is None or spec.max is None):
            raise ValueError(
                f"constraint {name!r} is numeric but missing 'min' or 'max'"
            )
        return spec


class ConstraintSet:
    """Ordered collection of ``ConstraintSpec``."""

    def __init__(self, specs: list[ConstraintSpec]):
        self.specs: list[ConstraintSpec] = list(specs)
        self._by_name: dict[str, ConstraintSpec] = {s.name: s for s in specs}
        if len(self._by_name) != len(specs):
            raise ValueError("duplicate constraint names")

    def __getitem__(self, name: str) -> ConstraintSpec:
        return self._by_name[name]

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def __iter__(self):
        return iter(self.specs)

    def __len__(self):
        return len(self.specs)

    def with_sampling(self) -> list[ConstraintSpec]:
        """Constraints participating in balanced sampling, sorted by priority desc."""
        active = [s for s in self.specs if s.sampling is not None]
        return sorted(active, key=lambda s: s.sampling.priority, reverse=True)

    @classmethod
    def from_dict(cls, d: dict) -> "ConstraintSet":
        return cls([ConstraintSpec.from_dict(name, body) for name, body in d.items()])


@dataclass
class DatasetSpec:
    """Static description of a dataset.

    ``segmentation_modality`` may be ``None`` for datasets without segmentation
    (e.g. test subjects fed to ``cinema fit`` without ground-truth labels). The
    decoder still emits a seg head if the trained model had one — the loss for
    that head is just dropped at fit time.
    """

    name: str
    tsv_file: str
    subject_ids: str
    intensity_modalities: list[str]
    segmentation_modality: Optional[str]
    world_bbox: list[float]
    normalize_intensities: str = "minmax"
    label_names: Optional[list[str]] = None
    class_weights: Optional[list[float]] = None
    # When `world_bbox: auto`, the bbox is computed at train time from the cohort's
    # foreground extent (+ world_bbox_margin) and frozen into the checkpoint. See
    # cli._resolve_auto_bbox. auto_bbox is set back to False once resolved.
    auto_bbox: bool = False
    world_bbox_margin: float = 0.15

    @property
    def all_modality_keys(self) -> list[str]:
        keys = list(self.intensity_modalities)
        if self.segmentation_modality is not None:
            keys.append(self.segmentation_modality)
        return keys

    @property
    def has_segmentation(self) -> bool:
        return self.segmentation_modality is not None

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetSpec":
        modalities = d.get("modalities")
        if not isinstance(modalities, dict):
            raise ValueError(
                "dataset.modalities must be a mapping with 'intensity' (list) "
                "and optional 'segmentation' (string or null)"
            )
        intensity = list(modalities["intensity"])
        seg = modalities.get("segmentation")
        seg = str(seg) if seg else None
        seg_classes = d.get("segmentation_classes") or {}
        wb = d["world_bbox"]
        auto_bbox = isinstance(wb, str) and wb.strip().lower() == "auto"
        world_bbox = [0.0, 0.0, 0.0] if auto_bbox else [float(x) for x in wb]
        spec = cls(
            name=str(d["name"]),
            tsv_file=str(d["tsv_file"]),
            subject_ids=str(d["subject_ids"]),
            intensity_modalities=intensity,
            segmentation_modality=seg,
            world_bbox=world_bbox,
            normalize_intensities=str(d.get("normalize_intensities", "minmax")),
            label_names=(list(seg_classes["label_names"])
                         if "label_names" in seg_classes else None),
            class_weights=(list(seg_classes["class_weights"])
                           if "class_weights" in seg_classes else None),
            auto_bbox=auto_bbox,
            world_bbox_margin=float(d.get("world_bbox_margin", 0.15)),
        )
        if spec.has_segmentation and spec.label_names is None:
            raise ValueError(
                f"dataset {spec.name!r} has segmentation modality "
                f"{spec.segmentation_modality!r} but no segmentation_classes.label_names"
            )
        return spec


# === Training side ===


@dataclass
class SchedulerConfig:
    type: str = "cosine"
    eta_min: float = 1e-5

    @classmethod
    def from_dict(cls, d: dict) -> "SchedulerConfig":
        return cls(
            type=str(d.get("type", "cosine")),
            eta_min=float(d.get("eta_min", 1e-5)),
        )


@dataclass
class OptimizerConfig:
    lr_inr: float
    lr_latent: float
    lr_tf: float
    inr_weight_decay: float = 0.0
    latent_weight_decay: float = 0.0
    tf_weight_decay: float = 0.0
    tf_weight: float = 0.0
    loss_metric: str = "l1"
    seg_weight: float = 1.0
    re_init_latents: bool = False
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    @classmethod
    def from_dict(cls, d: dict) -> "OptimizerConfig":
        return cls(
            lr_inr=float(d["lr_inr"]),
            lr_latent=float(d["lr_latent"]),
            lr_tf=float(d["lr_tf"]),
            inr_weight_decay=float(d.get("inr_weight_decay", 0.0)),
            latent_weight_decay=float(d.get("latent_weight_decay", 0.0)),
            tf_weight_decay=float(d.get("tf_weight_decay", 0.0)),
            tf_weight=float(d.get("tf_weight", 0.0)),
            loss_metric=str(d.get("loss_metric", "l1")),
            seg_weight=float(d.get("seg_weight", 1.0)),
            re_init_latents=bool(d.get("re_init_latents", False)),
            scheduler=SchedulerConfig.from_dict(d.get("scheduler", {})),
        )


@dataclass
class DecoderConfig:
    tf_dim: int
    cnn_kernel_size: int
    latent_dim: list[int]
    in_dim: int
    out_dim: list[int]
    hidden_size: int
    num_hidden_layers: int
    modulated_layers: list[int]
    omega: list[float]

    @classmethod
    def from_dict(cls, d: dict) -> "DecoderConfig":
        latent_dim = [int(x) for x in d["latent_dim"]]
        if len(latent_dim) not in (2, 4):
            raise ValueError(
                "decoder.latent_dim must be [channels, lx, ly, lz] (explicit grid) "
                "or [channels, max_size] (auto-anisotropic: the largest world_bbox "
                f"axis gets max_size cells, others scale down); got {latent_dim}"
            )
        return cls(
            tf_dim=int(d["tf_dim"]),
            cnn_kernel_size=int(d["cnn_kernel_size"]),
            latent_dim=latent_dim,
            in_dim=int(d["in_dim"]),
            out_dim=[int(x) for x in d["out_dim"]],
            hidden_size=int(d["hidden_size"]),
            num_hidden_layers=int(d["num_hidden_layers"]),
            modulated_layers=[int(x) for x in d["modulated_layers"]],
            omega=[float(x) for x in d["omega"]],
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class NSubjectsConfig:
    train: int
    val: int


@dataclass
class TrainingConfig:
    """Training-loop hyperparameters.

    ``refresh_epochs`` controls the *latent-refresh pass* run once at the end of
    training, just before the final checkpoint is written: the per-subject
    latents are re-fit against the now-frozen decoder with data augmentation
    disabled, then saved into the checkpoint. This removes both the mini-batch
    staleness (each latent was last updated against an older decoder state) and
    augmentation pollution, so any downstream consumer (atlas, evaluate) reads
    clean, decoder-consistent latents straight from the checkpoint — no further
    refresh needed. 0 disables it.
    """

    epochs: int
    validate_every: int
    batch_size: int
    n_samples: int
    num_workers: int
    n_subjects: NSubjectsConfig
    save_checkpoint_every: Optional[int] = None
    mask_reconstruction: bool = True
    # Segmentation-masking knobs (see cinema/data.py `_add_background_halo` and
    # cinema/models/inr_decoder.py `mask_by_largest_component`):
    # - `mask_halo_width`: width (Gaussian sigma, voxels) of the background ring
    #   painted around the foreground at *training* time. Wider = thicker
    #   "outside-brain = background" supervision, which yields a cleaner boundary
    #   and fewer foreground bridges. Bump it for lower-quality segmentations.
    # - `mask_open_radius`: erosion radius (voxels) of a morphological *opening*
    #   applied before largest-connected-component selection at *inference* time.
    #   Severs thin foreground bridges so bridged hallucinations get dropped.
    #   0 = off (no opening). This is the lever for "bridges survive masking".
    # (The final inference mask is also lightly Gaussian-smoothed at a fixed
    #  sigma of 1.0 voxel inside `mask_by_largest_component` — cosmetic only, not
    #  exposed here because it does nothing for bridges.)
    # - `intensity_floor`: at inference, voxels whose brightest intensity channel
    #   is below this are zeroed *and* set to background in the seg, before
    #   largest-component masking. Rounds the faint background haze (~0.02) to
    #   zero and strips faint hallucinations from the seg so masking catches them.
    #   Keep below the darkest real tissue (e.g. ~0.05; dark brain is rarely
    #   < 0.1 for min-max-normalised intensities). 0 = off.
    mask_halo_width: float = 1.5
    mask_open_radius: int = 0
    intensity_floor: float = 0.0
    refresh_epochs: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingConfig":
        return cls(
            epochs=int(d["epochs"]),
            validate_every=int(d["validate_every"]),
            batch_size=int(d["batch_size"]),
            n_samples=int(d["n_samples"]),
            num_workers=int(d["num_workers"]),
            n_subjects=NSubjectsConfig(
                train=int(d["n_subjects"]["train"]),
                val=int(d["n_subjects"]["val"]),
            ),
            save_checkpoint_every=(
                int(d["save_checkpoint_every"])
                if d.get("save_checkpoint_every") is not None
                else None
            ),
            mask_reconstruction=bool(d.get("mask_reconstruction", True)),
            mask_halo_width=float(d.get("mask_halo_width", 1.5)),
            mask_open_radius=int(d.get("mask_open_radius", 0)),
            intensity_floor=float(d.get("intensity_floor", 0.0)),
            refresh_epochs=int(d.get("refresh_epochs", 0)),
        )


@dataclass
class LatentAnalysisStep:
    """One disentangled latent-analysis call run during validation cycles."""

    type: str          # 'nca' | 'regressor' | 'predict_ext'
    target: str        # condition name
    extras: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "LatentAnalysisStep":
        d = dict(d)
        t = d.pop("type")
        target = d.pop("target")
        return cls(type=str(t), target=str(target), extras=d)


@dataclass
class ValidationConfig:
    """Each evaluation step is independently togglable. ``compute_metrics`` and
    ``save_imgs`` are honoured per-call by the evaluator regardless of
    segmentation availability."""

    fit_epochs: int = 0
    reconstruct: bool = True
    compute_metrics: bool = True
    save_imgs: bool = True
    latent_analysis: list[LatentAnalysisStep] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "ValidationConfig":
        return cls(
            fit_epochs=int(d.get("fit_epochs", 0)),
            reconstruct=bool(d.get("reconstruct", True)),
            compute_metrics=bool(d.get("compute_metrics", True)),
            save_imgs=bool(d.get("save_imgs", True)),
            latent_analysis=[
                LatentAnalysisStep.from_dict(s) for s in (d.get("latent_analysis") or [])
            ],
        )


@dataclass
class GrowthCurvesConfig:
    """Tissue-volume growth-curve overlay for atlas validation.

    When ``enabled``, the atlas step measures per-tissue volumes from the
    *training* segmentations (the same cohort the atlas is regressed from), fits
    mean +/- SD curves vs the temporal axis, and overlays the generated atlas's
    own measured volumes — see ``cinema/growth_curves.py``.

    ``group_by`` is an optional column/condition name: if it has few unique
    values (categorical, e.g. ``ExamType_numeric``) one curve is fit per value,
    and each atlas combination is routed onto the matching curve. Leave it unset
    for a single overall curve (the right choice for a continuous sweep such as
    ``lv_z``, where the atlas variants instead show up as a spread of points).
    ``group_labels`` maps raw group values to display names
    (e.g. ``{-1.0: PRE_OP, 1.0: POST_OP}``). ``tissues`` defaults to every
    non-background entry of the dataset's ``label_names`` plus ``TotalBrain``.
    """

    enabled: bool = False
    group_by: Optional[str] = None
    group_labels: dict = field(default_factory=dict)
    tissues: Optional[list[str]] = None
    kernel_sigma: float = 2.0

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "GrowthCurvesConfig":
        d = d or {}
        labels: dict = {}
        for k, v in (d.get("group_labels") or {}).items():
            try:
                labels[float(k)] = str(v)   # align numeric keys with condition values
            except (TypeError, ValueError):
                labels[k] = str(v)
        tissues = d.get("tissues")
        return cls(
            enabled=bool(d.get("enabled", False)),
            group_by=(str(d["group_by"]) if d.get("group_by") else None),
            group_labels=labels,
            tissues=[str(t) for t in tissues] if tissues else None,
            kernel_sigma=float(d.get("kernel_sigma", 2.0)),
        )

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "group_by": self.group_by,
            "group_labels": {str(k): v for k, v in self.group_labels.items()},
            "tissues": list(self.tissues) if self.tissues else None,
            "kernel_sigma": self.kernel_sigma,
        }


@dataclass
class AtlasRecipe:
    """Declarative atlas request: ages × condition combinations, in physical units."""

    ages: list[float]
    conditions: dict[str, list[float]] = field(default_factory=dict)
    spacing: list[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    mask_reconstruction: bool = True
    gaussian_span: float = 1.0
    n_max: int = 100
    # The condition whose values ``ages`` sweep (the temporal axis). Defaults to
    # ``scan_age``; set it for datasets whose age condition is named otherwise
    # (e.g. ``GA_MRI``). Must be a registered condition.
    temporal_condition: str = "scan_age"
    growth_curves: GrowthCurvesConfig = field(default_factory=GrowthCurvesConfig)

    @classmethod
    def from_dict(cls, d: dict) -> "AtlasRecipe":
        return cls(
            ages=[float(a) for a in d["ages"]],
            conditions={
                k: [float(v) for v in vs] for k, vs in (d.get("conditions") or {}).items()
            },
            spacing=[float(s) for s in d.get("spacing", [0.5, 0.5, 0.5])],
            mask_reconstruction=bool(d.get("mask_reconstruction", True)),
            gaussian_span=float(d.get("gaussian_span", 1.0)),
            n_max=int(d.get("n_max", 100)),
            temporal_condition=str(d.get("temporal_condition", "scan_age")),
            growth_curves=GrowthCurvesConfig.from_dict(d.get("growth_curves")),
        )

    def to_dict(self) -> dict:
        return {
            "ages": list(self.ages),
            "conditions": {k: list(v) for k, v in self.conditions.items()},
            "spacing": list(self.spacing),
            "mask_reconstruction": self.mask_reconstruction,
            "gaussian_span": self.gaussian_span,
            "n_max": self.n_max,
            "temporal_condition": self.temporal_condition,
            "growth_curves": self.growth_curves.to_dict(),
        }


@dataclass
class AtlasConfig:
    """Atlas generation block in a train config. ``recipe`` may be inline or a path."""

    enabled: bool
    recipe: Optional[AtlasRecipe]

    @classmethod
    def from_dict(cls, d: dict, base_dir: Path) -> "AtlasConfig":
        enabled = bool(d.get("enabled", False))
        rec = d.get("recipe")
        if rec is None:
            return cls(enabled=enabled, recipe=None)
        if isinstance(rec, str):
            rec_path = (base_dir / rec).resolve()
            with open(rec_path) as f:
                rec_dict = yaml.safe_load(f)
        elif isinstance(rec, dict):
            rec_dict = rec
        else:
            raise ValueError(
                f"atlas.recipe must be a mapping (inline) or string (path), got {type(rec).__name__}"
            )
        return cls(enabled=enabled, recipe=AtlasRecipe.from_dict(rec_dict))


@dataclass
class LoggingConfig:
    enabled: bool = False
    wandb_entity: str = ""
    project: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "LoggingConfig":
        return cls(
            enabled=bool(d.get("enabled", False)),
            wandb_entity=str(d.get("wandb_entity", "")),
            project=str(d.get("project", "")),
        )


@dataclass
class DataAugmentationConfig:
    """Pass-through to torchio. Validation deferred to the data module."""

    activate: bool = False
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "DataAugmentationConfig":
        return cls(activate=bool(d.get("activate", False)), raw=dict(d))


@dataclass
class TrainConfig:
    dataset_spec: DatasetSpec
    conditions: ConditionRegistry
    constraints: ConstraintSet
    decoder: DecoderConfig
    optimizer: OptimizerConfig
    training: TrainingConfig
    validation: ValidationConfig
    atlas: AtlasConfig
    logging: LoggingConfig
    augmentation: DataAugmentationConfig
    output_dir: str
    seed: int
    device: str = "cuda"
    amp: bool = True

    def _validate(self) -> None:
        enabled = set(self.conditions.enabled_names())
        if self.atlas.recipe is not None:
            for k in self.atlas.recipe.conditions:
                if k not in enabled:
                    raise ValueError(
                        f"atlas recipe references condition {k!r}, which is not "
                        f"enabled (use_in_decoder=true) in the dataset config"
                    )
        for step in self.validation.latent_analysis:
            if step.target not in self.conditions:
                raise ValueError(
                    f"latent_analysis step targets condition {step.target!r}, "
                    f"which is not registered"
                )
        if self.dataset_spec.has_segmentation:
            n_seg_classes = self.decoder.out_dim[-1]
            n_labels = len(self.dataset_spec.label_names)
            if n_seg_classes != n_labels:
                raise ValueError(
                    f"decoder.out_dim[-1]={n_seg_classes} segmentation channels but "
                    f"dataset has {n_labels} label_names"
                )
            n_intensity_dims = sum(self.decoder.out_dim[:-1])
            n_intensity_mods = len(self.dataset_spec.intensity_modalities)
            if n_intensity_dims != n_intensity_mods:
                raise ValueError(
                    f"decoder.out_dim[:-1]={self.decoder.out_dim[:-1]} sums to "
                    f"{n_intensity_dims} intensity channels but dataset has "
                    f"{n_intensity_mods} intensity modalities"
                )


# === Loaders ===


def _read_yaml(path: Union[str, Path]) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _registry_from_yaml(d: dict) -> ConditionRegistry:
    specs = []
    for name, body in d.items():
        if "name" in body and body["name"] != name:
            raise ValueError(
                f"condition body declares name={body['name']!r} but yaml key is {name!r}"
            )
        body_with_name = {"name": name, **{k: v for k, v in body.items() if k != "name"}}
        specs.append(ConditionSpec.from_dict(body_with_name))
    return ConditionRegistry(specs)


def _resolve_relative(base: Path, p: str) -> str:
    pp = Path(p)
    return str((base / pp).resolve()) if not pp.is_absolute() else str(pp)


def load_dataset_config(
    path: Union[str, Path],
) -> tuple[DatasetSpec, ConditionRegistry, ConstraintSet]:
    """Load a dataset YAML and return its three independent pieces.

    Relative paths in ``tsv_file`` and ``subject_ids`` are resolved against the
    YAML file's directory.
    """
    path = Path(path).resolve()
    base = path.parent
    d = _read_yaml(path)
    dataset = DatasetSpec.from_dict(d["dataset"])
    dataset.tsv_file = _resolve_relative(base, dataset.tsv_file)
    dataset.subject_ids = _resolve_relative(base, dataset.subject_ids)
    conditions = _registry_from_yaml(d.get("conditions") or {})
    constraints = ConstraintSet.from_dict(d.get("constraints") or {})
    return dataset, conditions, constraints


def load_split_subject_ids(spec: DatasetSpec, split: str) -> list[str]:
    """Return the explicit, disjoint subject-ID list for ``split``.

    Splits are defined *only* through the dataset's ``subject_ids`` YAML, keyed
    by dataset ``name`` then ``subject_ids`` then split::

        <dataset_name>:
          subject_ids:
            train: [...]
            val:   [...]
            test:  [...]

    These lists are disjoint by construction and are the only mechanism that
    holds the splits apart: without them the pipeline samples every split from
    the full TSV with the same seed and they overlap (train/val/test leakage).

    The file is therefore **mandatory**. Every way of failing to resolve a
    concrete, non-empty list raises — there is no fallback to random sampling
    over the whole TSV, since that is exactly the contamination this guards
    against.
    """
    path_str = spec.subject_ids
    if not path_str:
        raise ValueError(
            f"dataset {spec.name!r} declares no subject_ids file. A per-split "
            f"subject_ids YAML is required to keep train/val/test disjoint; set "
            f"dataset.subject_ids in the dataset YAML."
        )
    path = Path(path_str)
    if not path.is_file():
        raise FileNotFoundError(
            f"subject_ids file for dataset {spec.name!r} not found: {path}"
        )
    doc = _read_yaml(path) or {}
    entry = doc.get(spec.name)
    if not entry:
        raise ValueError(
            f"subject_ids file {path} has no entry for dataset {spec.name!r} "
            f"(have: {sorted(doc)})."
        )
    sids = entry.get("subject_ids") or {}
    if split not in sids:
        raise ValueError(
            f"subject_ids file {path} lists dataset {spec.name!r} but has no "
            f"{split!r} split (have: {sorted(sids)})."
        )
    ids = [str(x) for x in (sids[split] or [])]
    if not ids:
        raise ValueError(
            f"subject_ids file {path}: {split!r} split for dataset "
            f"{spec.name!r} is empty."
        )
    return ids


def build_train_config(d: dict, base_dir: Path) -> TrainConfig:
    """Build a TrainConfig from an already-loaded dict (for tests / overrides)."""
    base_dir = Path(base_dir)
    ds_path = (base_dir / d["dataset_config"]).resolve()
    dataset_spec, conditions, constraints = load_dataset_config(ds_path)
    cfg = TrainConfig(
        dataset_spec=dataset_spec,
        conditions=conditions,
        constraints=constraints,
        decoder=DecoderConfig.from_dict(d["decoder"]),
        optimizer=OptimizerConfig.from_dict(d["optimizer"]),
        training=TrainingConfig.from_dict(d["training"]),
        validation=ValidationConfig.from_dict(d.get("validation") or {}),
        atlas=AtlasConfig.from_dict(d.get("atlas") or {}, base_dir),
        logging=LoggingConfig.from_dict(d.get("logging") or {}),
        augmentation=DataAugmentationConfig.from_dict(d.get("data_augmentation") or {}),
        output_dir=str(d["output_dir"]),
        seed=int(d.get("seed", 42)),
        device=str(d.get("device", "cuda")),
        amp=bool(d.get("amp", True)),
    )
    cfg._validate()
    return cfg


def load_train_config(
    path: Union[str, Path],
    overrides: Optional[list[str]] = None,
) -> TrainConfig:
    """Load a train YAML, apply ``--set`` overrides, then validate."""
    path = Path(path).resolve()
    base = path.parent
    d = _read_yaml(path)
    if overrides:
        apply_overrides(d, overrides)
    return build_train_config(d, base)


def load_atlas_recipe(path: Union[str, Path]) -> AtlasRecipe:
    return AtlasRecipe.from_dict(_read_yaml(path))


# === CLI override application ===


def apply_overrides(d: dict, overrides: list[str]) -> dict:
    """Apply ``--set key.path=value`` overrides to ``d`` in place.

    Values are parsed via ``yaml.safe_load`` so types come out right:
    ``--set training.epochs=50`` -> int, ``--set atlas.enabled=true`` -> bool,
    ``--set atlas.recipe.ages=[30,32]`` -> list. Unknown keys raise.
    """
    for s in overrides:
        if "=" not in s:
            raise ValueError(f"override {s!r} must be of form key.path=value")
        k, v = s.split("=", 1)
        path_parts = k.split(".")
        value = yaml.safe_load(v)
        cur: Any = d
        for p in path_parts[:-1]:
            if not isinstance(cur, dict) or p not in cur:
                raise KeyError(
                    f"override path {k!r} does not exist in config (at {p!r})"
                )
            cur = cur[p]
        last = path_parts[-1]
        if not isinstance(cur, dict) or last not in cur:
            raise KeyError(
                f"override path {k!r} does not exist in config (at {last!r})"
            )
        cur[last] = value
    return d
