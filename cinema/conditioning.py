"""
Condition specifications and normalization strategies for CINeMA.

A ``ConditionSpec`` describes one column of the training TSV that may be fed to
the INR decoder as part of the conditioning vector. The spec owns its own
normalization strategy and per-condition ``cond_scale``, so inference code only
needs the spec (serialised in the checkpoint) to convert physical-unit user
input (e.g. ``lv_volume = 15 ml``) into the value the model expects.

Three normalization strategies are provided:

- ``MinMaxNormalization``: globally rescale ``[min, max]`` -> ``[-1, 1]``.
- ``IdentityNormalization``: pass-through (value already in ``[-1, 1]``).
- ``AgeRelativeNormalization``: at fit-time, kernel-regress per-age mean/std of
  the condition from the training data; at normalize-time map to a z-score and
  squash via ``tanh(z / clip_sigmas)``. Resolves the LV-volume problem where the
  meaningful range varies with age.

A ``ConditionRegistry`` is the ordered collection used by data loading, the
trainer, the atlas generator and the CLI.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


def kernel_regress(ages, vals, kernel_sigma, n_grid: int = 200):
    """Gaussian-kernel regression of ``mu(x)`` and ``sigma(x)`` over an x grid.

    Returns ``(grid, mu, sd)`` with ``grid`` spanning ``[min, max]`` of ``ages``.
    Inputs must be finite and length >= 1; callers handle their own NaN-masking
    and minimum-count checks. This is the shared implementation behind
    ``AgeRelativeNormalization.fit`` and the growth-curve module.
    """
    ages = np.asarray(ages, dtype=np.float64)
    vals = np.asarray(vals, dtype=np.float64)
    grid = np.linspace(ages.min(), ages.max(), n_grid)
    diffs = (ages[None, :] - grid[:, None]) / float(kernel_sigma)
    w = np.exp(-0.5 * diffs * diffs)
    w = w / np.clip(w.sum(axis=1, keepdims=True), 1e-12, None)
    mu = (w * vals[None, :]).sum(axis=1)
    var = (w * (vals[None, :] - mu[:, None]) ** 2).sum(axis=1)
    sd = np.sqrt(var + 1e-8)
    return grid, mu, sd


_NORM_TYPES: dict[str, type["Normalization"]] = {}


def _register_norm(type_name: str):
    def deco(cls):
        _NORM_TYPES[type_name] = cls
        cls._type_name = type_name
        return cls
    return deco


class Normalization(ABC):
    """Map physical-unit values to roughly ``[-1, 1]`` and back.

    The ``ConditionSpec`` wrapping the strategy applies ``cond_scale`` on top,
    so subclasses should *not* multiply by ``cond_scale`` themselves.
    """

    _type_name: str

    @abstractmethod
    def fit(self, column_name: str, df: pd.DataFrame) -> None: ...

    @abstractmethod
    def normalize(self, value, context: Optional[dict] = None): ...

    @abstractmethod
    def denormalize(self, normed, context: Optional[dict] = None): ...

    @abstractmethod
    def _payload(self) -> dict: ...

    def to_dict(self) -> dict:
        return {"type": self._type_name, **self._payload()}

    @classmethod
    def from_dict(cls, d: dict) -> "Normalization":
        t = d.get("type")
        if t not in _NORM_TYPES:
            raise ValueError(
                f"unknown normalization type: {t!r}. Known: {sorted(_NORM_TYPES)}"
            )
        payload = {k: v for k, v in d.items() if k != "type"}
        return _NORM_TYPES[t]._from_payload(payload)

    @classmethod
    @abstractmethod
    def _from_payload(cls, d: dict) -> "Normalization": ...


@_register_norm("minmax")
@dataclass(eq=False)
class MinMaxNormalization(Normalization):
    min: float
    max: float

    def fit(self, column_name, df):
        return  # bounds are user-specified

    def normalize(self, value, context=None):
        return 2.0 * ((value - self.min) / (self.max - self.min) - 0.5)

    def denormalize(self, normed, context=None):
        return (normed / 2.0 + 0.5) * (self.max - self.min) + self.min

    def _payload(self):
        return {"min": float(self.min), "max": float(self.max)}

    @classmethod
    def _from_payload(cls, d):
        return cls(min=float(d["min"]), max=float(d["max"]))


@_register_norm("identity")
@dataclass(eq=False)
class IdentityNormalization(Normalization):
    def fit(self, column_name, df):
        return

    def normalize(self, value, context=None):
        return value

    def denormalize(self, normed, context=None):
        return normed

    def _payload(self):
        return {}

    @classmethod
    def _from_payload(cls, d):
        return cls()


@_register_norm("age_relative")
@dataclass(eq=False)
class AgeRelativeNormalization(Normalization):
    """
    Normalize a condition relative to its expected distribution at the
    subject's age. At ``fit()`` time, kernel-regresses ``mu(age)`` and
    ``sigma(age)`` over a 200-point grid spanning the observed age range using
    a Gaussian kernel of width ``kernel_sigma_weeks``. At ``normalize()`` time,
    interpolates the grid, computes ``z = (value - mu) / sigma`` and squashes
    through ``tanh(z / clip_sigmas)``.

    The age column is named by ``age_key`` and must be a column of the training
    df. ``context`` passed to ``normalize/denormalize`` must contain the same
    key.
    """

    age_key: str
    kernel_sigma_weeks: float = 2.0
    clip_sigmas: float = 3.0
    _age_grid: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    _mu_grid: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    _sigma_grid: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    def fit(self, column_name, df):
        if self.age_key not in df.columns:
            raise ValueError(
                f"age_relative normalization references age_key={self.age_key!r}, "
                f"which is not a column of the training dataframe"
            )
        ages = df[self.age_key].to_numpy(dtype=np.float64)
        vals = df[column_name].to_numpy(dtype=np.float64)
        mask = ~(np.isnan(ages) | np.isnan(vals))
        ages, vals = ages[mask], vals[mask]
        if ages.size < 2:
            raise ValueError(
                f"age_relative normalization needs >=2 non-NaN training rows with "
                f"both '{self.age_key}' and '{column_name}'; got {ages.size}."
            )
        grid, mu, sd = kernel_regress(ages, vals, self.kernel_sigma_weeks)
        self._age_grid = grid
        self._mu_grid = mu
        self._sigma_grid = sd

    def _lookup(self, age):
        if self._age_grid is None:
            raise RuntimeError(
                f"AgeRelativeNormalization not fitted. "
                f"Call ConditionRegistry.fit(training_df) first."
            )
        mu = np.interp(age, self._age_grid, self._mu_grid)
        sigma = np.interp(age, self._age_grid, self._sigma_grid)
        return mu, sigma

    def _require_age(self, context):
        if context is None or self.age_key not in context:
            raise ValueError(
                f"age_relative normalization requires context[{self.age_key!r}] "
                f"to be supplied"
            )
        return context[self.age_key]

    def normalize(self, value, context=None):
        age = self._require_age(context)
        mu, sigma = self._lookup(age)
        z = (value - mu) / sigma
        return np.tanh(z / self.clip_sigmas)

    def denormalize(self, normed, context=None):
        age = self._require_age(context)
        mu, sigma = self._lookup(age)
        v = np.clip(normed, -1.0 + 1e-6, 1.0 - 1e-6)
        z = np.arctanh(v) * self.clip_sigmas
        return z * sigma + mu

    def from_z(self, z):
        """Map a z-score (in sigma units) directly to the ``[-1, 1]`` code.

        This is the age-invariant counterpart to ``normalize``: a given z maps
        to a fixed code regardless of age (the ``mu(age)``/``sigma(age)`` lookup
        cancels), which makes the z-score the natural knob for atlas generation.
        Requires no age context.
        """
        return np.tanh(np.asarray(z, dtype=np.float64) / self.clip_sigmas)

    def _payload(self):
        d: dict = {
            "age_key": self.age_key,
            "kernel_sigma_weeks": float(self.kernel_sigma_weeks),
            "clip_sigmas": float(self.clip_sigmas),
        }
        if self._age_grid is not None:
            d["fitted"] = {
                "age_grid": self._age_grid.tolist(),
                "mu_grid": self._mu_grid.tolist(),
                "sigma_grid": self._sigma_grid.tolist(),
            }
        return d

    @classmethod
    def _from_payload(cls, d):
        obj = cls(
            age_key=d["age_key"],
            kernel_sigma_weeks=float(d.get("kernel_sigma_weeks", 2.0)),
            clip_sigmas=float(d.get("clip_sigmas", 3.0)),
        )
        f = d.get("fitted")
        if f is not None:
            obj._age_grid = np.asarray(f["age_grid"], dtype=np.float64)
            obj._mu_grid = np.asarray(f["mu_grid"], dtype=np.float64)
            obj._sigma_grid = np.asarray(f["sigma_grid"], dtype=np.float64)
        return obj


@_register_norm("zscore")
@dataclass(eq=False)
class ZScoreNormalization(Normalization):
    """Parameter-free squash of a *precomputed* z-score to ``(-1, 1)``.

    The conditioned column already holds a z-score (sigma units) computed
    offline against a fixed normative reference — see
    ``scripts/compute_lv_zscore.py``, which fits per-cohort ``mu(age)``/
    ``sigma(age)`` on the full cohort and writes both the ``*_z`` column and a
    JSON reference artifact. This strategy just bounds it via
    ``tanh(z / clip_sigmas)``: no fit, no age context needed.

    It is mathematically identical to ``AgeRelativeNormalization.from_z``, but
    moves the reference out of the per-split, per-checkpoint kernel fit and into
    an external, inspectable, reproducible artifact. Prefer this whenever the
    z-score can be precomputed.
    """

    clip_sigmas: float = 3.0

    def fit(self, column_name, df):
        return  # nothing to fit — the z-score is already in the column

    def normalize(self, value, context=None):
        return np.tanh(np.asarray(value, dtype=np.float64) / self.clip_sigmas)

    def denormalize(self, normed, context=None):
        v = np.clip(normed, -1.0 + 1e-6, 1.0 - 1e-6)
        return np.arctanh(v) * self.clip_sigmas

    def _payload(self):
        return {"clip_sigmas": float(self.clip_sigmas)}

    @classmethod
    def _from_payload(cls, d):
        return cls(clip_sigmas=float(d.get("clip_sigmas", 3.0)))


@dataclass
class ConditionSpec:
    """One condition column with its normalization strategy and decoder scale."""

    name: str
    unit: str
    use_in_decoder: bool
    normalization: Normalization
    cond_scale: float = 0.1

    def fit(self, df: pd.DataFrame) -> None:
        if self.name not in df.columns:
            raise ValueError(
                f"condition {self.name!r} not found in training dataframe columns"
            )
        self.normalization.fit(self.name, df)

    def normalize(self, value, context: Optional[dict] = None):
        return self.normalization.normalize(value, context) * self.cond_scale

    def denormalize(self, normed, context: Optional[dict] = None):
        return self.normalization.denormalize(normed / self.cond_scale, context)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "unit": self.unit,
            "use_in_decoder": bool(self.use_in_decoder),
            "cond_scale": float(self.cond_scale),
            "normalization": self.normalization.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ConditionSpec":
        return cls(
            name=d["name"],
            unit=d["unit"],
            use_in_decoder=bool(d["use_in_decoder"]),
            cond_scale=float(d.get("cond_scale", 0.1)),
            normalization=Normalization.from_dict(d["normalization"]),
        )


class ConditionRegistry:
    """Ordered collection of ``ConditionSpec`` with vector-building helpers.

    Order is preserved: the conditioning vector seen by the decoder uses the
    declaration order of ``use_in_decoder=True`` specs. Mixing this order
    between training and inference would silently produce garbage, so the
    registry is serialised verbatim into the checkpoint.
    """

    def __init__(self, specs: list[ConditionSpec]):
        self.specs: list[ConditionSpec] = list(specs)
        self._by_name: dict[str, ConditionSpec] = {s.name: s for s in specs}
        if len(self._by_name) != len(specs):
            raise ValueError("duplicate condition names in registry")
        self._validate_cross_refs()

    def _validate_cross_refs(self) -> None:
        for spec in self.specs:
            if isinstance(spec.normalization, AgeRelativeNormalization):
                age_key = spec.normalization.age_key
                if age_key not in self._by_name:
                    raise ValueError(
                        f"condition {spec.name!r} uses age_relative normalization "
                        f"with age_key={age_key!r}, which is not a registered condition"
                    )

    def __getitem__(self, name: str) -> ConditionSpec:
        return self._by_name[name]

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def __iter__(self):
        return iter(self.specs)

    def __len__(self):
        return len(self.specs)

    def enabled_specs(self) -> list[ConditionSpec]:
        return [s for s in self.specs if s.use_in_decoder]

    def enabled_names(self) -> list[str]:
        return [s.name for s in self.specs if s.use_in_decoder]

    def fit(self, training_df: pd.DataFrame) -> None:
        for spec in self.specs:
            spec.fit(training_df)

    def vector_for_row(self, row: dict) -> np.ndarray:
        """Build the decoder condition vector for a training/val dataframe row.

        ``row`` itself is used as context for age-relative normalization.
        """
        enabled = self.enabled_specs()
        out = np.empty(len(enabled), dtype=np.float32)
        for i, spec in enumerate(enabled):
            out[i] = spec.normalize(row[spec.name], context=row)
        return out

    def vector_from_user(
        self,
        values: dict,
        context: Optional[dict] = None,
    ) -> np.ndarray:
        """Build a condition vector from user-supplied physical-unit values.

        ``context`` supplies dependencies that are not condition values
        themselves (e.g. ``scan_age`` for age-relative specs when the user did
        not list ``scan_age`` as one of the condition values).
        """
        enabled = self.enabled_specs()
        ctx = dict(values) if context is None else {**context, **values}
        out = np.empty(len(enabled), dtype=np.float32)
        for i, spec in enumerate(enabled):
            if spec.name not in values:
                raise ValueError(
                    f"missing value for enabled condition {spec.name!r}"
                )
            out[i] = spec.normalize(values[spec.name], context=ctx)
        return out

    def to_dict(self) -> dict:
        return {"specs": [s.to_dict() for s in self.specs]}

    @classmethod
    def from_dict(cls, d: dict) -> "ConditionRegistry":
        return cls([ConditionSpec.from_dict(s) for s in d["specs"]])
