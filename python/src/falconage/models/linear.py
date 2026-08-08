"""Linear clocks: align features, dot, transform.

Almost every published methylation clock is this shape. The interesting part is
not the dot product -- it is what happens to the features the data does not
carry, which is most of them most of the time, and which is where two
implementations of the same clock stop agreeing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..core.backend import DeviceSpec
from ..core.errors import FeatureCoverageError
from ..registry.registry import Clock
from . import ops


@dataclass
class Alignment:
    """The result of matching a clock's feature list against a dataset."""

    matrix: np.ndarray               # samples x n_clock_features, imputed
    present: np.ndarray              # bool, per clock feature
    coverage: float
    n_imputed: int
    imputation: str
    per_sample_missing: np.ndarray   # count of features imputed per sample
    notes: list[str] = field(default_factory=list)


def align(data, features: list[str], *, imputation: str = "reference",
          reference: dict[str, float] | None = None) -> Alignment:
    """Build the clock's feature matrix from whatever the dataset has.

    Parameters
    ----------
    imputation
        ``"reference"`` fills an absent feature with the value the clock's
        authors published for it, or the cohort's own column mean when no
        published value exists. ``"mean"`` always uses the column mean.
        ``"none"`` refuses, leaving NaN, so the coverage check downstream fails
        loudly instead of returning a number.

    Notes
    -----
    Zero is never used as a fill value, and this is not a stylistic preference.
    In beta space zero means *completely unmethylated*, which is a real and
    extreme measurement; a clock with a large positive coefficient on a probe
    the array does not carry will read "totally unmethylated" and shift the
    prediction by whole years. Half the disagreements between published
    reimplementations of the same clock come from exactly this.
    """
    X = data.X

    # One reindex, not a loop over features. The obvious per-feature
    # `X.iloc[:, i]` costs a pandas indexing call each time: profiling a
    # 4096-sample, eight-clock run showed 2,666 of them consuming 76% of the
    # total, against 0.5% for the arithmetic they fed. That is also why the GPU
    # looked barely worth using -- the device was never the bottleneck.
    #
    # reindex yields an all-NaN column for a feature the data lacks, which is
    # exactly the representation the imputation step below already expects.
    out = X.reindex(columns=features).to_numpy(dtype=np.float64, copy=True)

    # Absent and present-but-all-NaN collapse to the same thing here, and
    # should: a probe column that is NaN for every sample is not a measurement,
    # and counting it as covered is how a clock reports 80% coverage of probes
    # it cannot see. GEO series matrices carry these routinely.
    present = ~np.isnan(out).all(axis=0)

    coverage = float(present.sum()) / max(len(features), 1)
    notes: list[str] = []

    if imputation == "none":
        n_imputed = int(np.isnan(out).sum())
    else:
        # Column means come from the observed part of THIS dataset; the
        # published reference, when there is one, comes from the clock's own
        # training cohort and is the better answer.
        #
        # The suppression is deliberate: an all-NaN column is the normal case
        # here -- it is a feature the array does not carry -- and numpy's "Mean
        # of empty slice" for each of several hundred would bury the coverage
        # warning that actually matters.
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore", RuntimeWarning)
            col_mean = np.nanmean(out, axis=0)
        overall = float(np.nanmean(out)) if np.isfinite(out).any() else 0.5
        fill = np.where(np.isnan(col_mean), overall, col_mean)
        if reference and imputation == "reference":
            for j, f in enumerate(features):
                if f in reference:
                    fill[j] = reference[f]
            notes.append(f"{sum(f in (reference or {}) for f in features)} feature(s) "
                         "filled from the clock's published reference values")
        mask = np.isnan(out)
        n_imputed = int(mask.sum())
        out = np.where(mask, np.broadcast_to(fill, out.shape), out)

    per_sample = np.isnan(X.reindex(columns=features)).to_numpy().sum(axis=1)

    return Alignment(matrix=out, present=present, coverage=coverage,
                     n_imputed=n_imputed, imputation=imputation,
                     per_sample_missing=per_sample, notes=notes)


@dataclass
class LinearClock:
    """``postprocess(preprocess(X)[features] @ coefficients)``."""

    clock: Clock
    features: list[str]
    coefficients: np.ndarray

    def __post_init__(self) -> None:
        if len(self.features) != len(self.coefficients):
            raise ValueError(
                f"{self.clock.id}: {len(self.features)} features but "
                f"{len(self.coefficients)} coefficients")

    def predict(self, data, spec: DeviceSpec, *, imputation: str = "reference",
                min_coverage: float = 0.8) -> tuple[pd.Series, Alignment]:
        al = align(data, self.features, imputation=imputation)

        if al.coverage < min_coverage:
            raise FeatureCoverageError(
                f"{self.clock.id}: {al.coverage:.1%} of its {len(self.features)} "
                f"features are present, below the {min_coverage:.0%} floor.\n"
                f"  The dataset is {data.platform or 'an unknown platform'} and this "
                f"clock was trained on {', '.join(self.clock.platform) or 'unknown'}.\n"
                "  Lower min_coverage to score anyway, and read the result as an "
                "extrapolation rather than a measurement."
            )

        xp = spec.xp()
        x = spec.asarray(al.matrix)
        w = spec.asarray(self.coefficients)

        x = ops.apply_chain(x, self.clock.preprocess, ops.PREPROCESS, xp=xp)
        raw = x @ w
        out = ops.apply_chain(raw, self.clock.postprocess, ops.POSTPROCESS, xp=xp)

        values = np.asarray(spec.tonumpy(out), dtype=np.float64).ravel()
        return pd.Series(values, index=data.sample_ids, name=self.clock.id), al

    @classmethod
    def from_registry(cls, registry, clock_id: str) -> LinearClock:
        c = registry.get(clock_id)
        feats, coefs = registry.coefficients(clock_id)
        return cls(clock=c, features=list(feats), coefficients=coefs)


@dataclass
class ScaffoldClock:
    """A tier C clock: everything except the numbers.

    Instantiating one is fine and is what the shape tests exercise. Calling
    :meth:`predict` without registered coefficients raises, by design -- a
    silent skip would leave a results table with a column quietly missing and
    no indication why.
    """

    clock: Clock
    registry: object

    @property
    def expected_features(self) -> int | None:
        return self.clock.n_features

    def predict(self, data, spec, **kw):
        from ..core.errors import WeightsUnavailableError

        raise WeightsUnavailableError(
            self.clock.id, self.registry.unavailable_message(self.clock.id))


def build(registry, clock_id: str):
    """Return the right model object for a registry entry."""
    c = registry.get(clock_id)
    if c.availability == "C" and not registry.has_coefficients(clock_id):
        return ScaffoldClock(clock=c, registry=registry)
    if c.formula:
        from .clinical import ClinicalClock

        return ClinicalClock(clock=c)
    return LinearClock.from_registry(registry, clock_id)
