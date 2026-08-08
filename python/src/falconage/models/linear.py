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
    # Fraction of the model's total |coefficient| carried by present features.
    # None when align() was called without coefficients -- there is no honest
    # default, and 1.0 would read as "all the weight is here".
    mass_coverage: float | None = None
    # The heaviest absent features, worst first, as (feature, |coef| share).
    # What a user needs to decide whether the gap matters.
    missing_mass: list[tuple[str, float]] = field(default_factory=list)


def align(data, features: list[str], *, imputation: str = "reference",
          reference: dict[str, float] | None = None,
          coefficients: np.ndarray | None = None) -> Alignment:
    """Build the clock's feature matrix from whatever the dataset has.

    Parameters
    ----------
    imputation
        ``"reference"`` fills an absent feature with the value the clock's
        authors published for it, or the cohort's own column mean when no
        published value exists. ``"mean"`` always uses the column mean.
        ``"none"`` refuses, leaving NaN, so the coverage check downstream fails
        loudly instead of returning a number.
    coefficients
        The clock's weights, in ``features`` order. Supplying them adds
        ``mass_coverage`` to the result. Optional because alignment is also
        used where there are no weights -- the QC path, and the coverage report
        the registry builds before any model is constructed.

    Notes
    -----
    Zero is never used as a fill value, and this is not a stylistic preference.
    In beta space zero means *completely unmethylated*, which is a real and
    extreme measurement; a clock with a large positive coefficient on a probe
    the array does not carry will read "totally unmethylated" and shift the
    prediction by whole years. Half the disagreements between published
    reimplementations of the same clock come from exactly this.

    WHY COUNTING FEATURES IS NOT ENOUGH. A plain present/total ratio treats
    every probe as interchangeable, and an elastic-net clock's weights are not
    remotely uniform -- a handful of CpGs routinely carry a large share of the
    total. So 92% feature coverage can mean "eight percent of probes missing,
    all of them negligible" or "eight percent missing, and they carry a third
    of the model". Those two datasets produce very different numbers and were
    indistinguishable here until ``mass_coverage`` existed. This is the
    mechanism behind EPICv2 probe loss disrupting the traditional clocks while
    barely moving the PC ones (Life Science Alliance 2025;8:e202403155).
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
        # Read off the mask, not from a second X.reindex(). The frame has
        # already been reindexed once above and that pass is the dominant cost
        # of a scoring run -- doing it again to recover a count we are holding
        # was worth 1.17-1.25x across 1k-16k samples when it was removed.
        per_sample = mask.sum(axis=1)
        out = np.where(mask, np.broadcast_to(fill, out.shape), out)

    if imputation == "none":
        per_sample = np.isnan(out).sum(axis=1)

    mass_coverage, missing_mass = None, []
    if coefficients is not None:
        w = np.abs(np.asarray(coefficients, dtype=np.float64))
        total = float(w.sum())
        # An all-zero coefficient vector is not a real clock, but it is a
        # legitimate degenerate case in tests; dividing by it is not.
        if total > 0:
            mass_coverage = float(w[present].sum()) / total
            absent = np.flatnonzero(~present)
            order = absent[np.argsort(-w[absent])]
            missing_mass = [(features[j], float(w[j]) / total)
                            for j in order[:10] if w[j] > 0]

    return Alignment(matrix=out, present=present, coverage=coverage,
                     n_imputed=n_imputed, imputation=imputation,
                     per_sample_missing=per_sample, notes=notes,
                     mass_coverage=mass_coverage, missing_mass=missing_mass)


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
        al = align(data, self.features, imputation=imputation,
                   coefficients=self.coefficients)

        where = (f"  The dataset is {data.platform or 'an unknown platform'} and this "
                 f"clock was trained on {', '.join(self.clock.platform) or 'unknown'}.\n"
                 "  Lower min_coverage to score anyway, and read the result as an "
                 "extrapolation rather than a measurement.")

        if al.coverage < min_coverage:
            raise FeatureCoverageError(
                f"{self.clock.id}: {al.coverage:.1%} of its {len(self.features)} "
                f"features are present, below the {min_coverage:.0%} floor.\n" + where
            )

        # The same floor, applied to the weights rather than the count. A clock
        # can clear the feature floor comfortably and still have lost the probes
        # that do most of the work, which is the failure this catches.
        if al.mass_coverage is not None and al.mass_coverage < min_coverage:
            worst = ", ".join(f"{f} ({s:.1%})" for f, s in al.missing_mass[:3])
            raise FeatureCoverageError(
                f"{self.clock.id}: {al.coverage:.1%} of features are present, "
                f"but they carry only {al.mass_coverage:.1%} of the model's "
                f"total |coefficient| -- below the {min_coverage:.0%} floor.\n"
                f"  Heaviest absent features: {worst}.\n"
                "  Feature count alone would have passed this dataset. The "
                "probes that are missing are the ones the clock leans on.\n" + where
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
