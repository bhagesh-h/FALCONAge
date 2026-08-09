"""Clocks that aggregate a probe set rather than weight it.

Five entries in the registry are not linear models at all. epiTOC1 and
EPICmitHyper take the **mean** beta over a designated set of CpGs; stemTOC takes
the **95th percentile**; ReedBMI takes a **weighted mean**. There is no
intercept and no fitted slope -- the model *is* the probe list plus a summary
statistic, and until v1.1 all five fell through to :class:`LinearClock`, which
asked for coefficients that do not exist and refused.

WHAT A "COEFFICIENT FILE" MEANS FOR ONE OF THESE. The probe list. For the mean
and percentile forms every weight is 1.0 and only the identifiers matter; for
the weighted form the weights are the weights. That is deliberately the same
two-column CSV :meth:`ClockRegistry.register_local_weights` already validates,
so obtaining a published probe set and scoring with it needs no new plumbing:

    fa.registry.register_local_weights("epitoc1", "epitoc1_probes.csv")

WHY THE STATISTIC IS READ FROM ``model_type``. It is already declared there, in
words, for all five: "mean methylation aggregation", "95th-percentile
methylation aggregation", "weighted methylation aggregation". A second field
saying the same thing in a different vocabulary would be a second place for it
to be wrong, and the registry's own convention is that a clock's architecture is
described once.

WHY THE PERCENTILE MATTERS. stemTOC's 95th percentile is not a robustness
choice. Its premise is that the *most* hypermethylated probes in a sample track
the stem-cell division history, and a mean over the same set would average that
signal away against the bulk. Substituting one statistic for the other returns a
number on the same scale that measures something else.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..core.backend import DeviceSpec
from ..core.errors import FeatureCoverageError, ScoringError
from ..registry.registry import Clock
from . import ops
from .linear import Alignment, align

__all__ = ["AggregationClock", "is_aggregation", "parse_statistic"]


def is_aggregation(clock: Clock) -> bool:
    return "aggregation" in (clock.model_type or "").lower()


def parse_statistic(model_type: str) -> tuple[str, float | None]:
    """``model_type`` in words to a statistic and its parameter.

    Returns ``("mean", None)``, ``("quantile", 0.95)`` or
    ``("weighted_mean", None)``. Raises on anything else rather than falling
    back to the mean: a percentile clock scored as a mean returns a plausible
    number on the right scale that measures a different thing, which is the
    failure mode this package refuses everywhere else.
    """
    s = (model_type or "").lower()
    m = re.search(r"(\d+)(?:st|nd|rd|th)?[- ]percentile", s)
    if m:
        return "quantile", int(m.group(1)) / 100.0
    if "weighted" in s:
        return "weighted_mean", None
    if "mean" in s:
        return "mean", None
    raise ScoringError(
        f"cannot read an aggregation statistic out of model_type {model_type!r}.\n"
        "  Recognised: 'mean methylation aggregation', "
        "'<n>th-percentile methylation aggregation', "
        "'weighted methylation aggregation'.")


@dataclass
class AggregationClock:
    """``postprocess(statistic(X[features]))``.

    No dot product and no intercept. ``coefficients`` carries the weights for
    the weighted form and is ignored -- deliberately, and asserted in the tests
    -- by the mean and percentile forms, where the published definition is over
    the probe set and a weight column in the file would be a fact nobody put
    there.
    """

    clock: Clock
    features: list[str]
    coefficients: np.ndarray
    statistic: str
    q: float | None = None

    def predict(self, data, spec: DeviceSpec, *, imputation: str = "reference",
                min_coverage: float = 0.8) -> tuple[pd.Series, Alignment]:
        al = align(data, self.features, imputation=imputation,
                   coefficients=self.coefficients)
        if al.coverage < min_coverage:
            raise FeatureCoverageError(
                f"{self.clock.id}: {al.coverage:.1%} of its {len(self.features)} "
                f"probes are present, below the {min_coverage:.0%} floor.\n"
                "  An aggregate over a fraction of its probe set is an aggregate "
                "over a different probe set.")

        x = al.matrix                                   # samples x features
        if self.statistic == "mean":
            raw = np.nanmean(x, axis=1)
        elif self.statistic == "quantile":
            # Per sample across probes, not across samples. The premise of a
            # percentile clock is about the spread within one methylome.
            raw = np.nanquantile(x, self.q, axis=1)
        elif self.statistic == "weighted_mean":
            w = np.abs(np.asarray(self.coefficients, dtype=np.float64))
            tot = float(w.sum())
            if tot <= 0:
                raise ScoringError(
                    f"{self.clock.id}: a weighted aggregation needs non-zero "
                    "weights, and the registered file is all zeros")
            raw = (x * w[None, :]).sum(axis=1) / tot
        else:  # pragma: no cover - parse_statistic is the only producer
            raise ScoringError(f"unknown aggregation statistic {self.statistic!r}")

        out = ops.apply_chain(np.asarray(raw, dtype=np.float64),
                              self.clock.postprocess, ops.POSTPROCESS)
        values = np.asarray(out, dtype=np.float64).ravel()
        return pd.Series(values, index=data.sample_ids, name=self.clock.id), al

    @classmethod
    def from_registry(cls, registry, clock_id: str) -> AggregationClock:
        c = registry.get(clock_id)
        feats, coefs = registry.coefficients(clock_id)
        stat, q = parse_statistic(c.model_type)
        return cls(clock=c, features=list(feats), coefficients=coefs,
                   statistic=stat, q=q)
