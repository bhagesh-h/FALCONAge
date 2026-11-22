"""Agreement between clocks and intraclass correlation."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from ..core.errors import AnalysisError


# ---------------------------------------------------------------------------
# reliability
# ---------------------------------------------------------------------------
def icc(values: pd.DataFrame, subject_col: str, value_col: str) -> float:
    """ICC(2,1): two-way random effects, absolute agreement, single measure.

    The variant matters and papers rarely say which they used. ICC(2,1) is the
    one that answers "would a repeat measurement of this person give the same
    number", which is the question a clock's technical reliability is about.
    ICC(3,1) assumes the raters are the only ones of interest and reports a
    higher number for the same data.
    """
    g = values.groupby(subject_col)[value_col]
    k = g.count()
    if (k < 2).all():
        raise AnalysisError("ICC needs at least two measurements of some subject")
    n = len(k)
    kbar = float(k.mean())
    grand = float(values[value_col].mean())

    ms_between = float((k * (g.mean() - grand) ** 2).sum() / (n - 1))
    within = sum(float(((v - v.mean()) ** 2).sum()) for _, v in g)
    dfw = float(values.shape[0] - n)
    ms_within = within / dfw if dfw > 0 else np.nan
    denom = ms_between + (kbar - 1) * ms_within
    return float((ms_between - ms_within) / denom) if denom > 0 else np.nan


def pool_icc(values: Sequence[float], weights: Sequence[float] | None = None) -> float:
    """Pool ICCs across studies through Fisher's z.

    Averaging correlations directly under-weights the high ones; the z
    transform is what makes the pooled value comparable to its inputs.
    """
    v = np.clip(np.asarray(values, float), -0.999999, 0.999999)
    w = np.ones_like(v) if weights is None else np.asarray(weights, float)
    ok = np.isfinite(v)
    if not ok.any():
        return np.nan
    z = np.arctanh(v[ok])
    return float(np.tanh(np.average(z, weights=w[ok])))


# ---------------------------------------------------------------------------
# agreement between clocks
# ---------------------------------------------------------------------------
def agreement(result, method: str = "spearman") -> pd.DataFrame:
    """Between-clock correlation.

    Spearman by default. Two clocks on different scales -- years and a
    log-hazard -- have no meaningful Pearson correlation but a perfectly
    meaningful rank one, and mixing scales in a correlation matrix is the normal
    case rather than the exception.
    """
    return result.scores.corr(method=method)
