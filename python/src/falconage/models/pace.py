"""DunedinPACE, scored as its authors' ``PACEProjector`` scores it.

WHY A MODEL CLASS OF ITS OWN. DunedinPACE is 173 weights applied after every
sample has been quantile-normalised to a 20,000-probe reference distribution
(Belsky et al. 2022, eLife 11:e73420). The normalisation is what makes the
weights mean anything, and it needs the whole background set, not the 173.
The coefficients are research-use only and are not distributed; a user who has
installed the authors' R package registers them with
``fa.registry.load().import_dunedinpace(path)``.

WHAT IS REPRODUCED, STEP BY STEP, from ``PACEProjector.R``
(danbelsky/DunedinPACE at 4b56998):

1. The model and the background set must each be at least
   ``proportionOfProbesRequired`` present by name: 0.8, or 0.7 on EPIC v2,
   where the authors lower it because v2 lacks some 450K and EPIC v1 probes.
2. Background probes the data lacks are added at their gold-standard mean.
3. A sample missing more than ``1 - proportion`` of the background is dropped
   and scored NA.
4. A probe missing some values but present in at least ``proportion`` of the
   samples takes its cohort mean; one present in fewer takes the Dunedin model
   mean for every sample, which exists only for the 173 model probes, so a
   background probe in that state stays NA, as it does in R.
5. Each sample is quantile-normalised to the gold-standard means with
   ``preprocessCore::normalize.quantiles.use.target``, reproduced below in
   both of its branches, tie handling included.
6. The score is the intercept plus the weighted sum over the model probes.

EPIC v2 probe suffixes are already collapsed by :func:`falconage.prepare`, by
averaging, which is what ``PACEProjector`` does itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from ..core.backend import DeviceSpec
from ..core.errors import FeatureCoverageError
from ..registry.registry import Clock
from .linear import Alignment, align_present

_EPS = np.finfo(np.float64).eps


def quantile_normalize_to_target(M: np.ndarray, target) -> np.ndarray:
    """``preprocessCore::normalize.quantiles.use.target(M, target)``, columns as samples.

    Ranks are averaged over ties, as R ranks them. A column with no missing
    value, as long as the target, maps rank r to the r-th smallest target
    value, and a tie whose average rank has a fractional part above 0.4 to
    the mean of the two neighbours. A column with missing values, or a target
    of another length, maps each value by linear interpolation at its sample
    percentile; missing values stay missing.
    """
    t = np.sort(np.asarray(target, dtype=np.float64))
    t = t[~np.isnan(t)]
    T = len(t)
    M = np.asarray(M, dtype=np.float64)
    out = np.full_like(M, np.nan)
    rows = M.shape[0]
    for j in range(M.shape[1]):
        col = M[:, j]
        ok = ~np.isnan(col)
        n = int(ok.sum())
        if n == 0:
            continue
        r = rankdata(col[ok], method="average")
        if rows == T and n == rows:
            fl = np.floor(r).astype(int)
            half = (r - fl) > 0.4
            v = t[fl - 1].copy()
            v[half] = 0.5 * (t[fl[half] - 1] + t[fl[half]])
        else:
            p = (r - 1.0) / (n - 1.0) if n > 1 else np.zeros_like(r)
            tid = 1.0 + (T - 1.0) * p
            fl = np.floor(tid + 4 * _EPS)
            frac = tid - fl
            frac[np.abs(frac) <= 4 * _EPS] = 0.0
            v = np.empty_like(r)
            exact = frac == 0.0
            v[exact] = t[np.floor(fl[exact] + 0.5).astype(int) - 1]
            one = frac == 1.0
            v[one] = t[np.floor(fl[one] + 1.5).astype(int) - 1]
            rest = ~exact & ~one
            ti = np.floor(fl[rest] + 0.5).astype(int)
            f = frac[rest]
            inner = (ti < T) & (ti > 0)
            vr = np.where(ti >= T, t[T - 1], t[0])
            vr[inner] = (1.0 - f[inner]) * t[ti[inner] - 1] + f[inner] * t[ti[inner]]
            v[rest] = vr
        out[ok, j] = v
    return out


@dataclass
class PaceModel:
    """The pieces of ``mPACE_Models`` the projector reads, for one model."""

    probes: list[str]
    weights: np.ndarray
    intercept: float
    gold_means: pd.Series          # indexed by background probe
    model_means: pd.Series         # indexed by model probe


def project(X: pd.DataFrame, model: PaceModel, proportion: float) -> pd.Series:
    """Scores for ``X`` (samples x probes), NaN where the authors' code gives NA.

    Raises when the data carry too little of the model or the background,
    where the authors' code returns NA for everyone: a column of NA with no
    reason is the failure this package is built to avoid.
    """
    gold = list(model.gold_means.index)
    have = set(map(str, X.columns))
    overlap = sum(p in have for p in model.probes) / len(model.probes)
    background = sum(p in have for p in gold) / len(gold)
    if overlap < proportion or background < proportion:
        raise FeatureCoverageError(
            f"dunedinpace: {overlap:.1%} of the model probes and {background:.1%} of "
            f"the {len(gold):,} background probes are present; the authors require "
            f"{proportion:.0%} of each")

    # Samples x background probes, absent probes at their gold-standard mean.
    B = X.reindex(columns=gold).to_numpy(dtype=np.float64, copy=True)
    absent = np.array([p not in have for p in gold])
    B[:, absent] = model.gold_means.to_numpy()[absent]

    keep = (1.0 - np.isnan(B).mean(axis=1)) >= proportion
    scores = pd.Series(np.nan, index=X.index, name="dunedinpace")
    if not keep.any():
        return scores
    B = B[keep]

    present = 1.0 - np.isnan(B).mean(axis=0)
    partial = (present < 1.0) & (present >= proportion)
    if partial.any():
        with np.errstate(invalid="ignore"):
            means = np.nanmean(B[:, partial], axis=0)
        sub = B[:, partial]
        B[:, partial] = np.where(np.isnan(sub), means, sub)
    sparse = present < proportion
    if sparse.any():
        B[:, sparse] = model.model_means.reindex(
            [g for g, s in zip(gold, sparse) if s]).to_numpy()

    norm = quantile_normalize_to_target(B.T, model.gold_means.to_numpy()).T
    idx = [gold.index(p) for p in model.probes]
    scores[keep] = model.intercept + norm[:, idx] @ model.weights
    return scores


@dataclass
class PaceClock:
    """DunedinPACE from weights and a background set the user registered."""

    clock: Clock
    model: PaceModel

    def predict(self, data, spec: DeviceSpec, *, imputation: str = "reference",
                min_coverage: float = 0.8) -> tuple[pd.Series, Alignment]:
        # The authors' own threshold, not FALCONAge's floor: 0.7 on EPIC v2.
        proportion = 0.7 if data.platform == "EPICv2" else 0.8
        scores = project(data.X, self.model, proportion)
        al = align_present(data, self.model.probes, coefficients=self.model.weights)
        return scores.rename(self.clock.id), al

    @classmethod
    def from_registry(cls, registry, clock_id: str) -> PaceClock:
        return cls(clock=registry.get(clock_id), model=registry.pace_model(clock_id))


def is_pace(clock: Clock) -> bool:
    return any(s.get("op") == "quantile_normalize_gold" for s in clock.preprocess)
