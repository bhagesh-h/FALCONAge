"""scAge: an age per cell, from a methylome that is almost entirely missing.

WHY A SINGLE CELL NEEDS A DIFFERENT MODEL. A bulk methylome gives every CpG a
fraction between 0 and 1, averaged over millions of cells. A single-cell
methylome gives a handful of reads per site, so each covered CpG is effectively
**binary** -- methylated or not -- and 95 to 99 percent of sites are not covered
at all. A linear clock over 353 probes has nothing to dot: it would see three of
them, impute 350, and return its intercept with a rounding error on top.

THE METHOD (Trapp, Kerepesi & Gladyshev, Nature Aging 2021;1:1189-1201)

Fit, in a bulk reference cohort, a per-CpG linear model of methylation on age:

    m_i(age) = intercept_i + slope_i * age

For a candidate age, that predicts a methylation *probability* at every CpG.
Given the cell's observed binary calls, the log-likelihood of the candidate is

    sum over covered i of  log( p_i )    if the site is methylated
                           log(1 - p_i)  if it is not

and the cell's age is the candidate maximising it. Profiled over a grid rather
than solved: the likelihood is not concave in general, the grid is one
vectorised evaluation per age, and a grid maximum can report the whole curve --
which is the useful output, because a flat curve means the cell had too few
informative sites and that is worth seeing.

WHICH CPGS. Only those whose bulk association with age is strong enough to say
anything, ranked by ``|r|`` and capped. Including everything drowns the signal
in sites whose slope is noise, which is the failure the original paper's
percentile cut exists to avoid.

WHAT THIS IS NOT. Not a clock in the registry sense: it needs a reference fitted
on a bulk cohort with ages, which is data the user brings, so there is no
coefficient file to ship or checksum. It is a model class plus the fitting step,
and the reference it produces is the artefact worth keeping.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..core.errors import AnalysisError, DataError

__all__ = ["ScAgeReference", "fit_scage_reference", "scage"]

#: Predicted probabilities are clamped inside this. A site whose fitted line
#: leaves [0, 1] at some candidate age would otherwise contribute log(0), and
#: one such site would decide the answer on its own.
CLAMP = 1e-3


@dataclass
class ScAgeReference:
    """Per-CpG linear models of bulk methylation on age.

    ``slope`` and ``intercept`` per site, with the correlation that selected it
    and the age range it was fitted over. The range matters: the model is a
    straight line, and a candidate age outside where it was fitted is
    extrapolation with no data behind it.
    """

    slope: pd.Series
    intercept: pd.Series
    r: pd.Series
    age_min: float
    age_max: float
    n_reference: int

    def probability(self, age: float) -> np.ndarray:
        p = self.intercept.to_numpy() + self.slope.to_numpy() * float(age)
        return np.clip(p, CLAMP, 1.0 - CLAMP)

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (f"ScAgeReference({len(self.slope):,} CpGs, fitted on "
                f"{self.n_reference} samples aged {self.age_min:g}-{self.age_max:g})")


def fit_scage_reference(data, *, age_col: str = "age", min_abs_r: float = 0.2,
                        max_sites: int = 30_000) -> ScAgeReference:
    """Fit the per-CpG age models on a bulk reference cohort.

    Parameters
    ----------
    min_abs_r
        Keep only sites whose absolute correlation with age clears this.
        Including every site drowns the signal in slopes that are noise.
    max_sites
        Cap, applied after ranking by ``|r|``. Thirty thousand informative sites
        is far more than any single cell will cover, and the likelihood is a sum
        over the covered ones only.
    """
    if age_col not in data.obs.columns:
        raise AnalysisError(f"no {age_col!r} column in obs")
    age = pd.to_numeric(data.obs[age_col], errors="coerce").to_numpy(dtype=np.float64)
    ok = np.isfinite(age)
    if ok.sum() < 20:
        raise AnalysisError(
            f"{int(ok.sum())} samples with an age; a per-CpG regression on fewer "
            "than twenty is fitting noise site by site")

    X = data.X.to_numpy(dtype=np.float64)[ok]
    a = age[ok]
    a_c = a - a.mean()
    denom = float((a_c ** 2).sum())

    with np.errstate(invalid="ignore", divide="ignore"):
        mu = np.nanmean(X, axis=0)
        slope = np.nansum((X - mu) * a_c[:, None], axis=0) / denom
        intercept = mu - slope * a.mean()
        sd_x = np.nanstd(X, axis=0, ddof=1)
        r = slope * (np.sqrt(denom / max(len(a) - 1, 1)) / np.where(sd_x == 0, np.nan, sd_x))

    keep = np.isfinite(r) & (np.abs(r) >= min_abs_r) & np.isfinite(slope)
    if keep.sum() == 0:
        raise AnalysisError(
            f"no CpG reaches |r| >= {min_abs_r} against age in this reference. "
            "Either the cohort's age range is too narrow or the threshold is "
            "too high.")
    idx = np.flatnonzero(keep)
    if idx.size > max_sites:
        idx = idx[np.argsort(-np.abs(r[idx]))[:max_sites]]

    cols = data.X.columns[idx]
    return ScAgeReference(
        slope=pd.Series(slope[idx], index=cols),
        intercept=pd.Series(intercept[idx], index=cols),
        r=pd.Series(r[idx], index=cols),
        age_min=float(np.nanmin(a)), age_max=float(np.nanmax(a)),
        n_reference=int(ok.sum()))


def scage(cells, reference: ScAgeReference, *, grid: np.ndarray | None = None,
          min_sites: int = 20, binarise: float = 0.5) -> pd.DataFrame:
    """An age per cell, by profile likelihood over a grid of candidate ages.

    Parameters
    ----------
    cells
        A :class:`~falconage.core.FalconData` of single cells: mostly NaN, with
        covered sites near 0 or 1.
    grid
        Candidate ages. Defaults to a 0.5-year grid over the reference's own
        fitted range -- outside it the linear models are extrapolation.
    min_sites
        A cell covering fewer informative sites than this gets NaN rather than a
        number. Twenty binary observations against a straight line is not an age
        estimate, and returning one anyway is how a per-cell table fills with
        confident noise.

    Returns
    -------
    One row per cell: the maximising age, how many informative sites it had, the
    log-likelihood at the maximum, and the curvature-free width of the peak --
    the range of ages within 2 log-likelihood units, which is what says whether
    the cell had enough sites to distinguish anything.
    """
    shared = [c for c in cells.X.columns if c in reference.slope.index]
    if not shared:
        raise DataError(
            "this reference and these cells share no CpGs.\n"
            "  A single-cell methylome is keyed by coordinate and a bulk array "
            "by probe id; one of the two needs mapping before they can meet.")

    ref_slope = reference.slope[shared]
    ages = (np.arange(reference.age_min, reference.age_max + 1e-9, 0.5)
            if grid is None else np.asarray(grid, dtype=np.float64))
    if ages.size < 2:
        raise DataError("the age grid needs at least two candidates")

    sub = ScAgeReference(slope=ref_slope, intercept=reference.intercept[shared],
                         r=reference.r[shared], age_min=reference.age_min,
                         age_max=reference.age_max, n_reference=reference.n_reference)
    probs = np.stack([sub.probability(a) for a in ages])        # ages x sites

    M = cells.X[shared].to_numpy(dtype=np.float64)
    covered = np.isfinite(M)
    meth = covered & (M >= binarise)

    rows = []
    log_p = np.log(probs)
    log_q = np.log1p(-probs)
    for i in range(M.shape[0]):
        c = covered[i]
        n = int(c.sum())
        if n < min_sites:
            rows.append({"cell": cells.X.index[i], "age": np.nan, "n_sites": n,
                         "loglik": np.nan, "interval_width": np.nan,
                         "reason": f"only {n} informative sites"})
            continue
        m = meth[i][c]
        ll = (log_p[:, c][:, m].sum(axis=1) + log_q[:, c][:, ~m].sum(axis=1))
        k = int(np.argmax(ll))
        within = ages[ll >= ll[k] - 2.0]
        rows.append({"cell": cells.X.index[i], "age": float(ages[k]), "n_sites": n,
                     "loglik": float(ll[k]),
                     "interval_width": float(within.max() - within.min()),
                     "reason": ""})

    out = pd.DataFrame(rows).set_index("cell")
    out.attrs["grid"] = (float(ages.min()), float(ages.max()))
    out.attrs["n_reference_sites"] = len(shared)
    return out
