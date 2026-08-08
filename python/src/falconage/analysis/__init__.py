"""Downstream statistics: acceleration, association, reliability, benchmarking.

The rule that runs through all of it: a clock's ``scale_type`` decides which
operations are defined. Age acceleration is a residual against chronological
age; it means something for a clock that outputs years, nothing for one that
outputs a log-hazard, and something actively misleading for a pace of aging,
which is already a rate. :class:`~falconage.core.errors.IllegalOperationError`
is raised rather than computed, because the alternative is a number that looks
like every other number in the table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from ..core.errors import AnalysisError, IllegalOperationError

__all__ = [
    "BenchmarkResult", "acceleration", "agreement", "icc", "run_benchmark",
    "associate", "cox_hazard",
]


def _check_legal(registry, clock_id: str, op: str) -> None:
    c = registry.get(clock_id)
    if op not in c.legal_operations:
        raise IllegalOperationError(
            f"{op!r} is not defined for {clock_id}, whose output is "
            f"{c.scale_type} ({', '.join(c.unit) or 'no unit'}).\n"
            f"  Legal here: {', '.join(sorted(c.legal_operations))}.\n"
            + ("  A pace of aging is already a rate; subtracting chronological "
               "age from it is a units error, not a conservative choice.\n"
               if c.scale_type == "pace_ratio" else "")
            + ("  A log-hazard has no zero point on the age scale; use "
               "cox_hazard or rank it.\n" if c.scale_type == "mortality_log_hazard" else "")
        )


# ---------------------------------------------------------------------------
# age acceleration
# ---------------------------------------------------------------------------
def acceleration(result, *, age_col: str = "age", method: str = "residual",
                 group: str | None = None, clocks: Sequence[str] | None = None
                 ) -> pd.DataFrame:
    """Age acceleration, in whichever of the three conventions you mean.

    Parameters
    ----------
    method
        ``"absolute"``
            ``predicted - chronological``. Interpretable in years, and
            confounded by the clock's own bias: a clock that over-predicts
            everyone by three years gives everyone three years of acceleration.
        ``"residual"``
            The residual from regressing predicted on chronological age. Centred
            at zero by construction, which removes that bias and also removes
            any real cohort-wide effect. The field's default.
        ``"within_group"``
            Residual from a regression fitted separately within each level of
            ``group``. What the AA2 benchmark needs: it asks whether cases
            accelerate relative to *their own* controls, not relative to a line
            fitted through both.

    Notes
    -----
    Which one a paper used is often not stated, and the three disagree by
    several years on the same data. The convention is recorded in the returned
    frame's ``method`` column so a downstream reader does not have to guess.
    """
    if age_col not in result.obs.columns:
        raise AnalysisError(
            f"no {age_col!r} column in obs; age acceleration needs chronological age.\n"
            f"  obs has: {', '.join(map(str, result.obs.columns)) or '(nothing)'}")

    age = pd.to_numeric(result.obs[age_col], errors="coerce")

    # Naming clocks explicitly means every one must work -- an explicit request
    # is never silently dropped. Not naming any means "the ones this makes sense
    # for", which excludes the pace and log-hazard scales rather than refusing
    # to compute anything because one column in the table is a rate.
    if clocks:
        cols = list(clocks)
        for cid in cols:
            _check_legal(result.registry, cid, "acceleration")
    else:
        cols = [c for c in result.scores.columns
                if "acceleration" in result.registry.get(c).legal_operations]
        if not cols:
            raise IllegalOperationError(
                "no clock in this result has an age scale, so age acceleration is "
                "undefined for all of them.\n  Scales present: "
                + ", ".join(sorted({result.registry.get(c).scale_type
                                    for c in result.scores.columns})))

    out: dict[str, pd.Series] = {}
    for cid in cols:
        y = result.scores[cid]
        ok = age.notna() & y.notna()
        if ok.sum() < 3:
            raise AnalysisError(f"{cid}: fewer than 3 samples with both age and a score")

        if method == "absolute":
            out[cid] = y - age
        elif method == "residual":
            out[cid] = _residual(y, age, ok)
        elif method == "within_group":
            if group is None or group not in result.obs.columns:
                raise AnalysisError(
                    "method='within_group' needs group= naming a column in obs")
            res = pd.Series(np.nan, index=y.index)
            for _, idx in result.obs.groupby(group).groups.items():
                sub = ok.loc[idx]
                if sub.sum() >= 3:
                    res.loc[idx] = _residual(y.loc[idx], age.loc[idx], sub)
            out[cid] = res
        else:
            raise AnalysisError("method must be 'absolute', 'residual' or 'within_group'")

    df = pd.DataFrame(out, index=result.scores.index)
    df.attrs["method"] = method
    return df


def _residual(y: pd.Series, age: pd.Series, ok: pd.Series) -> pd.Series:
    slope, intercept = np.polyfit(age[ok].to_numpy(float), y[ok].to_numpy(float), 1)
    return y - (slope * age + intercept)


# ---------------------------------------------------------------------------
# association and survival
# ---------------------------------------------------------------------------
def associate(result, outcome: str, *, covariates: Sequence[str] = ("age", "sex"),
              clocks: Sequence[str] | None = None) -> pd.DataFrame:
    """Ordinary least squares of each clock on an outcome, adjusted for covariates.

    Returns beta, standard error, t, p and the Benjamini-Hochberg q. OLS rather
    than a mixed model on purpose: the clock scores are the predictors here and
    the design is a single cross-section, so the extra machinery would buy
    nothing and hide the assumption.
    """
    if outcome not in result.obs.columns:
        raise AnalysisError(f"no {outcome!r} column in obs")
    y = pd.to_numeric(result.obs[outcome], errors="coerce")
    cols = list(clocks) if clocks else list(result.scores.columns)

    rows = []
    for cid in cols:
        design = pd.DataFrame({"score": result.scores[cid]})
        for cov in covariates:
            if cov in result.obs.columns:
                v = result.obs[cov]
                design[cov] = pd.to_numeric(v, errors="coerce") if v.dtype != object \
                    else pd.Categorical(v).codes
        d = design.join(y.rename("_y")).dropna()
        if len(d) < len(design.columns) + 3:
            rows.append({"clock": cid, "n": len(d), "beta": np.nan, "se": np.nan,
                         "t": np.nan, "p": np.nan})
            continue
        Xm = np.column_stack([np.ones(len(d)), d.drop(columns="_y").to_numpy(float)])
        yv = d["_y"].to_numpy(float)
        coef, *_ = np.linalg.lstsq(Xm, yv, rcond=None)
        resid = yv - Xm @ coef
        dof = len(d) - Xm.shape[1]
        s2 = float(resid @ resid) / dof
        se = np.sqrt(np.diag(s2 * np.linalg.pinv(Xm.T @ Xm)))
        t = coef[1] / se[1]
        rows.append({"clock": cid, "n": len(d), "beta": float(coef[1]),
                     "se": float(se[1]), "t": float(t),
                     "p": float(2 * stats.t.sf(abs(t), dof))})

    df = pd.DataFrame(rows).set_index("clock")
    df["q"] = _bh(df["p"].to_numpy())
    return df.sort_values("p")


def cox_hazard(result, *, time_col: str, event_col: str,
               clocks: Sequence[str] | None = None) -> pd.DataFrame:
    """Univariable Cox hazard ratio per clock, by Breslow-tied partial likelihood.

    Implemented directly rather than via lifelines to keep the dependency set
    small; it is Newton-Raphson on a one-parameter partial likelihood, which is
    twenty lines and exactly reproducible. Anything more elaborate -- competing
    risks, time-varying covariates -- belongs in a survival package, and the
    docs say so rather than pretending this covers it.
    """
    for c in (time_col, event_col):
        if c not in result.obs.columns:
            raise AnalysisError(f"no {c!r} column in obs")

    t = pd.to_numeric(result.obs[time_col], errors="coerce")
    e = pd.to_numeric(result.obs[event_col], errors="coerce")
    cols = list(clocks) if clocks else list(result.scores.columns)

    rows = []
    for cid in cols:
        x = result.scores[cid]
        ok = t.notna() & e.notna() & x.notna()
        if ok.sum() < 10 or e[ok].sum() < 3:
            rows.append({"clock": cid, "n": int(ok.sum()), "events": int(e[ok].sum()),
                         "hr": np.nan, "p": np.nan})
            continue
        beta, se = _cox_newton(x[ok].to_numpy(float), t[ok].to_numpy(float),
                               e[ok].to_numpy(float))
        z = beta / se if se > 0 else np.nan
        rows.append({"clock": cid, "n": int(ok.sum()), "events": int(e[ok].sum()),
                     "beta": beta, "se": se, "hr": float(np.exp(beta)),
                     "hr_lo": float(np.exp(beta - 1.96 * se)),
                     "hr_hi": float(np.exp(beta + 1.96 * se)),
                     "p": float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan})

    df = pd.DataFrame(rows).set_index("clock")
    if "p" in df:
        df["q"] = _bh(df["p"].to_numpy())
    return df


def _cox_newton(x: np.ndarray, t: np.ndarray, e: np.ndarray,
                iters: int = 40) -> tuple[float, float]:
    # Standardise first: raw clock scores span 20-90 for an age clock and
    # -2 to 2 for a log-hazard, and Newton on the raw scale converges for one
    # and oscillates for the other.
    mu, sd = float(x.mean()), float(x.std()) or 1.0
    z = (x - mu) / sd
    order = np.argsort(t)
    z, t, e = z[order], t[order], e[order]

    beta = 0.0
    for _ in range(iters):
        r = np.exp(beta * z)
        # risk set = everyone with time >= this event time (Breslow ties)
        cum_r = np.cumsum(r[::-1])[::-1]
        cum_rz = np.cumsum((r * z)[::-1])[::-1]
        cum_rz2 = np.cumsum((r * z * z)[::-1])[::-1]
        m1 = cum_rz / cum_r
        m2 = cum_rz2 / cum_r - m1**2
        grad = float(np.sum(e * (z - m1)))
        hess = float(np.sum(e * m2))
        if hess <= 1e-12:
            break
        step = grad / hess
        beta += step
        if abs(step) < 1e-9:
            break
    se = 1.0 / np.sqrt(hess) if hess > 0 else np.nan
    return beta / sd, se / sd


def _bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg, NaN-safe."""
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    v = p[ok]
    n = v.size
    order = np.argsort(v)
    ranked = v[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    q[ok] = out
    return q


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


# ---------------------------------------------------------------------------
# ComputAgeBench AA1 / AA2
# ---------------------------------------------------------------------------
@dataclass
class BenchmarkResult:
    per_dataset: pd.DataFrame
    summary_table: pd.DataFrame

    def summary(self) -> pd.DataFrame:
        return self.summary_table

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (f"BenchmarkResult({self.summary_table.shape[0]} clocks, "
                f"{self.per_dataset['dataset'].nunique()} dataset(s))")


def run_benchmark(result, *, condition_col: str = "condition", control: str = "HC",
                  dataset_col: str | None = None, age_col: str = "age",
                  alpha: float = 0.05) -> BenchmarkResult:
    """The AA1 and AA2 tests, and the score that combines them.

    **AA2** -- for a dataset with controls: is the condition group's age
    acceleration higher than its own controls'? One-sided Mann-Whitney, BH
    corrected across datasets.

    **AA1** -- for a dataset without controls: is the condition group's
    acceleration above zero? One-sided Wilcoxon signed-rank.

    **MedAE and MedE** -- median absolute error and median signed error against
    chronological age, on healthy controls only. MedE is the bias, and it
    discounts the AA1 credit in the total:

    .. code-block:: text

        total = AA2 + AA1 * (1 - max(0, MedE) / MedAE)

    Without that discount a clock that simply over-predicts everybody sweeps
    AA1, because every group looks accelerated when the baseline is wrong. This
    is the correction ComputAgeBench introduced and it is the reason the
    benchmark ranks differently from a plain "does it separate cases" tally.

    Median absolute error against chronological age is reported but never
    ranked on. A perfect chronological oracle would score zero here and be
    useless -- it would have no age acceleration to detect anything with.
    """
    if condition_col not in result.obs.columns:
        raise AnalysisError(f"no {condition_col!r} column in obs")

    obs = result.obs
    datasets = obs[dataset_col] if dataset_col and dataset_col in obs.columns \
        else pd.Series("all", index=obs.index)

    rows = []
    for cid in result.scores.columns:
        c = result.registry.get(cid)
        # age_years only, not everything that admits an acceleration. MedAE and
        # MedE are errors against chronological age, and the median absolute
        # difference between a telomere length in kilobases and an age in years
        # is a number with no meaning that would still sort a table.
        if c.scale_type != "age_years":
            continue
        for ds, idx in datasets.groupby(datasets).groups.items():
            sub_obs = obs.loc[idx]
            y = result.scores.loc[idx, cid]
            age = pd.to_numeric(sub_obs[age_col], errors="coerce")
            ok = y.notna() & age.notna()
            if ok.sum() < 6:
                continue

            is_ctrl = sub_obs[condition_col].astype(str) == control
            conds = [x for x in sub_obs[condition_col].astype(str).unique() if x != control]

            # Acceleration is fitted on the controls when there are any: a line
            # fitted through cases and controls together absorbs part of the
            # effect being tested.
            fit_mask = (is_ctrl & ok) if (is_ctrl & ok).sum() >= 3 else ok
            slope, intercept = np.polyfit(age[fit_mask].to_numpy(float),
                                          y[fit_mask].to_numpy(float), 1)
            aa = y - (slope * age + intercept)

            ctrl_aa = aa[is_ctrl & ok]
            med_ae = float(np.median(np.abs(y[is_ctrl & ok] - age[is_ctrl & ok]))) \
                if (is_ctrl & ok).sum() else np.nan
            med_e = float(np.median(y[is_ctrl & ok] - age[is_ctrl & ok])) \
                if (is_ctrl & ok).sum() else np.nan

            for cond in conds:
                m = (sub_obs[condition_col].astype(str) == cond) & ok
                if m.sum() < 3:
                    continue
                case_aa = aa[m]
                if len(ctrl_aa) >= 3:
                    stat, p = stats.mannwhitneyu(case_aa, ctrl_aa, alternative="greater")
                    test = "AA2"
                    delta = float(case_aa.median() - ctrl_aa.median())
                else:
                    stat, p = stats.wilcoxon(case_aa, alternative="greater")
                    test = "AA1"
                    delta = float(case_aa.median())
                rows.append({"clock": cid, "dataset": ds, "condition": cond,
                             "test": test, "n_case": int(m.sum()),
                             "n_control": int(len(ctrl_aa)), "delta": delta,
                             "statistic": float(stat), "p": float(p),
                             "medae": med_ae, "mede": med_e})

    per = pd.DataFrame(rows)
    if per.empty:
        raise AnalysisError(
            "the benchmark found no testable comparison.\n"
            f"  It needs a {condition_col!r} column with at least one group that is "
            f"not {control!r}, at least three samples in it, and an {age_col!r} column."
        )

    per["q"] = np.nan
    for test, idx in per.groupby("test").groups.items():
        per.loc[idx, "q"] = _bh(per.loc[idx, "p"].to_numpy())
    per["significant"] = per["q"] < alpha

    agg = []
    for cid, g in per.groupby("clock"):
        aa2 = int(((g["test"] == "AA2") & g["significant"]).sum())
        aa1 = int(((g["test"] == "AA1") & g["significant"]).sum())
        medae = float(np.nanmedian(g["medae"]))
        mede = float(np.nanmedian(g["mede"]))
        discount = 1.0 - (max(0.0, mede) / medae) if medae and np.isfinite(medae) else 1.0
        agg.append({"clock": cid, "AA2": aa2, "AA1": aa1,
                    "MedAE": round(medae, 3), "MedE": round(mede, 3),
                    "total": round(aa2 + aa1 * max(discount, 0.0), 3)})

    summary = pd.DataFrame(agg).set_index("clock").sort_values("total", ascending=False)
    return BenchmarkResult(per_dataset=per, summary_table=summary)
