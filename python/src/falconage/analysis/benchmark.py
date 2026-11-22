"""The AA1/AA2 benchmark across case/control studies."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from ..core.errors import AnalysisError
from ._common import _bh


# ---------------------------------------------------------------------------
# ComputAgeBench AA1 / AA2
# ---------------------------------------------------------------------------
@dataclass
class BenchmarkResult:
    per_dataset: pd.DataFrame
    summary_table: pd.DataFrame
    #: One row per clock, dataset and condition for every clock whose scale
    #: admits a group comparison at all -- not only the year-scaled ones AA1
    #: and AA2 are defined for. See :func:`run_benchmark`.
    rank_effects: pd.DataFrame = field(default_factory=pd.DataFrame)

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
    ranks = _rank_effects(result, obs, datasets, condition_col, control, alpha)
    return BenchmarkResult(per_dataset=per, summary_table=summary,
                           rank_effects=ranks)


def _rank_effects(result, obs, datasets, condition_col, control, alpha):
    """Case-against-control separation for every clock the scale allows.

    WHY THIS EXISTS SEPARATELY FROM AA1/AA2. Those two are defined on age
    acceleration, so `run_benchmark` above considers only `age_years` clocks:
    a median absolute error against chronological age is not a quantity for a
    pace ratio or a division count. That is right, and it left seven of the
    seventeen clocks a run scores with no group comparison of any kind, which
    the clock atlas then drew as an absent row rather than as a clock whose
    question is a different one.

    What is legal for the rest is declared already. `LEGAL_OPS` gives
    `divisions`, `pace_ratio` and `telomere_kb` a group `difference` and most
    scales a `rank`, so a rank test between cases and controls on the score
    itself is permitted where an acceleration is not. Cliff's delta reports it
    without units: -1 and +1 are complete separation, 0 is none, and the
    number means the same thing on a telomere length in kilobases as on a
    count of stem-cell divisions.

    NOT AGE ADJUSTED, deliberately and visibly. AA2 residualises against
    chronological age before testing; this cannot, because residualising is
    exactly the operation these scales do not admit. A cohort whose cases are
    older than its controls will show separation here for that reason alone,
    so the figure labels the panel as unadjusted and reports the group age gap
    beside the effect rather than leaving a reader to assume it was handled.
    """
    rows = []
    for cid in result.scores.columns:
        legal = result.registry.get(cid).legal_operations
        if not ({"rank", "difference"} & set(legal)):
            continue
        for ds, idx in datasets.groupby(datasets).groups.items():
            sub_obs = obs.loc[idx]
            y = pd.to_numeric(result.scores.loc[idx, cid], errors="coerce")
            conds = sub_obs[condition_col].astype(str)
            ctrl = y[(conds == control) & y.notna()]
            if len(ctrl) < 3:
                continue
            ages = pd.to_numeric(sub_obs.get("age", pd.Series(index=sub_obs.index,
                                                              dtype=float)),
                                 errors="coerce")
            for cond in [c for c in conds.unique() if c != control]:
                case = y[(conds == cond) & y.notna()]
                if len(case) < 3:
                    continue
                stat, p = stats.mannwhitneyu(case, ctrl, alternative="two-sided")
                # Cliff's delta straight from U: the fraction of case-control
                # pairs in which the case scores higher, rescaled to [-1, 1].
                cliffs = 2.0 * float(stat) / (len(case) * len(ctrl)) - 1.0
                gap = float(ages[(conds == cond)].median()
                            - ages[(conds == control)].median()) \
                    if ages.notna().any() else np.nan
                rows.append({"clock": cid, "dataset": ds, "condition": cond,
                             "scale_type": result.registry.get(cid).scale_type,
                             "n_case": int(len(case)), "n_control": int(len(ctrl)),
                             "cliffs_delta": round(cliffs, 4),
                             "median_case": float(case.median()),
                             "median_control": float(ctrl.median()),
                             "age_gap_years": gap,
                             "p": float(p)})

    if not rows:
        return pd.DataFrame(columns=["clock", "dataset", "condition", "scale_type",
                                     "n_case", "n_control", "cliffs_delta",
                                     "median_case", "median_control",
                                     "age_gap_years", "p", "q", "significant"])
    out = pd.DataFrame(rows)
    out["q"] = _bh(out["p"].to_numpy())
    out["significant"] = out["q"] < alpha
    return out
