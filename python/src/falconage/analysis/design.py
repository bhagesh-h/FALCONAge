"""Study design: sample size, detectable effects, and the multi-clock consensus rule."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from ..core.errors import AnalysisError
from ._common import _bh, pc_counterpart
from .acceleration import _residual


# ---------------------------------------------------------------------------
# study design
# ---------------------------------------------------------------------------
@dataclass
class PowerResult:
    """What a design can see, and how much of the cost is the assay."""

    clock: str
    effect: float
    sd: float
    alpha: float
    power: float
    n_per_group: int
    n_total: int
    icc: float | None
    icc_source: str
    n_if_perfectly_measured: int | None
    replicates: int
    assumptions: str

    def __repr__(self) -> str:  # pragma: no cover - display only
        pen = ("" if self.n_if_perfectly_measured is None else
               f"; {self.n_total - self.n_if_perfectly_measured} of those "
               f"samples exist only to average out assay noise")
        return (f"PowerResult({self.clock}: n={self.n_per_group} per group "
                f"for {self.effect:g} at {self.power:.0%} power{pen})")


def power(clock: str, *, effect: float, sd: float | None = None,
          result=None, icc: float | None = None, alpha: float = 0.05,
          power: float = 0.80, replicates: int = 1,
          registry=None) -> PowerResult:
    """How many samples to see an effect of this size on this clock.

    The first thing a laboratory needs, and it is needed before any array is
    run. Two independent groups, two-sided:

        n per group = 2 (z_{1-alpha/2} + z_{power})^2 sigma^2 / delta^2

    WHY RELIABILITY IS PART OF THE ANSWER. The sigma a user measures already
    contains the assay's noise. Splitting it out with the clock's test-retest
    ICC says how much of the sample size is buying signal and how much is
    averaging out the instrument -- which is the arithmetic behind the finding
    that the original clocks need 3-16 replicates per condition where their PC
    versions need 1-2 (Nat Aging 2022, s43587-022-00248-2).

    Parameters
    ----------
    effect
        The difference worth detecting, in the clock's own units. No default:
        a power calculation with an assumed effect size is a way of writing
        down an assumption without noticing.
    sd
        Population SD of the score. Taken from ``result`` when one is given.
    result
        A scored :class:`~falconage.score.FalconResult` from a pilot. Supplies
        ``sd``, and -- if :func:`falconage.technical_se` has been called on it --
        a measured ICC for *this* laboratory rather than a published one.
    replicates
        Assay each sample this many times and average. Reduces the error
        variance by the same factor, so it trades array cost against sample
        recruitment.

    Raises
    ------
    AnalysisError
        When no ``sd`` is available. There is no sensible default: the answer
        scales with its square, so a guessed SD is a guessed sample size
        reported to three significant figures.
    """
    from ..registry import load as _load

    reg = registry if registry is not None else (
        result.registry if result is not None else _load())
    c = reg.get(clock)

    src = "given"
    if sd is None and result is not None and clock in result.scores.columns:
        sd = float(result.scores[clock].std(ddof=1))
        src = f"cohort SD of {result.scores.shape[0]} scored sample(s)"
    if sd is None or not np.isfinite(sd) or sd <= 0:
        raise AnalysisError(
            f"power() needs the population SD of {clock} and none was given.\n"
            "  Pass sd=, or pass result= from a pilot run so it can be measured.\n"
            "  It is not defaulted because n scales with sd squared, so a "
            "guess here is a guessed answer with a confident number of digits.")

    icc_source = "given" if icc is not None else "not established"
    if icc is None and result is not None and getattr(result, "se", None) is not None \
            and clock in result.se.columns:
        se = float(np.sqrt(np.mean(result.se[clock].to_numpy(dtype=float) ** 2)))
        if sd > 0:
            # Not clipped to zero. An implied ICC at or below zero means the
            # cohort's spread on this clock is no larger than the assay's noise,
            # which is a real and reportable state -- usually a narrow age range
            # rather than a broken clock -- and rounding it up to "0.0, fine"
            # would hide the one thing the user needs to know.
            icc = float(1.0 - (se / sd) ** 2)
            icc_source = ("measured on this cohort by technical_se()" if icc > 0 else
                          "measured on this cohort by technical_se(), and it came "
                          "out <= 0: the assay noise is as large as the spread "
                          "between these samples, so no reliability-adjusted n "
                          "can be given")
    if icc is None and c.reliability.technical_icc is not None:
        icc = float(c.reliability.technical_icc)
        icc_source = c.reliability.source or "registry"

    # Averaging r replicates divides the error variance by r; the true-signal
    # variance is untouched. sigma_r^2 = sigma_true^2 + sigma_err^2 / r.
    sd_eff = sd
    if icc is not None and icc > 0 and replicates > 1:
        var_true, var_err = icc * sd ** 2, (1.0 - icc) * sd ** 2
        sd_eff = float(np.sqrt(var_true + var_err / replicates))

    z_a = float(stats.norm.ppf(1.0 - alpha / 2.0))
    z_b = float(stats.norm.ppf(power))
    per = 2.0 * (z_a + z_b) ** 2 * sd_eff ** 2 / float(effect) ** 2
    n_per = int(np.ceil(per))

    ideal = None
    if icc is not None and icc > 0:
        ideal_sd = float(np.sqrt(icc)) * sd
        ideal = int(np.ceil(2.0 * (z_a + z_b) ** 2 * ideal_sd ** 2 / float(effect) ** 2))

    return PowerResult(
        clock=clock, effect=float(effect), sd=float(sd), alpha=alpha, power=power,
        n_per_group=n_per, n_total=2 * n_per, icc=icc, icc_source=icc_source,
        n_if_perfectly_measured=ideal, replicates=replicates,
        assumptions=("two independent groups, two-sided, equal variance, "
                     f"sd from {src}"),
    )


def detectable_effect(clock: str, n_per_group: int, **kw) -> float:
    """The smallest effect a given n can see. The inverse of :func:`power`."""
    probe = power(clock, effect=1.0, **kw)
    return float(np.sqrt(probe.n_per_group / n_per_group))


# ---------------------------------------------------------------------------
# the intervention false-positive protocol
# ---------------------------------------------------------------------------
#: Which generations count as "trained on an outcome" rather than on age. A
#: change seen only in the first-generation column is the published signature of
#: a false positive (PMC11526921).
_OUTCOME_TRAINED = {"second", "pace", "causal"}


@dataclass
class ConsensusReport:
    """Whether a group difference survives the multi-clock rule.

    ``verdict`` is one of ``supported``, ``unsupported`` or ``inconclusive``,
    and ``why`` always carries the counts it was computed from -- a verdict
    without its arithmetic is an oracle, and this package's whole posture is the
    opposite of that.
    """

    verdict: str
    why: str
    table: pd.DataFrame
    n_tests: int
    alpha: float
    correction: str
    design: str = "independent"
    #: Clocks not tested, and why: a clock whose scores do not vary across
    #: these samples carries no information about a difference between them.
    left_out: dict = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        return self.table

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"ConsensusReport({self.verdict}: {self.why})"


def _paired_test(y: pd.Series, g: pd.Series, s: pd.Series, ref: str, other: str) -> dict | None:
    """Paired t on each person's change from ``ref`` to ``other``."""
    wide = pd.DataFrame({"y": y, "g": g, "s": s}).dropna().pivot(
        index="s", columns="g", values="y")
    if ref not in wide or other not in wide:
        return None
    diff = (wide[other] - wide[ref]).dropna()
    if len(diff) < 2:
        return None
    sd = float(diff.std(ddof=1))
    t, p = stats.ttest_1samp(diff, 0.0)
    return {"n_case": len(diff), "n_control": len(diff), "n_subjects": len(diff),
            "delta": float(diff.mean()),
            "cohens_dz": float(diff.mean() / sd) if sd > 0 else np.nan,
            "t": float(t), "p": float(p), "df": len(diff) - 1,
            # MDC95 = 1.96 * sqrt(2) * SEM, and a difference of two visits has
            # variance 2 * SEM^2, so it is 1.96 times the SD of the differences.
            "sem_within": sd / np.sqrt(2.0), "mdc95": 1.96 * sd}


def _mixed_test(y: pd.Series, g: pd.Series, s: pd.Series, ref: str,
                others: list[str]) -> dict | None:
    """``y ~ visit + (1 | person)`` by REML; one contrast per non-reference level."""
    from .mixed import fit_random_intercept

    d = pd.DataFrame({"y": y, "g": g, "s": s}).dropna()
    d = d[d["g"].isin([ref, *others])]
    if d["s"].nunique() < 3 or not all((d["g"] == lv).any() for lv in [ref, *others]):
        return None
    X = np.column_stack([np.ones(len(d))] + [(d["g"] == lv).to_numpy(float) for lv in others])
    try:
        fit = fit_random_intercept(d["y"].to_numpy(float), X, d["s"].to_numpy())
    except (ValueError, np.linalg.LinAlgError):
        return None
    idx = list(range(1, len(others) + 1))
    try:
        F, p, q, d2 = fit.wald(idx)
    except np.linalg.LinAlgError:
        return None
    se = np.sqrt(np.diag(fit.cov))
    row = {"n_case": int((d["g"] != ref).sum()), "n_control": int((d["g"] == ref).sum()),
           "n_subjects": int(d["s"].nunique()), "p": p, "df": d2,
           "sem_within": float(np.sqrt(fit.sigma2)),
           "mdc95": float(1.96 * np.sqrt(2.0) * np.sqrt(fit.sigma2))}
    for k, lv in zip(idx, others):
        row[f"delta_{lv}"] = float(fit.beta[k])
    if len(others) == 1:
        row["delta"] = float(fit.beta[1])
        row["t"] = float(fit.beta[1] / se[1])
    else:
        row["delta"] = max((row[f"delta_{lv}"] for lv in others), key=abs)
        row["F"] = F
    return row


def consensus(result, group_col: str, *, reference=None, alpha: float = 0.05,
              age_col: str = "age", min_generations: int = 2,
              clocks: Sequence[str] | None = None, design: str = "independent",
              subject_col: str | None = None) -> ConsensusReport:
    """Does a group difference hold up across clocks, or is it one clock?

    Implements the decision rule from *When to Trust Epigenetic Clocks*
    (PMC11526921). Re-analysing six intervention datasets, the authors found
    that in five of them exactly one clock reached significance -- a
    first-generation clock every time -- and four of those five lost it under
    multiple-testing correction. In no case did the principal-component version
    of the same clock corroborate the finding. Their conclusion, stated plainly:
    **a single significant clock after an intervention is likely a false
    positive.**

    So this runs every scored clock, corrects across the whole set actually
    tested, and returns a verdict rather than a table of p-values to pick from:

    ``supported``
        Significant after Bonferroni, across at least ``min_generations``
        generations, including at least one outcome-trained clock (second
        generation, pace, or causal).
    ``unsupported``
        One clock, or first-generation clocks only, or nothing surviving
        correction.
    ``inconclusive``
        Something in between -- most often several clocks at BH but not at
        Bonferroni.

    Each clock is tested on its acceleration residual where that is a legal
    operation for its scale, and on the raw score where it is not. A pace of
    aging has no residual to take, and taking one anyway is the units error
    ``LEGAL_OPS`` exists to prevent.

    **The corroboration column, and when it is empty.** ``high_reliability_partner``
    names each clock's principal-component version and ``partner_corroborates``
    says whether that version agreed, which is the paper's sharpest single
    diagnostic. It is ``None`` when the partner was not scored, and on a default
    install that is every one of them: the PC clocks are ``licensed``, since
    the authors' data file carries no licence and is not ours to ship.
    ``why`` says so explicitly rather than omitting the clause, because a
    verdict that silently drops its strongest check reads like one that passed
    it. ``fa.registry.load().import_pc_clocks(path)`` registers five of them
    from the authors' ``CalcAllPCClocks.RData`` and turns it on.

    **Repeated measures.** ``design`` says how the samples relate:

    ``"independent"``
        Two groups of different people; Welch's t-test.
    ``"paired"``
        ``group_col`` holds two visits and ``subject_col`` the person; each
        person's change is tested with a paired t-test. A person needs one
        sample at each visit.
    ``"mixed"``
        ``group_col`` holds two or more visits: ``y ~ visit + (1 | person)``,
        fitted by REML as nlme's ``lme`` does (:mod:`falconage.analysis.mixed`),
        with Wald t or F on nlme's containment degrees of freedom. A person with
        a missed visit still contributes. With more than two visits the test is
        the joint one over every visit contrast, and each contrast is reported.

    Both repeated designs add ``mdc95``, the minimum detectable change,
    :math:`1.96\\sqrt{2}\\,SEM` with the SEM the within-person SD across these
    visits (Weir 2005, doi:10.1519/15184.1), and ``below_mdc``: a group mean can
    move significantly by less than any one person can be shown to have moved,
    and ``why`` says how many significant clocks did. That SEM carries
    day-to-day biology and assay noise together; :func:`technical_se` gives the
    assay part alone, and :func:`variance_components` separates the two when the
    design has technical replicates.
    """
    reg = result.registry
    if design not in ("independent", "paired", "mixed"):
        raise AnalysisError(f"design must be independent, paired or mixed, not {design!r}")
    if group_col not in result.obs.columns:
        raise AnalysisError(f"no {group_col!r} column in obs")
    if design != "independent":
        if subject_col is None or subject_col not in result.obs.columns:
            raise AnalysisError(
                f"design={design!r} needs subject_col, the column naming the person "
                "each sample came from")
        subj = result.obs[subject_col].astype(str)
    g = result.obs[group_col].astype(str)
    levels = [x for x in dict.fromkeys(g) if x and x.lower() not in ("nan", "none")]
    if len(levels) < 2 or (design != "mixed" and len(levels) != 2):
        raise AnalysisError(
            f"{group_col!r} has {len(levels)} levels {levels[:4]}; this design "
            "compares exactly two (design='mixed' takes two or more). Subset the "
            "result first.")
    ref = str(reference) if reference is not None else levels[0]
    if ref not in levels:
        raise AnalysisError(f"reference {ref!r} is not a level of {group_col!r}")
    others = [x for x in levels if x != ref]
    other = others[0]
    if design == "paired" and pd.DataFrame({"s": subj, "g": g}).duplicated().any():
        raise AnalysisError(
            "some person has more than one sample at a visit; a paired test needs "
            "one each. Average technical replicates first, or use design='mixed'.")

    use = list(clocks) if clocks else list(result.scores.columns)
    rows = []
    left_out: dict[str, str] = {}
    for cid in use:
        c = reg.get(cid)
        y = result.scores[cid].astype(float)
        if y.dropna().nunique() <= 1:
            left_out[cid] = "the same score for every sample"
            continue
        basis = "score"
        if "acceleration" in c.legal_operations and age_col in result.obs.columns:
            age = pd.to_numeric(result.obs[age_col], errors="coerce")
            ok = age.notna() & y.notna()
            if ok.sum() > 2:
                y, basis = _residual(y, age, ok), "residual"
        if design != "independent":
            row = (_paired_test(y, g, subj, ref, other) if design == "paired"
                   else _mixed_test(y, g, subj, ref, others))
            if row is None:
                continue
            row["below_mdc"] = bool(abs(row["delta"]) < row["mdc95"])
            rows.append({"clock": cid, "generation": c.generation, "basis": basis, **row})
            continue
        a = y[(g == other).to_numpy()].dropna()
        b = y[(g == ref).to_numpy()].dropna()
        if len(a) < 2 or len(b) < 2:
            continue
        t, p = stats.ttest_ind(a, b, equal_var=False)
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
        rows.append({
            "clock": cid, "generation": c.generation, "basis": basis,
            "n_case": len(a), "n_control": len(b),
            "delta": float(a.mean() - b.mean()),
            "cohens_d": float((a.mean() - b.mean()) / pooled) if pooled > 0 else np.nan,
            "t": float(t), "p": float(p),
        })

    if not rows:
        raise AnalysisError(
            "no clock had at least two samples in both groups" if design == "independent"
            else "no clock had enough people measured at the compared visits")

    tab = pd.DataFrame(rows)
    n = len(tab)
    tab["q_bh"] = _bh(tab["p"].to_numpy())
    # Bonferroni across the tests actually run, which is the number the paper
    # corrects over -- not across the registry, and not across the clocks
    # someone might have run.
    tab["p_bonferroni"] = np.clip(tab["p"] * n, 0, 1)
    tab["sig_bh"] = tab["q_bh"] < alpha
    tab["sig_bonferroni"] = tab["p_bonferroni"] < alpha

    strict = tab[tab["sig_bonferroni"]]
    gens = set(strict["generation"])
    outcome = gens & _OUTCOME_TRAINED

    # PC corroboration. The discriminating signal in the paper: for every
    # sporadic first-generation hit, the PC version of the same clock was
    # silent.
    scored = set(tab["clock"])
    tab["high_reliability_partner"] = [pc_counterpart(c) or "" for c in tab["clock"]]
    tab["partner_corroborates"] = [
        bool(tab.loc[tab["clock"] == pc, "sig_bonferroni"].iloc[0])
        if pc and pc in scored else None
        for pc in tab["high_reliability_partner"]]

    pc_checks, unpaired = [], []
    for cid in strict["clock"]:
        pc = pc_counterpart(cid)
        if not pc:
            continue
        if pc in scored:
            agreed = bool(tab.loc[tab["clock"] == pc, "sig_bonferroni"].iloc[0])
            pc_checks.append(f"{pc} {'agrees' if agreed else 'does NOT corroborate'}")
        else:
            unpaired.append(f"{cid} (would need {pc})")

    counts = (f"{len(strict)} of {n} clock(s) significant at Bonferroni "
              f"(alpha {alpha}), {int(tab['sig_bh'].sum())} at BH; "
              f"generations {sorted(gens) or 'none'}")
    if design != "independent" and len(strict):
        small = int(strict["below_mdc"].sum())
        counts += (f"; {small} of the {len(strict)} significant mean change(s) are "
                   "smaller than the minimum detectable change for one person "
                   "(mdc95), so they describe the group and not any individual")
    if pc_checks:
        counts += "; " + "; ".join(pc_checks)
    if unpaired:
        # Not a footnote. The PC clocks are licensed, so on a default install
        # this check cannot run at all, and a verdict that quietly omitted it
        # would read as though it had passed.
        counts += ("; high-reliability corroboration NOT checked for "
                   + ", ".join(unpaired)
                   + " -- register the authors' PC clocks with "
                     "fa.registry.load().import_pc_clocks(path) to enable it")

    if len(strict) == 0:
        verdict, why = "unsupported", f"nothing survives correction -- {counts}"
    elif len(strict) == 1:
        verdict = "unsupported"
        why = ("a single significant clock after an intervention is likely a "
               f"false positive (PMC11526921) -- {counts}")
    elif not outcome:
        verdict = "unsupported"
        why = ("only age-trained clocks moved, and no clock trained on an outcome "
               "(mortality, pace of aging, disease) did; an effect that replicates "
               f"moves those as well -- {counts}")
    elif len(gens) < min_generations:
        verdict, why = "inconclusive", (
            f"significant clocks span {len(gens)} generation(s), "
            f"{min_generations} wanted -- {counts}")
    else:
        verdict, why = "supported", counts

    if left_out:
        why += (f"; {len(left_out)} clock(s) left out because their scores did not "
                f"vary: {', '.join(sorted(left_out))}")
    return ConsensusReport(verdict=verdict, why=why, table=tab.set_index("clock"),
                           n_tests=n, alpha=alpha, correction="bonferroni+bh",
                           design=design, left_out=left_out)
