"""Association and survival models, and which marker carries an association."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from ..core.errors import AnalysisError
from ._common import _bh
from .acceleration import _level, _strata


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
    if not pd.api.types.is_numeric_dtype(result.obs[outcome]):
        raise AnalysisError(
            f"{outcome!r} is not numeric; code it (for example 0/1) before an OLS "
            "association, rather than have every value read as missing")
    y = pd.to_numeric(result.obs[outcome], errors="coerce")
    cols = list(clocks) if clocks else list(result.scores.columns)

    # A text covariate becomes one indicator per level after the first. Codes
    # would make a three-level factor an ordinal number and code a missing
    # value as -1; and on pandas 3 text is dtype "str", not object, so the old
    # object test sent sex through to_numeric and dropped every row.
    cov_frame = pd.DataFrame(index=result.obs.index)
    for cov in covariates:
        if cov not in result.obs.columns:
            continue
        v = result.obs[cov]
        if pd.api.types.is_numeric_dtype(v):
            cov_frame[cov] = pd.to_numeric(v, errors="coerce")
        else:
            dummies = pd.get_dummies(v.astype("object"), prefix=cov, drop_first=True,
                                     dtype=float)
            dummies[v.isna().to_numpy()] = np.nan
            cov_frame = cov_frame.join(dummies)

    rows = []
    for cid in cols:
        design = pd.DataFrame({"score": result.scores[cid]}).join(cov_frame)
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


def _cox_breslow(X: np.ndarray, t: np.ndarray, e: np.ndarray,
                 iters: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Cox proportional hazards by Newton-Raphson on Breslow's partial likelihood.

    Cox 1972 with Breslow's 1974 treatment of tied event times: the risk set at
    an event time is everyone whose time is at least that time, tied subjects
    included. Returns coefficients and their covariance (the inverse observed
    information), on the scale of the columns of ``X``. Agrees with R's
    ``survival::coxph(..., ties = "breslow")`` to the sixth decimal, which the
    tests check.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    # Standardise for the Newton steps: clock scores span 20 to 90 and
    # log-hazards -2 to 2, and Newton on the raw scale converges for one and
    # oscillates for the other. Transformed back below.
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    order = np.argsort(t, kind="mergesort")
    Z, t, e = Z[order], np.asarray(t, float)[order], np.asarray(e, float)[order]
    # First row of each tied block: a reverse cumulative sum read from there
    # covers every subject with time >= this time, ties included.
    first = np.searchsorted(t, t, side="left")
    n, p = Z.shape
    beta = np.zeros(p)
    info = np.eye(p)
    for _ in range(iters):
        r = np.exp(Z @ beta)
        s0 = np.cumsum(r[::-1])[::-1][first]
        s1 = np.cumsum((r[:, None] * Z)[::-1], axis=0)[::-1][first]
        s2 = np.cumsum((r[:, None, None] * Z[:, :, None] * Z[:, None, :])[::-1],
                       axis=0)[::-1][first]
        m1 = s1 / s0[:, None]
        grad = (e[:, None] * (Z - m1)).sum(axis=0)
        info = (e[:, None, None] * (s2 / s0[:, None, None]
                                    - m1[:, :, None] * m1[:, None, :])).sum(axis=0)
        try:
            step = np.linalg.solve(info, grad)
        except np.linalg.LinAlgError:
            break
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    try:
        cov = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        cov = np.full((p, p), np.nan)
    return beta / sd, cov / np.outer(sd, sd)


def _cox_newton(x: np.ndarray, t: np.ndarray, e: np.ndarray,
                iters: int = 50) -> tuple[float, float]:
    """One covariate: coefficient and standard error from :func:`_cox_breslow`."""
    b, cov = _cox_breslow(np.asarray(x, dtype=np.float64)[:, None], t, e, iters=iters)
    return float(b[0]), float(np.sqrt(cov[0, 0])) if cov[0, 0] > 0 else np.nan


# ---------------------------------------------------------------------------
# which marker carries an association
# ---------------------------------------------------------------------------
#: Label of the baseline row of :func:`leave_one_marker_out`: nothing removed.
FULL_PANEL = "(none)"


def _marker_frame(x, what: str) -> pd.DataFrame:
    """Markers as one table: a DataFrame as given, a FalconData as X joined to obs.

    The join is the one :class:`~falconage.models.clinical.ClinicalClock` makes
    before scoring, so a column found in both keeps the value from ``X``.
    """
    if isinstance(x, pd.DataFrame):
        return x
    if hasattr(x, "X") and hasattr(x, "obs"):
        return x.X.join(x.obs, how="left", rsuffix="_obs")
    raise AnalysisError(f"{what}= must be a FalconData or a DataFrame of markers, "
                        f"not {type(x).__name__}")


def _numeric_covariates(obs: pd.DataFrame, covariates: Sequence[str]
                        ) -> tuple[pd.DataFrame, list[str]]:
    """Covariates as numbers :func:`associate` can use: text becomes indicators.

    One 0/1 column per level after the first, missing where the value is.
    Needed because on pandas 3 a text column has dtype ``str`` rather than
    ``object``, and :func:`associate` then reads ``sex`` through
    ``to_numeric`` as all-missing and drops every row. Indicators rather than
    integer codes, so a covariate with three levels is not treated as a dose.
    """
    obs = obs.copy()
    out: list[str] = []
    for cov in covariates:
        if cov not in obs.columns or pd.api.types.is_numeric_dtype(obs[cov]):
            out.append(cov)
            continue
        lv = pd.Series([_level(v) for v in obs[cov].tolist()], index=obs.index, dtype=object)
        for level in sorted(set(lv.dropna()))[1:]:
            name = f"{cov}[{level}]"
            obs[name] = (lv == level).astype(float).where(lv.notna())
            out.append(name)
    return obs, out


def leave_one_marker_out(result, clock: str, *, data, reference,
                         markers: Sequence[str] | None = None,
                         test: str | None = None,
                         covariates: Sequence[str] = ("age", "sex"),
                         age_col: str = "age",
                         sex_col: str | None = None) -> pd.DataFrame:
    """Recompute a clinical clock without each marker in turn, and say what moved.

    A composite clock associated with an outcome does not say which of its
    markers carries the association. A KDM built on nine markers can owe most
    of an association to one of them, and reporting it as "biological aging"
    then says more than the data do.
    This removes one marker at a time, recomputes the clock, and reports how
    far the score moved and, with ``test=``, how much of the association went
    with the marker.

    Parameters
    ----------
    result
        The scored :class:`~falconage.score.FalconResult`. Its ``clock``
        column is the baseline, and its ``obs`` supplies ``test`` and the
        covariates.
    clock
        ``phenoage``, ``kdm`` or ``hd``, or any clock the registry computes
        with one of those three formulas.
    data
        The marker data the result was scored from, as a
        :class:`~falconage.core.FalconData` or a DataFrame indexed by sample.
        A result holds scores, not markers, so they are passed again.
    reference
        The reference population's markers, as a FalconData or a DataFrame.
        For KDM and HD, the rows the scoring reference was fitted on (the
        frame passed to ``fit_kdm`` or ``fit_hd``), because removing a marker
        means refitting on what remains. For PhenoAge, the population whose
        marker means a removed marker is held at; see Notes.
    markers
        For KDM and HD, the panel the scoring reference was fitted on, in any
        order; required, because neither clock has a fixed panel. For
        PhenoAge the panel is fixed at Levine's nine biomarkers, and this
        chooses which of them to remove (all nine by default).
    test
        A column of ``result.obs``. When given, each version of the clock is
        tested against it with :func:`associate`: ordinary least squares of
        ``test`` on the score and ``covariates``.
    covariates
        Passed to :func:`associate`, with text columns (``sex``) entered as
        one indicator per level after the first. Adjusting for age makes the
        tested quantity the age-independent part of the score, the same thing
        an acceleration residual isolates.
    sex_col
        KDM and HD only: fit the reference separately within each level of
        this column (present in both ``data`` and ``reference``) and score
        each sample against its own level's fit, as
        :func:`~falconage.models.clinical.validate_panel` and BioAge do. Leave
        it ``None`` when the result was scored with one pooled reference.

    Returns
    -------
    pandas.DataFrame
        One row per removed marker, after a first row ``"(none)"`` for the
        full panel, with columns:

        ``n``
            Samples scored by both the full and the reduced clock.
        ``r_spearman``
            Rank correlation of the reduced clock with the full one.
        ``mean_change``, ``mean_abs_change``
            Mean and mean absolute of ``reduced - full``, in the clock's unit.
            Only where the scale admits a ``difference``; NaN for HD, whose
            Mahalanobis distance shrinks when a dimension is removed whatever
            that dimension carried, so its raw change measures the panel size.
        ``n_test``, ``beta``, ``se``, ``p``
            With ``test=``: the association from :func:`associate`, ``beta``
            in units of ``test`` per unit of the clock.
        ``partial_r``
            With ``test=``: the partial correlation of ``test`` with the score
            given the covariates, ``t / sqrt(t^2 + df)`` from the same fit, the
            t test of a regression coefficient being also the test of the
            corresponding partial correlation (Cohen, Cohen, West and Aiken
            2003, *Applied Multiple Regression/Correlation Analysis for the
            Behavioral Sciences*, 3rd ed.). Scale-free, so the full and
            reduced clocks, whose spreads differ, compare directly.
        ``attenuation``
            With ``test=``: ``1 - partial_r / partial_r(full panel)``, the
            share of the full panel's association lost without the marker. 1
            means the marker carried all of it; below 0, what remains carries
            the association more cleanly without it. Read it only when the
            full panel's association is itself clearly non-zero.

        Rows after the first are sorted by ``attenuation`` (largest first)
        with ``test=``, and otherwise by ``r_spearman`` (smallest first), so
        the marker that matters most is on top either way. ``attrs`` records
        the clock, how a marker was removed, the values PhenoAge's markers
        were held at, and the test and covariates.

    Raises
    ------
    AnalysisError
        For a clock that is not one of the three clinical formulas, a missing
        ``reference``, ``markers`` (KDM, HD), marker or ``test`` column, a
        panel too small to lose a marker (KDM needs two, HD three), or when
        recomputing the full panel from ``data`` and ``reference`` does not
        reproduce the scores in ``result``: the decomposition would then
        describe some other clock.

    Notes
    -----
    **KDM and HD are refitted.** Neither has fixed weights: KDM's come from
    per-marker regressions on age in the reference, HD's from its covariance,
    so the clock without a marker is the clock fitted without it, by
    :func:`~falconage.models.clinical.fit_kdm` or
    :func:`~falconage.models.clinical.fit_hd` exactly as the full one was.
    For KDM that also re-estimates ``r_char``, ``s_R`` and ``s_BA^2`` on the
    smaller panel, as :func:`~falconage.models.clinical.validate_panel` does.
    This is leave-one-covariate-out, refitting without the variable (Lei et
    al. 2018, J Am Stat Assoc 113:1094-1111,
    doi:10.1080/01621459.2017.1307116).

    **PhenoAge is not refitted.** Its weights are Levine's (2018), fitted on
    NHANES III mortality, and refitting them on a user's reference would
    produce a different clock that happens to share a name. A marker is
    removed instead by holding it at its mean in ``reference``; for CRP, the
    mean of ``log(CRP)``, since that is the quantity its weight multiplies,
    which is the geometric mean of CRP. This is well defined because
    PhenoAge is an affine function of its linear predictor ``xb``: the
    Gompertz inversion gives ``-log(1 - M) = exp(xb) (exp(120 gamma) - 1) /
    gamma``, so ``PhenoAge = const + xb / 0.090165``. Holding marker ``j`` at
    ``c_j`` therefore moves each person by exactly
    ``-beta_j (x_j - c_j) / 0.090165`` years, whatever the other markers are.
    That is the marker's additive contribution against the reference as
    baseline, the attribution a linear model gives each input (Strumbelj and
    Kononenko 2014, Knowl Inf Syst 41:647-665, doi:10.1007/s10115-013-0679-x;
    Lundberg and Lee 2017, NeurIPS, the linear case of SHAP). Which mean is
    used moves every score by the same amount, so ``p`` and ``attenuation``
    do not depend on it; ``mean_change`` does, and against a reference such
    as NHANES III it says how much of the cohort's departure from that
    population the marker accounts for. Passing this cohort's own markers as
    ``reference`` holds each at the cohort mean and makes ``mean_change``
    zero by construction.

    **What it cannot say.** Refitting lets correlated markers take over from
    the removed one, so two markers carrying the same signal can each show
    little attenuation although together they carry all of it; holding a
    PhenoAge marker fixed does not, because the other weights stay put. A
    marker with high attenuation carries the association given the rest of
    the panel, which is a statement about the clock, not a causal one about
    the marker.
    """
    from types import SimpleNamespace

    from ..models import clinical as clin

    if clock not in result.scores.columns:
        raise AnalysisError(
            f"{clock!r} is not scored in this result; it has "
            f"{', '.join(map(str, result.scores.columns))}")
    c = result.registry.get(clock)
    formula = c.formula
    if formula not in ("phenoage", "kdm", "hd"):
        raise AnalysisError(
            f"leave_one_marker_out is defined for the clinical clocks (PhenoAge, "
            f"KDM, HD); {clock} is a {c.model_type} {c.data_type} clock.\n"
            "  For a clock with fixed coefficients over many features, "
            "fa.coefficient_mass says where its weight sits.")
    if test is not None and test not in result.obs.columns:
        raise AnalysisError(
            f"no {test!r} column in obs to test against.\n"
            f"  obs has: {', '.join(map(str, result.obs.columns)) or '(nothing)'}")
    if test is not None and not pd.api.types.is_numeric_dtype(result.obs[test]):
        # associate() reads the outcome through to_numeric, so text becomes
        # missing and every row of the table would be NaN without a word.
        seen = list(dict.fromkeys(result.obs[test].dropna().astype(str)))[:4]
        raise AnalysisError(
            f"{test!r} is not numeric (values such as {', '.join(seen)}), and the "
            "association is an ordinary least squares fit.\n"
            "  Code it as a number first, 0/1 for a binary outcome, so that which "
            "level counts as 1 is your choice rather than an alphabetical one.")
    if data is None:
        raise AnalysisError("data= is required: a result holds scores, not the markers "
                            "they were computed from")
    df = _marker_frame(data, "data")
    absent_ids = result.scores.index.difference(df.index)
    if len(absent_ids):
        raise AnalysisError(
            f"{len(absent_ids)} sample(s) of the result are not in data=, e.g. "
            f"{', '.join(map(str, absent_ids[:3]))}; pass the data the result was "
            "scored from")
    df = df.loc[result.scores.index]
    if reference is None:
        raise AnalysisError(
            f"{clock} needs reference=, the reference population's markers.\n"
            + ("  PhenoAge's weights are fixed, so a marker is removed by holding it "
               "at a constant; the reference supplies that constant (its mean). Pass "
               "the population to compare against, or this cohort's own markers to "
               "hold each at the cohort mean."
               if formula == "phenoage" else
               "  Removing a marker means refitting on the remaining ones, which "
               f"needs the rows fit_{formula} was given."))
    ref = _marker_frame(reference, "reference")
    if isinstance(markers, str):
        markers = [markers]

    held: dict[str, float] = {}
    if formula == "phenoage":
        if sex_col is not None:
            raise AnalysisError(
                "sex_col= fits the reference by sex, which applies to the refitted "
                "clocks (KDM, HD). PhenoAge is not refitted and holds each marker at "
                "one reference mean; pass a reference of one sex to hold at that "
                "sex's mean.")
        panel = [m for m in clin.PHENOAGE_UNITS if m != "age"]
        chosen = list(dict.fromkeys(markers)) if markers is not None else panel
        unknown = [m for m in chosen if m not in panel]
        if unknown:
            raise AnalysisError(
                f"PhenoAge has no marker {', '.join(map(repr, unknown))}; its nine "
                f"are {', '.join(panel)}")
        lacking = [m for m in chosen if m not in ref.columns]
        if lacking:
            raise AnalysisError(
                f"the reference has no {', '.join(map(repr, lacking))} column, so "
                "there is no mean to hold it at")
        for m in chosen:
            v = pd.to_numeric(ref[m], errors="coerce").dropna().to_numpy(dtype=float)
            if v.size == 0:
                raise AnalysisError(f"the reference has no value of {m!r} to average")
            if m == "crp":
                if (v <= 0).any():
                    raise AnalysisError(
                        "the reference has CRP values at or below zero; PhenoAge "
                        "weights log(CRP), so its mean is taken on the log scale "
                        "and needs every value positive")
                held[m] = float(np.exp(np.mean(np.log(v))))
            else:
                held[m] = float(np.mean(v))

        def recompute(drop: str | None) -> pd.Series:
            x = df if drop is None else df.assign(**{drop: held[drop]})
            return clin.phenoage(x)

        how = "held at its mean in the reference (log scale for CRP)"
    else:
        if not markers:
            raise AnalysisError(
                f"{clock} has no fixed panel: pass markers=, the markers the "
                f"reference given to fa.score was fitted on (fit_{formula}(df, markers)).")
        chosen = list(dict.fromkeys(markers))
        minimum = 3 if formula == "hd" else 2
        if len(chosen) < minimum:
            raise AnalysisError(
                f"a {formula.upper()} panel of {len(chosen)} marker(s) cannot lose "
                f"one; it needs at least {minimum}"
                + (" (a covariance of one marker is a variance, and the distance "
                   "a z-score)" if formula == "hd" else ""))
        need = chosen + ([age_col] if formula == "kdm" else [])
        for name, frame in (("data", df), ("the reference", ref)):
            lacking = [m for m in need if m not in frame.columns]
            if lacking:
                raise AnalysisError(f"{name} has no {', '.join(map(repr, lacking))} column")

        if sex_col is None:
            key = pd.Series("all", index=df.index, dtype=object)
            ref_key = pd.Series("all", index=ref.index, dtype=object)
        else:
            for name, frame in (("data", df), ("the reference", ref)):
                if sex_col not in frame.columns:
                    raise AnalysisError(f"{name} has no {sex_col!r} column to fit by")
            key, ref_key = _strata(df, [sex_col]), _strata(ref, [sex_col])
            lacking = sorted(set(key.dropna()) - set(ref_key.dropna()))
            if lacking:
                raise AnalysisError(
                    f"the reference has no rows with {'; '.join(lacking)}. Levels are "
                    "compared as text ignoring case; recode one side first.")
        levels = list(dict.fromkeys(key.dropna()))

        def recompute(drop: str | None) -> pd.Series:
            ms = [m for m in chosen if m != drop]
            out = pd.Series(np.nan, index=df.index, name=formula)
            for lv in levels:
                r, s = ref[ref_key == lv], df[key == lv]
                if formula == "kdm":
                    out[s.index] = clin.kdm(s, clin.fit_kdm(r, ms, age_col=age_col),
                                            age_col=age_col)
                else:
                    out[s.index] = clin.hd(s, clin.fit_hd(r, ms))
            return out

        how = "refitted on the remaining markers"

    # The baseline must be the clock the result holds. Recomputing it from
    # what was passed and comparing is the only check that the reference and
    # markers are the ones it was scored with; without it a mismatched
    # reference would decompose a clock nobody reported.
    full = recompute(None)
    scored = pd.to_numeric(result.scores[clock], errors="coerce")
    fin_s, fin_f = scored.notna(), full.notna()
    gap = (scored - full).abs()[fin_s & fin_f]
    tol = 1e-6 * max(1.0, float(scored.abs().max()))
    if (fin_s != fin_f).any() or (len(gap) and float(gap.max()) > tol):
        where = (f"largest difference {float(gap.max()):.4g} at {gap.idxmax()}"
                 if len(gap) and float(gap.max()) > tol else
                 f"{int((fin_s != fin_f).sum())} sample(s) scored by one and not the other")
        raise AnalysisError(
            f"recomputing {clock} from data= and reference= does not reproduce the "
            f"scores in this result ({where}).\n"
            "  The decomposition would describe a different clock. Pass the data the "
            "result was scored from and the reference (and, for KDM and HD, the "
            "markers and sex_col) it was scored with.")

    reduced: dict[str, pd.Series] = {}
    for m in chosen:
        try:
            reduced[m] = recompute(m)
        except AnalysisError as exc:
            raise AnalysisError(f"{clock} without {m!r}: {exc}") from exc

    diffable = "difference" in c.legal_operations
    rows = []
    for name, s in [(FULL_PANEL, full), *reduced.items()]:
        ok = full.notna() & s.notna()
        d = (s - full)[ok]
        rows.append({
            "removed": name, "n": int(ok.sum()),
            "r_spearman": (float(full[ok].corr(s[ok], method="spearman"))
                           if ok.sum() > 2 else np.nan),
            "mean_change": float(d.mean()) if diffable and len(d) else np.nan,
            "mean_abs_change": float(d.abs().mean()) if diffable and len(d) else np.nan,
        })
    tab = pd.DataFrame(rows).set_index("removed")

    if test is not None:
        frame = pd.DataFrame({FULL_PANEL: full, **reduced})
        obs, covs = _numeric_covariates(result.obs, covariates)
        assoc = associate(SimpleNamespace(scores=frame, obs=obs), test,
                          covariates=covs).reindex(tab.index)
        # The partial correlation of the outcome with the score given the
        # covariates, from the OLS t by r = t / sqrt(t^2 + df), df the
        # residual degrees of freedom (intercept, score and each covariate
        # present). It is compared rather than beta because it does not
        # depend on the score's scale, and a clock without a marker has a
        # different spread from the clock with it.
        k = sum(cov in obs.columns for cov in covs)
        dof = assoc["n"] - 2 - k
        t = assoc["t"] if "t" in assoc.columns else pd.Series(np.nan, index=tab.index)
        tab["n_test"] = assoc["n"]
        tab["beta"] = assoc["beta"]
        tab["se"] = assoc["se"]
        tab["p"] = assoc["p"]
        tab["partial_r"] = t / np.sqrt(t ** 2 + dof)
        r0 = float(tab.at[FULL_PANEL, "partial_r"])
        tab["attenuation"] = (1.0 - tab["partial_r"] / r0
                              if np.isfinite(r0) and r0 != 0 else np.nan)

    rest = tab.drop(index=FULL_PANEL)
    rest = (rest.sort_values("attenuation", ascending=False) if test is not None
            else rest.sort_values("r_spearman"))
    tab = pd.concat([tab.loc[[FULL_PANEL]], rest])
    tab.attrs.update({
        "clock": clock, "formula": formula, "removal": how, "held_at": held,
        "test": test, "covariates": list(covariates) if test is not None else [],
        "sex_col": sex_col,
    })
    return tab
