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

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from ..core.errors import AnalysisError, IllegalOperationError

__all__ = [
    "BenchmarkResult", "ConsensusReport", "PowerResult", "acceleration",
    "agreement", "associate", "consensus", "cox_hazard", "detectable_effect",
    "icc", "leave_one_marker_out", "pc_counterpart", "power", "run_benchmark",
]


#: Base clock to its high-reliability principal-component counterpart, where the
#: id does not follow the ``pc`` + id convention.
#:
#: This map exists because deriving the partner by string concatenation is
#: almost right. It gives pchorvath2013, pchannum, pcskinandblood, pcdnamtl and
#: pcgrimage correctly, and then silently misses the one that matters most:
#: DNAmPhenoAge's PC version is published as ``pcphenoage``, not
#: ``pcdnamphenoage``, and PhenoAge is the clock *When to Trust Epigenetic
#: Clocks* singles out as the case where the PC version changes the answer. A
#: check that skips its most important case reads exactly like a check that
#: passed.
#: The PC clocks Higgins-Chen et al. 2022 retrained, by the clock each one
#: replaces. Nothing else has a high-reliability version: deriving one by
#: prefixing "pc" named clocks that do not exist (pccellpopage) and asked the
#: verdict to wait for them.
PC_COUNTERPART = {
    "horvath2013": "pchorvath2013",
    "skinandblood": "pcskinandblood",
    "hannum": "pchannum",
    "dnamphenoage": "pcphenoage",
    "dnamtl": "pcdnamtl",
    "grimage": "pcgrimage",
}


def pc_counterpart(clock_id: str) -> str | None:
    """The high-reliability version of a clock, by id, or None if there is none.

    Naming only. It does not promise the counterpart is in the registry or that
    it can be scored, because on a default install none of them can be: the PC
    clocks are ``licensed``, and five of them are registered from the authors'
    own file with ``fa.registry.load().import_pc_clocks(path)``.
    """
    return PC_COUNTERPART.get(clock_id)


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
            + ("  This clock is in years and tracks age with a slope near one, "
               "but its origin is not fixed: its offset against chronological "
               "age moves by over a hundred years between cohorts. Subtracting "
               "chronological age from it measures the cohort. Use "
               "method='residual' or method='within_group', which fit inside "
               "the data you give them and are what the published analyses "
               "use.\n" if c.scale_type == "age_years_relative" else "")
            + ("  A log-hazard has no zero point on the age scale; use "
               "cox_hazard or rank it.\n" if c.scale_type == "mortality_log_hazard" else "")
        )


# ---------------------------------------------------------------------------
# age acceleration
# ---------------------------------------------------------------------------
def cell_composition(result, *, min_clocks: int = 2) -> pd.DataFrame:
    """Cell-type proportions estimated in this same run, as a covariate frame.

    Every clock in the result whose scale is ``proportion`` -- the
    reference-based deconvolution models -- one column each.

    WHY THIS IS WORTH A FUNCTION. Blood composition changes with age, and it
    changes with whatever else is happening to a person. A study of 10,000+
    blood samples found significant associations between immune cell
    composition and epigenetic age acceleration for every one of six widely
    used clocks (Aging Cell 2024;23:e14071), which means an unadjusted
    acceleration is measuring two things at once and reporting one number. The
    proportions needed to separate them are usually already sitting in the same
    result, computed by the deconvolution clocks; nothing connected the two.

    Returns an empty frame when the run had no deconvolution clocks, so callers
    can treat "no adjustment available" as data rather than as an exception.
    """
    cols = [c for c in result.scores.columns
            if result.registry.get(c).scale_type == "proportion"]
    if len(cols) < min_clocks:
        return pd.DataFrame(index=result.scores.index)
    return result.scores[cols].copy()


def _regress_out(y: pd.Series, design: pd.DataFrame, ok: pd.Series) -> pd.Series:
    """Residual of y on an intercept plus every column of design."""
    x = np.column_stack([np.ones(int(ok.sum())),
                         design.loc[ok].to_numpy(dtype=float)])
    beta, *_ = np.linalg.lstsq(x, y[ok].to_numpy(float), rcond=None)
    full = np.column_stack([np.ones(len(y)), design.to_numpy(dtype=float)])
    return y - pd.Series(full @ beta, index=y.index)


def acceleration(result, *, age_col: str = "age", method: str = "residual",
                 group: str | None = None, clocks: Sequence[str] | None = None,
                 adjust: str | Sequence[str] | None = None, reference=None,
                 match: Sequence[str] = ("age", "sex")) -> pd.DataFrame:
    """Age acceleration, in whichever convention you mean.

    Parameters
    ----------
    adjust
        Extra covariates to regress out alongside chronological age.

        ``"cell_composition"``
            Use the deconvolution clocks scored in this same run. An
            acceleration adjusted this way answers "is this person's blood
            aging faster", where the unadjusted version answers "is this
            person's blood aging faster **or** is its cell mix different",
            and reports both as one number.
        a sequence of column names
            Columns of ``result.obs``, for measured counts or anything else.

        Only available with ``method="residual"``: the absolute convention has
        no regression to add terms to, and ``within_group`` fits per stratum
        where a composition term would usually be rank-deficient.
    method
        ``"absolute"``
            ``predicted - chronological``. Interpretable in years, and
            confounded by the clock's own bias: a clock that over-predicts
            everyone by three years gives everyone three years of acceleration.
        ``"residual"``
            The residual from regressing predicted on chronological age. Centred
            at zero by construction, which removes that bias and also removes
            any real cohort-wide effect. The field's default.
        ``"both"``
            Absolute and residual side by side, two columns per clock named
            ``<clock>_absolute`` and ``<clock>_residual``. Suffixed rather than
            stacked, because the two disagree by several years and a reader who
            cannot tell which column is which is worse off than with one.
        ``"within_group"``
            Residual from a regression fitted separately within each level of
            ``group``. What the AA2 benchmark needs: it asks whether cases
            accelerate relative to *their own* controls, not relative to a line
            fitted through both.
        ``"reference"``
            Residual from a line fitted in another population, ``reference``,
            and applied unchanged to this one: ``score - (a + b * age)`` with
            ``a`` and ``b`` estimated on the reference only, separately within
            each level of the categorical ``match`` columns. It answers "how
            does this person compare with people of the same age and sex in
            the reference", which the within-cohort residual cannot, because
            that residual is centred on this cohort and so absorbs any effect
            the whole cohort shares.
    reference
        For ``method="reference"``: a scored
        :class:`~falconage.score.FalconResult` for the reference population,
        scored with the same fitted model (for KDM, the same fitted
        :class:`~falconage.models.clinical.KDMReference` passed to
        :func:`~falconage.score.score` for both). It must carry every clock
        asked for and every ``match`` column in its ``obs``.
    match
        For ``method="reference"``: the ``obs`` columns the comparison is
        matched on. ``age_col`` must be one of them; it is the regressor. Every
        other entry (usually ``"sex"``) is categorical, and the line is fitted
        separately within each of its levels. Levels are compared as text,
        ignoring case and a trailing ``.0``, so ``"F"`` matches ``"f"`` but
        not ``"female"`` or ``2``: recode one side first.

    Raises
    ------
    IllegalOperationError
        When a clock's scale does not admit the convention asked for.
        ``"absolute"``, ``"both"`` and ``"reference"`` need the scale to admit
        ``acceleration``, because each compares this cohort's scores with a
        zero point set somewhere else.
    AnalysisError
        With ``method="reference"``: no reference given, a clock or a
        ``match`` column absent from it, a level of a ``match`` column in this
        result that the reference lacks, fewer than three usable reference
        samples in a stratum, or a study age outside the ages the reference
        covers in that stratum (the range is reported; the line is never
        extrapolated).

    Notes
    -----
    Which one a paper used is often not stated, and they disagree by several
    years on the same data. The convention is recorded in the returned frame's
    ``method`` attribute so a downstream reader does not have to guess.

    WHY A REFERENCE POPULATION. A clock fitted by regression regresses toward
    the mean age of its training population: it over-predicts the young and
    under-predicts the old, and a KDM projected from a reference inherits the
    same pull. In an older cohort ``predicted - chronological`` is then
    negative for most people whatever their health, and the within-cohort
    residual removes that bias only by centring the cohort on itself. Brain-age
    work documents the same bias (Smith et al. 2019, NeuroImage 200:528-539,
    doi:10.1016/j.neuroimage.2019.06.017) and corrects it with a regression of
    the prediction on age whose parameters can be estimated in a training or
    reference sample and applied unchanged to new data (de Lange and Cole
    2020, NeuroImage: Clinical 26:102229, doi:10.1016/j.nicl.2020.102229).
    ``"reference"`` is that regression, fitted in the reference and taken as a
    residual so that it is on the same footing as ``"residual"`` above.

    With ``method="reference"`` the frame's ``attrs["reference_fit"]`` lists,
    per clock and stratum, the reference sample size, its age range, the
    fitted intercept and slope and the residual SD, so the comparison can be
    audited without the reference in hand.
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
    # The conventions need different permissions, and at first both asked for
    # the same one. `absolute` is predicted minus chronological, so it needs the
    # clock's zero to mean something; `residual` fits a line inside the data at
    # hand and therefore does not. LEGAL_OPS has listed the two separately since
    # v1.0 -- nothing read the distinction, which is why a clock whose intercept
    # moves 162 years between cohorts could still be handed to `absolute`.
    # `reference` asks for the stricter of the two. Its line is fitted in
    # another dataset, so it carries that dataset's origin into this one, which
    # is exactly what a scale without a fixed origin (age_years_relative) or
    # without an age axis at all cannot support.
    needed = "acceleration" if method in ("absolute", "both", "reference") else "residual"
    if clocks:
        cols = list(clocks)
        for cid in cols:
            _check_legal(result.registry, cid, needed)
    else:
        cols = [c for c in result.scores.columns
                if needed in result.registry.get(c).legal_operations]
        if not cols:
            raise IllegalOperationError(
                f"no clock in this result admits {needed!r}, so age acceleration "
                f"by method={method!r} is undefined for all of them.\n"
                "  Scales present: "
                + ", ".join(sorted({result.registry.get(c).scale_type
                                    for c in result.scores.columns})))

    # Build the extra design columns once, and refuse clearly rather than
    # quietly ignoring `adjust=` on a method that cannot honour it.
    extra = pd.DataFrame(index=result.scores.index)
    if adjust is not None:
        if method != "residual":
            raise AnalysisError(
                f"adjust= needs method='residual'; got {method!r}.\n"
                "  'absolute' is a subtraction with no regression to extend, "
                "'within_group' fits inside each stratum where a composition "
                "term is usually rank-deficient, and 'reference' fits in another "
                "population, which would need the same covariates measured the "
                "same way there.")
        if adjust == "cell_composition":
            extra = cell_composition(result)
            if extra.empty:
                raise AnalysisError(
                    "adjust='cell_composition' needs deconvolution clocks in the "
                    "same result, and this one has none.\n"
                    "  Score them alongside: "
                    'score(data, clocks="compatible") includes them when the '
                    "platform supports it, or name them explicitly.\n"
                    "  Measured cell counts work too: adjust=['cd8t', 'mono', ...] "
                    "naming columns of obs.")
        else:
            names = [adjust] if isinstance(adjust, str) else list(adjust)
            missing = [n for n in names if n not in result.obs.columns]
            if missing:
                raise AnalysisError(
                    f"adjust= names {', '.join(missing)}, not in obs.\n"
                    f"  obs has: {', '.join(map(str, result.obs.columns)) or '(nothing)'}")
            extra = result.obs[names].apply(pd.to_numeric, errors="coerce")

        # A constant column carries no information and makes the design
        # singular; dropping it silently is better than a LinAlgError, but only
        # if it is said out loud.
        constant = [c for c in extra.columns if extra[c].nunique(dropna=True) < 2]
        if constant:
            extra = extra.drop(columns=constant)

    if method == "reference":
        # Branches before the loop below because its sample-size floor is on
        # this cohort, and here the line is fitted elsewhere: one study sample
        # can be compared with a reference as well as a thousand can.
        return _reference_acceleration(result, reference, cols, age, age_col, match)

    out: dict[str, pd.Series] = {}
    for cid in cols:
        y = result.scores[cid]
        ok = age.notna() & y.notna()
        for c in extra.columns:
            ok &= extra[c].notna()
        if ok.sum() < 3 + extra.shape[1]:
            raise AnalysisError(
                f"{cid}: {int(ok.sum())} usable sample(s) for "
                f"{1 + extra.shape[1]} predictor(s); need at least "
                f"{3 + extra.shape[1]}")

        def _resid() -> pd.Series:
            if extra.shape[1]:
                design = pd.concat([age.rename("__age"), extra], axis=1)
                return _regress_out(y, design, ok)
            return _residual(y, age, ok)

        if method == "absolute":
            out[cid] = y - age
        elif method == "residual":
            out[cid] = _resid()
        elif method == "both":
            # Two columns per clock rather than two calls. The two conventions
            # disagree by several years on the same data and papers often do
            # not say which they used, so having them side by side is the
            # honest way to read a result -- and suffixed names mean a reader
            # cannot mistake one column for the other.
            out[f"{cid}_absolute"] = y - age
            out[f"{cid}_residual"] = _resid()
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
            raise AnalysisError(
                "method must be 'absolute', 'residual', 'both', 'within_group' or "
                f"'reference'; got {method!r}")

    df = pd.DataFrame(out, index=result.scores.index)
    df.attrs["method"] = method
    # Which convention AND which adjustment. An acceleration adjusted for cell
    # composition is a different quantity from one that is not, and a frame
    # that does not say which it is gets compared with the other one.
    df.attrs["adjusted_for"] = list(extra.columns)
    return df


def _residual(y: pd.Series, age: pd.Series, ok: pd.Series) -> pd.Series:
    slope, intercept = np.polyfit(age[ok].to_numpy(float), y[ok].to_numpy(float), 1)
    return y - (slope * age + intercept)


def _level(v) -> str | None:
    """A categorical value as comparable text: 'F', 'f' and ' f ' agree, 2 and 2.0 agree."""
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return None
    s = str(v).strip().lower().removesuffix(".0")
    return s if s and s not in ("nan", "none", "<na>") else None


def _strata(obs: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    """One label per sample naming its stratum, or None where any column is missing."""
    if not cols:
        return pd.Series("all", index=obs.index, dtype=object)
    columns = [[_level(v) for v in obs[c].tolist()] for c in cols]
    labels = [None if any(v is None for v in vals)
              else ", ".join(f"{c}={v}" for c, v in zip(cols, vals))
              for vals in zip(*columns)]
    return pd.Series(labels, index=obs.index, dtype=object)


def _reference_acceleration(result, reference, cols: list[str], age: pd.Series,
                            age_col: str, match) -> pd.DataFrame:
    """``method="reference"`` of :func:`acceleration`; see its docstring."""
    if reference is None or not hasattr(reference, "scores") or not hasattr(reference, "obs"):
        raise AnalysisError(
            "method='reference' needs reference=, a scored FalconResult for the "
            "reference population.\n"
            "  Score the reference with the same clocks and, for KDM, the same "
            "fitted reference object:\n"
            "  ref_res = fa.score(reference_data, clocks=[...], reference=fitted)")
    match = [match] if isinstance(match, str) else list(match)
    if age_col not in match:
        raise AnalysisError(
            f"match={match} does not include {age_col!r}.\n"
            "  Age is the regressor of the reference line, so it is always matched; "
            f"pass match=({age_col!r}, ...). Without it the comparison is the "
            "absolute convention measured against someone else's mean.")
    strata = [m for m in match if m != age_col]

    for name, obs in (("this result", result.obs), ("the reference", reference.obs)):
        missing = [m for m in match if m not in obs.columns]
        if missing:
            raise AnalysisError(
                f"{name} has no {', '.join(repr(m) for m in missing)} column in obs, "
                f"and method='reference' matches on {match}.\n"
                f"  obs has: {', '.join(map(str, obs.columns)) or '(nothing)'}")
    absent = [c for c in cols if c not in reference.scores.columns]
    if absent:
        have = [c for c in cols if c in reference.scores.columns]
        raise AnalysisError(
            f"the reference was not scored on {', '.join(absent)}.\n"
            "  A reference comparison needs the same clock, fitted the same way, in "
            "both populations. Score the reference on it, or pass clocks="
            f"{have} to compare only the clocks both carry.")

    # Two runs that recorded different coefficient digests for a clock did not
    # score it with the same model, and their difference would be the model's.
    for cid in cols:
        a = (getattr(result.manifest, "weights", {}) or {}).get(cid, {}).get("sha256")
        b = (getattr(reference.manifest, "weights", {}) or {}).get(cid, {}).get("sha256")
        if a and b and a != b:
            raise AnalysisError(
                f"{cid}: this result and the reference were scored with different "
                f"coefficients (sha256 {a[:12]} against {b[:12]}), so their "
                "difference would measure the two models rather than the two "
                "populations.")

    ref_age = pd.to_numeric(reference.obs[age_col], errors="coerce")
    key = _strata(result.obs, strata)
    ref_key = _strata(reference.obs, strata)

    out: dict[str, pd.Series] = {}
    fits: list[dict] = []
    for cid in cols:
        y = pd.to_numeric(result.scores[cid], errors="coerce")
        yr = pd.to_numeric(reference.scores[cid], errors="coerce").reindex(reference.obs.index)
        ok = age.notna() & y.notna() & key.notna()
        ok_ref = ref_age.notna() & yr.notna() & ref_key.notna()

        levels = list(dict.fromkeys(key[ok]))
        have = set(ref_key[ok_ref])
        lacking = [lv for lv in levels if lv not in have]
        if lacking:
            raise AnalysisError(
                f"{cid}: the reference has no scored sample with "
                f"{'; '.join(lacking)}.\n"
                f"  Reference strata: {'; '.join(sorted(have)) or '(none)'}.\n"
                "  Levels are compared as text ignoring case, so a reference coded "
                "1/2 and a study coded M/F do not meet: recode one side first.")

        # One line per stratum rather than one line with sex as a covariate.
        # A covariate forces the same age slope on both sexes and moves only
        # the intercept; fitting separately lets the slope differ, which costs
        # nothing when it does not. It is the convention of the clinical clocks
        # this is mostly used with: BioAge fits KDM separately by sex (Kwon and
        # Belsky 2021, GeroScience 43:2795-2808), and validate_panel and
        # kdm_bioage in falconage.models.clinical follow it. It is also how 'within_group'
        # above treats a stratum, so the two conventions differ only in whose
        # data the line comes from. The cost is a reference large enough in
        # every stratum, which the n_reference column of attrs makes visible.
        res = pd.Series(np.nan, index=y.index)
        for lv in levels:
            rsel = ok_ref & (ref_key == lv)
            ssel = ok & (key == lv)
            n_ref = int(rsel.sum())
            if n_ref < 3:
                raise AnalysisError(
                    f"{cid}: {n_ref} usable reference sample(s) with {lv}; a line "
                    "needs at least 3.")
            lo, hi = float(ref_age[rsel].min()), float(ref_age[rsel].max())
            s_age = age[ssel]
            outside = (s_age < lo) | (s_age > hi)
            if outside.any():
                raise AnalysisError(
                    f"{cid}: {int(outside.sum())} sample(s) with {lv} are aged "
                    f"{float(s_age[outside].min()):.4g} to {float(s_age[outside].max()):.4g}, "
                    f"outside the reference's {lo:.4g} to {hi:.4g} for the same "
                    "stratum.\n"
                    "  A line fitted in the reference describes only the ages it saw, "
                    "and this refuses rather than extrapolate it. Restrict the study "
                    "to that range, or use a reference that covers it.")
            # np.polyfit, as _residual fits the within-cohort line, so that
            # 'reference' and 'residual' differ only in where the line was fitted.
            slope, intercept = np.polyfit(ref_age[rsel].to_numpy(float),
                                          yr[rsel].to_numpy(float), 1)
            fitted_ref = slope * ref_age[rsel] + intercept
            res[ssel] = y[ssel] - (slope * age[ssel] + intercept)
            fits.append({
                "clock": cid, "stratum": lv, "n_reference": n_ref,
                "n_study": int(ssel.sum()), "age_min": lo, "age_max": hi,
                "intercept": float(intercept), "slope": float(slope),
                # ddof=2: two parameters were estimated from these points.
                "resid_sd": float(np.std(yr[rsel] - fitted_ref, ddof=2)),
            })
        out[cid] = res

    df = pd.DataFrame(out, index=result.scores.index)
    df.attrs["method"] = "reference"
    df.attrs["adjusted_for"] = []
    df.attrs["match"] = match
    # Records rather than a DataFrame: pandas compares attrs when frames are
    # concatenated, and a DataFrame inside attrs makes that comparison raise.
    df.attrs["reference_fit"] = fits
    return df


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
