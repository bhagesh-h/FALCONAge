"""Age acceleration in its three conventions, gated on each clock's scale type."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from ..core.errors import AnalysisError, IllegalOperationError
from ._common import _check_legal


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
