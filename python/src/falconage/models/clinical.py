"""Clinical chemistry clocks: PhenoAge, Klemera-Doubal, homeostatic dysregulation.

These three have no coefficient file to download, for three different reasons,
and the difference matters when reading a result.

**PhenoAge** is a closed form. Levine 2018 prints ten coefficients and the
Gompertz calibration constants, so the whole model is in the paper and there is
nothing to fetch or trace. What there *is* to get wrong is units -- see the
warning below, and :mod:`falconage.core.units`.

**Klemera-Doubal** has no fixed coefficients at all. Each biomarker is regressed
on chronological age in a reference cohort and the panel is inverted to a
maximum-likelihood age. Score the same person against NHANES III and against a
hospital cohort and you get two different numbers, both correct; the manifest
records which reference was used.

**Homeostatic dysregulation** is a Mahalanobis distance to a reference centre.
The reference is part of the definition rather than a parameter: substituting
the sample's own distribution turns "how far from healthy is this person" into
"how unusual is this person within this batch", which is a different question
with the same units.

THESE THREE ARE CPU-ONLY, DELIBERATELY. A methylation clock reduces thousands
of probes; a clinical clock reduces nine markers. PhenoAge sums ten terms, KDM
fits one univariate regression per marker, HD inverts a 9x9 covariance. That is
less arithmetic than a single CUDA kernel launch costs to dispatch, so a device
implementation would be slower, and it would pull torch into the one modality
that otherwise needs nothing beyond numpy. :class:`ClinicalClock` therefore
declares ``CPU_ONLY`` rather than accepting a device and ignoring it, and the
manifest records ``cpu`` for these clocks even in a run launched with
``device="cuda"``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..core.errors import AnalysisError, DataError

# ---------------------------------------------------------------------------
# Levine 2018 clinical PhenoAge
# ---------------------------------------------------------------------------
#
# Coefficients as published, in the paper's own units. FALCONAge converts the
# input to these before applying them; it does not restate them in some other
# unit, so the numbers below can be checked line by line against Table 1 (and
# Supplementary Table S1, which adds the constant). Alkaline phosphatase is
# printed there as 0.0019. The 0.00188 that circulates comes from the equation
# in Liu et al. 2018 (PLoS Med 15:e1002718), which also prints glucose as 0.0195,
# RDW as 0.3356 and the final divisor as 0.09165, all three typos.
#
#   albumin              g/L
#   creatinine           umol/L
#   glucose              mmol/L
#   log(CRP)             mg/dL     <- note: dL, unlike every other marker here
#   lymphocyte percent   %
#   mean cell volume     fL
#   red cell dist width  %
#   alkaline phosphatase U/L
#   white blood cells    10^3/uL
#   chronological age    years
PHENOAGE_COEF: dict[str, float] = {
    "albumin": -0.0336,
    "creatinine": 0.0095,
    "glucose": 0.1953,
    "log_crp": 0.0954,
    "lymphocyte_percent": -0.0120,
    "mean_cell_volume": 0.0268,
    "red_cell_distribution_width": 0.3306,
    "alkaline_phosphatase": 0.0019,
    "white_blood_cell_count": 0.0554,
    "age": 0.0804,
}
PHENOAGE_INTERCEPT = -19.9067

#: Gompertz calibration. gamma is the shape; t=120 months is the horizon the
#: mortality score is expressed at; the rest fall out of the published inversion
#: (Supplement 1: gamma = 0.0076927, age = 141.50225 + ln(-0.00553 ln(1 - m)) / 0.090165).
PHENOAGE_GAMMA = 0.0076927
PHENOAGE_T = 120.0
PHENOAGE_INVERSION = 0.00553

#: The same model at the precision BioAge carries for ``phenoage0`` (Kwon and
#: Belsky, GeroScience 2021; ``R/phenoage_calc.R``). Every weight rounds to the
#: Table 1 value, so these are the fit the table was printed from. On NHANES III
#: they give PhenoAge a mean 0.074 years below the table values (n = 8,924).
PHENOAGE_BIOAGE: dict[str, object] = {
    "coef": {
        "albumin": -0.03359355,
        "creatinine": 0.009506491,
        "glucose": 0.1953192,
        "log_crp": 0.09536762,
        "lymphocyte_percent": -0.01199984,
        "mean_cell_volume": 0.02676401,
        "red_cell_distribution_width": 0.3306156,
        "alkaline_phosphatase": 0.001868778,
        "white_blood_cell_count": 0.05542406,
        "age": 0.08035356,
    },
    "intercept": -19.90667,
    "gamma": 0.007692696,
    "inversion": 0.0055305,
}

#: The units PhenoAge's coefficients expect, marker by marker. Anything else is
#: converted on the way in; anything unconvertible is an error, never a guess.
PHENOAGE_UNITS: dict[str, str] = {
    "albumin": "g/L",
    "creatinine": "umol/L",
    "glucose": "mmol/L",
    "crp": "mg/dL",
    "lymphocyte_percent": "%",
    "mean_cell_volume": "fL",
    "red_cell_distribution_width": "%",
    "alkaline_phosphatase": "U/L",
    "white_blood_cell_count": "10^3/uL",
    "age": "years",
}


def phenoage(df: pd.DataFrame, crp_transform: str = "log",
             coefficients: str = "levine2018") -> pd.Series:
    """Clinical Phenotypic Age in years.

    Parameters
    ----------
    df
        One row per sample, columns named as in :data:`PHENOAGE_UNITS`, already
        converted to those units. :func:`falconage.preprocess.clinical.prepare`
        does the conversion; calling this directly means asserting it is done.

    Notes
    -----
    CRP enters as ``log(crp)`` with crp in **mg/dL**, which is the one place the
    paper departs from SI and the single most common transcription error in
    reimplementations -- a factor of ten in CRP moves PhenoAge by about 0.22
    years per log unit, small enough to look plausible and large enough to
    matter across a cohort.

    ``crp_transform="log1p"`` uses ``log(1 + crp)``, BioAge's ``lncrp``. It is
    not the paper's transform: on NHANES III it raises PhenoAge by a mean 1.57
    years (SD 0.45, n = 8,924), more at low CRP, because
    ln(1 + CRP) - ln(CRP) = ln(1 + 1/CRP). Use it only to compare with BioAge
    output.

    ``coefficients="levine2018"`` applies Table 1 of the paper as printed.
    ``"bioage"`` applies :data:`PHENOAGE_BIOAGE`, the same fit at full
    precision. With both BioAge options the result is BioAge's ``phenoage0``:
    on its 8,924 NHANES III rows the largest difference is below 0.00001 years.
    """
    missing = [m for m in PHENOAGE_UNITS if m not in df.columns]
    if missing:
        raise DataError(
            "clinical PhenoAge needs all ten markers; missing: " + ", ".join(missing)
            + "\n  It is a closed form with no imputation step -- the published "
              "model has no term for an absent marker, and substituting a cohort "
              "mean invents one."
        )

    if crp_transform not in ("log", "log1p"):
        raise DataError(f"crp_transform must be 'log' (Levine 2018) or 'log1p' (BioAge), "
                        f"not {crp_transform!r}")
    if coefficients == "levine2018":
        coef, b0 = PHENOAGE_COEF, PHENOAGE_INTERCEPT
        gamma, inversion = PHENOAGE_GAMMA, PHENOAGE_INVERSION
    elif coefficients == "bioage":
        coef, b0 = PHENOAGE_BIOAGE["coef"], PHENOAGE_BIOAGE["intercept"]
        gamma, inversion = PHENOAGE_BIOAGE["gamma"], PHENOAGE_BIOAGE["inversion"]
    else:
        raise DataError(f"coefficients must be 'levine2018' (Table 1 as printed) or "
                        f"'bioage' (full precision), not {coefficients!r}")
    x = df.copy()
    crp = np.asarray(x["crp"], dtype=np.float64)
    if crp_transform == "log1p":
        x["log_crp"] = np.log1p(crp)
    else:
        if np.nanmin(crp) <= 0:
            raise DataError(
                "CRP must be strictly positive: PhenoAge takes its logarithm. "
                "Values reported as 0 are usually below the assay's detection limit; "
                "substitute the limit itself rather than zero."
            )
        x["log_crp"] = np.log(crp)

    xb = np.full(len(x), b0, dtype=np.float64)
    for marker, beta in coef.items():
        xb += beta * np.asarray(x[marker], dtype=np.float64)

    # Mortality score at 120 months under the Gompertz hazard, then inverted
    # back onto the age scale.
    mortality = 1.0 - np.exp(-np.exp(xb) * (np.exp(gamma * PHENOAGE_T) - 1.0) / gamma)
    mortality = np.clip(mortality, 1e-12, 1.0 - 1e-12)
    return pd.Series(
        141.50225 + np.log(-inversion * np.log(1.0 - mortality)) / 0.090165,
        index=df.index, name="phenoage",
    )


# ---------------------------------------------------------------------------
# Klemera-Doubal
# ---------------------------------------------------------------------------
@dataclass
class KDMReference:
    """Per-biomarker regressions on chronological age, fitted on a reference.

    Follows Klemera and Doubal 2006 as implemented in the BioAge R package
    (Kwon and Belsky 2021, ``R/kdm_calc.R``), including the ``s_R``
    correction. Every symbol below is theirs:

    ``k``, ``q``, ``s``
        slope, intercept and residual standard deviation of each biomarker
        regressed on chronological age.
    ``r``
        the absolute correlation of each biomarker with age, which for a
        one-predictor regression is exactly the square root of its R-squared.
        BioAge stores R-squared and takes ``sqrt(r)``; this stores the root, so
        ``r_char`` uses it directly. Taking the root a second time was the
        FALCONAge 1.0.0 bug recorded in the CHANGELOG.
    ``r_char``
        the characteristic correlation, a ``|k/s|``-weighted mean of ``r``.
    ``s_r``
        the variance the estimator would have if biological and chronological
        age were the same thing. Subtracting it from the observed variance is
        what stops KDM collapsing onto chronological age when the biomarkers
        carry little information -- omit it and every KDM paper's headline
        finding becomes an artefact of the age range.
    ``s_ba2``
        the variance of biological age around chronological age, estimated on
        the **reference** at fit time and reused when scoring another cohort,
        which is how BioAge projects a fit into new data.
    """

    markers: list[str]
    k: np.ndarray
    q: np.ndarray
    s: np.ndarray
    r: np.ndarray
    r_char: float
    s_r: float
    n_reference: int
    age_range: tuple[float, float]
    #: Estimated on the reference by :func:`fit_kdm`. ``None`` only on a
    #: hand-built reference, in which case :func:`kdm` estimates it from the
    #: scored cohort instead and says so in its docstring.
    s_ba2: float | None = None
    #: Rows each marker's regression was fitted on. Markers are regressed one
    #: at a time on the rows where that marker is present, so these differ.
    n_per_marker: np.ndarray | None = None
    #: Each marker's reference distribution; see :func:`marker_ranges`.
    ranges: pd.DataFrame | None = None

    def describe(self) -> pd.DataFrame:
        return pd.DataFrame({"slope": self.k, "intercept": self.q,
                             "resid_sd": self.s, "cor_with_age": self.r},
                            index=self.markers)


def fit_kdm(reference: pd.DataFrame, markers: list[str], age_col: str = "age",
            s_ba2: float | None = None) -> KDMReference:
    """Fit the KDM reference regressions, as BioAge's ``kdm_calc`` does.

    Each step follows BioAge (Kwon and Belsky, GeroScience 2021;43:2795-2808),
    and together they reproduce the ``kdm0`` column BioAge ships with its NHANES
    III extract to 0.0007 years mean absolute difference (maximum 0.007 years,
    n = 9,583), where FALCONAge 1.0.0 was 1.77 years out:

    1. Each biomarker is regressed on age on the rows where **that biomarker**
       is present (``svyglm`` per marker, missing rows dropped per model), not
       on the rows where every biomarker is present.
    2. The residual SD is ``sd(residuals)``, divisor n - 1, as BioAge's
       ``get_effs`` takes it for a ``glm`` fit.
    3. ``r_char`` weights the absolute correlation, the square root of R-squared.
    4. ``s_R`` uses the age range of the whole reference.
    5. ``s_BA^2`` is the variance of ``BA_E - CA`` on the reference rows with
       every biomarker present, minus ``s_R``, and is kept with the fit.

    ``s_ba2`` overrides step 5. Pass a value to put several fits on one scale.
    """
    ref_all = reference[reference[age_col].notna()]
    age_all = ref_all[age_col].to_numpy(dtype=np.float64)
    k, q, s, r, n_used = [], [], [], [], []
    degenerate = []
    for m in markers:
        present = ref_all[m].notna().to_numpy()
        if int(present.sum()) < 30:
            raise AnalysisError(
                f"KDM reference has {int(present.sum())} rows with {m!r} and an age; "
                "fewer than 30 makes the per-marker regression meaningless. Supply a "
                "larger reference cohort or leave the marker out."
            )
        age = age_all[present]
        y = ref_all[m].to_numpy(dtype=np.float64)[present]
        slope, intercept = np.polyfit(age, y, 1)
        resid = y - (slope * age + intercept)
        sd = float(np.std(resid, ddof=1))
        # A marker with no residual spread is not a perfect predictor, it is a
        # column that does not vary -- a unit conversion that collapsed it, a
        # single value carried down a spreadsheet, a lab that reported one
        # figure for the whole cohort. Every KDM term divides by this, so one
        # such marker turns k/s into an infinity, r into NaN through
        # corrcoef of a constant, r_char into NaN and s_r into inf. The
        # arithmetic then continues through nansum and returns a plausible
        # number computed from a poisoned reference, which is the worst
        # possible outcome and exactly what this package exists not to do.
        #
        # The comparison is relative, not `sd <= 0`. A genuinely constant
        # column does not give a residual standard deviation of exactly zero:
        # polyfit's least-squares solve leaves rounding noise around 1e-15, so
        # an exact test passes the very case it exists to catch. Scaling by the
        # column's own magnitude also keeps the test meaningful for a marker
        # measured in millions and one measured in tenths.
        scale = max(float(np.nanstd(y)), abs(float(np.nanmean(y))), 1.0)
        if not np.isfinite(sd) or sd <= 1e-10 * scale:
            degenerate.append(m)
        k.append(slope)
        q.append(intercept)
        s.append(sd)
        # |correlation| is the square root of this regression's R-squared, which
        # is the quantity BioAge's r1 = |k/s| * sqrt(r.squared) weights.
        r.append(abs(float(np.corrcoef(age, y)[0, 1])))
        n_used.append(int(present.sum()))

    if degenerate:
        raise AnalysisError(
            "KDM cannot use a marker that does not vary: "
            + ", ".join(f"{m!r}" for m in degenerate)
            + "\n  Each has zero residual spread across the reference rows that "
            "carry it, so its contribution to the estimate is a division by zero.\n"
            "  This is nearly always a data problem rather than a biological one "
            "-- a unit conversion that\n  collapsed the column, or one value "
            "filled down. Check the column, then either fix it or\n  leave it out "
            "of `markers=`; KDM is defined for any panel size."
        )

    k, q, s, r = (np.asarray(v, dtype=np.float64) for v in (k, q, s, r))

    ks = np.abs(k / s)
    r_char = float(np.sum(ks * r) / np.sum(ks))
    lo, hi = float(age_all.min()), float(age_all.max())
    s_r = float(((1 - r_char**2) / r_char**2) * ((hi - lo) ** 2 / (12 * len(markers)))) \
        if r_char > 0 else np.inf

    complete = ref_all.dropna(subset=list(markers))
    if s_ba2 is None:
        if len(complete) < 30:
            raise AnalysisError(
                f"KDM reference has {len(complete)} rows with every marker present; "
                "fewer than 30 cannot estimate s_BA^2, the variance that anchors "
                "biological age to chronological age. Supply a larger reference, "
                "fewer markers, or a fixed s_ba2= from another fit."
            )
        s_ba2 = _s_ba2(complete[list(markers)].to_numpy(dtype=np.float64),
                       complete[age_col].to_numpy(dtype=np.float64), q, k, s, s_r)

    return KDMReference(list(markers), k, q, s, r, r_char, s_r, len(complete), (lo, hi),
                        float(s_ba2), np.asarray(n_used), marker_ranges(ref_all, markers))


def _s_ba2(x: np.ndarray, age: np.ndarray, q, k, s, s_r: float) -> float:
    """``s_BA^2 = var(BA_E - CA) - s_R`` on complete rows.

    The variance is the population form, ``mean((d - mean(d))^2)``, which is
    BioAge's ``s2 = mean(t1)``.
    """
    ba_e = np.sum((x - q) * k / s**2, axis=1) / float(np.sum((k / s) ** 2))
    d = ba_e - age
    return float(np.mean((d - d.mean()) ** 2) - s_r)


def kdm(df: pd.DataFrame, ref: KDMReference, age_col: str = "age",
        max_missing: int | None = 2) -> pd.Series:
    """Klemera-Doubal biological age in years.

    Parameters
    ----------
    max_missing
        A sample missing more than this many of the reference's markers scores
        NaN. Two is BioAge's rule (``kdm = ifelse(BA_nmiss > 2, NA, kdm)``).
        ``None`` scores any sample with at least one marker, which is not a
        published behaviour and should be justified wherever it is used.

    Notes
    -----
    For a sample missing some markers, the numerator sums over the markers it
    has while the denominator sums over all of them. That is BioAge's formula
    and it pulls such a sample toward chronological age; ``max_missing`` is
    what keeps the pull small.

    ``s_BA^2`` comes from the reference. A hand-built reference without one
    falls back to estimating it on this cohort's complete rows, which is the
    right answer only when the reference and the scored cohort are the same
    people.
    """
    missing = [m for m in ref.markers if m not in df.columns]
    if missing:
        raise DataError("KDM needs the reference's markers; missing: " + ", ".join(missing))
    x = df[ref.markers].to_numpy(dtype=np.float64)
    age = df[age_col].to_numpy(dtype=np.float64)

    num = np.nansum((x - ref.q) * ref.k / ref.s**2, axis=1)
    den = float(np.sum((ref.k / ref.s) ** 2))
    n_obs = np.isfinite(x).sum(axis=1)
    n_miss = len(ref.markers) - n_obs
    drop = n_obs == 0 if max_missing is None else n_miss > max_missing

    s_ba2 = ref.s_ba2
    if s_ba2 is None:
        full = n_miss == 0
        s_ba2 = _s_ba2(x[full], age[full], ref.q, ref.k, ref.s, ref.s_r) \
            if full.sum() >= 2 else np.nan

    if not np.isfinite(s_ba2) or s_ba2 <= 0:
        # The biomarkers carry no information beyond age. Klemera and Doubal's
        # correction is larger than the observed spread, and the honest answer
        # is the biomarker estimate itself rather than a division by a negative
        # variance that silently flips the sign of the age term. BA_E is
        # rescaled for absent markers exactly as BioAge's BA_e is.
        out = (num / den) * (len(ref.markers) / np.maximum(n_obs, 1))
    else:
        out = (num + age / s_ba2) / (den + 1.0 / s_ba2)
    return pd.Series(np.where(drop, np.nan, out), index=df.index, name="kdm")


# ---------------------------------------------------------------------------
# Is a reduced panel still the same clock?
# ---------------------------------------------------------------------------
def validate_panel(reference: pd.DataFrame, full: list[str], reduced: list[str], *,
                   age_col: str = "age", sex_col: str | None = None,
                   time_col: str | None = None, event_col: str | None = None) -> pd.DataFrame:
    """Compare a reduced KDM panel with the full one on a reference cohort.

    A cohort that measured only some of a clock's biomarkers can still use KDM,
    because KDM is defined for any panel, but the reduced clock is a different
    clock until shown otherwise. This fits both panels on the same reference
    rows (every biomarker of the full panel present, fitted by sex when
    ``sex_col`` is given, as BioAge fits), scores those rows, and reports how
    closely the two agree.

    With ``time_col`` and ``event_col`` it also reports each panel's mortality
    hazard ratio in BioAge's ``table_surv`` specification: KDM advance
    (``kdm - age``) standardised within sex, in a Cox model with age and sex as
    covariates (age alone without ``sex_col``), Breslow ties, as a hazard ratio
    per standard deviation with a 95% interval.

    Returns one row per panel. ``attrs["agreement"]`` holds the correlation of
    the two biological ages, of the two advances, and their mean absolute
    difference in years.
    """
    missing = [m for m in reduced if m not in full]
    if missing:
        raise AnalysisError(f"the reduced panel names markers the full one does not: {missing}")
    need = [age_col, *full] + ([sex_col] if sex_col else [])
    rows = reference.dropna(subset=need)
    groups = [rows] if sex_col is None else [g for _, g in rows.groupby(sex_col)]

    scored = {}
    for name, markers in (("full", list(full)), ("reduced", list(reduced))):
        parts = []
        for g in groups:
            ref = fit_kdm(g, markers, age_col=age_col)
            k = kdm(g, ref, age_col=age_col, max_missing=0)
            adv = k - g[age_col]
            parts.append(pd.DataFrame({"kdm": k, "advance": adv,
                                       "advance_z": (adv - adv.mean()) / adv.std(ddof=1)}))
        scored[name] = pd.concat(parts).loc[rows.index]

    out = []
    for name, markers in (("full", list(full)), ("reduced", list(reduced))):
        s = scored[name]
        row = {"panel": name, "n_markers": len(markers), "n": int(len(s)),
               "r_with_age": float(np.corrcoef(s["kdm"], rows[age_col])[0, 1])}
        if time_col and event_col:
            from ..analysis import _cox_breslow

            cov = [s["advance_z"].to_numpy(), rows[age_col].to_numpy(dtype=np.float64)]
            if sex_col:
                codes = pd.factorize(rows[sex_col])[0].astype(np.float64)
                cov.append(codes)
            ok = rows[time_col].notna() & rows[event_col].notna()
            X = np.column_stack(cov)[ok.to_numpy()]
            b, v = _cox_breslow(X, rows.loc[ok, time_col].to_numpy(dtype=np.float64),
                                rows.loc[ok, event_col].to_numpy(dtype=np.float64))
            se = float(np.sqrt(v[0, 0]))
            row.update({"n_survival": int(ok.sum()),
                        "events": int(rows.loc[ok, event_col].sum()),
                        "hr_per_sd": float(np.exp(b[0])),
                        "hr_lo": float(np.exp(b[0] - 1.96 * se)),
                        "hr_hi": float(np.exp(b[0] + 1.96 * se))})
        out.append(row)

    table = pd.DataFrame(out).set_index("panel")
    f, r = scored["full"], scored["reduced"]
    table.attrs["agreement"] = {
        "r_kdm": float(np.corrcoef(f["kdm"], r["kdm"])[0, 1]),
        "r_advance": float(np.corrcoef(f["advance"], r["advance"])[0, 1]),
        "mean_abs_diff_years": float((f["kdm"] - r["kdm"]).abs().mean()),
    }
    return table


# ---------------------------------------------------------------------------
# BioAge's kdm0, packaged
# ---------------------------------------------------------------------------
#: Where a sex label may come from. NHANES codes 1 = male and 2 = female.
_SEX = {"1": "male", "m": "male", "male": "male",
        "2": "female", "f": "female", "female": "female"}


def load_kdm_bioage() -> dict[str, KDMReference]:
    """BioAge's ``kdm0`` references, one per sex, as fitted on NHANES III.

    Built by ``python/tools/build_kdm_bioage.py`` from the NHANES III rows
    BioAge trains ``kdm0`` on (30 to 75, non-pregnant), and recorded with the
    agreement the refit reaches against BioAge's own column.
    """
    import json

    from ..registry.registry import DATA_DIR

    raw = json.loads((DATA_DIR / "kdm_bioage_nhanes3.json").read_text())
    refs = {}
    for sex, f in raw["fits"].items():
        refs[sex] = KDMReference(
            markers=list(f["markers"]), k=np.asarray(f["k"]), q=np.asarray(f["q"]),
            s=np.asarray(f["s"]), r=np.asarray(f["r"]), r_char=float(f["r_char"]),
            s_r=float(f["s_r"]), n_reference=int(f["n_reference"]),
            age_range=tuple(f["age_range"]), s_ba2=float(f["s_ba2"]),
            n_per_marker=np.asarray(f["n_per_marker"]),
            ranges=pd.DataFrame.from_dict(f["ranges"], orient="index"))
    return refs


def kdm_bioage(df: pd.DataFrame, sex_col: str = "sex", age_col: str = "age",
               max_missing: int | None = 2) -> pd.Series:
    """KDM biological age on the scale of BioAge's ``kdm0``.

    What "KDM biological age" means in most NHANES papers: nine biomarkers,
    fitted by sex on NHANES III aged 30 to 75, non-pregnant, and projected with
    the training ``s_BA^2`` (Kwon and Belsky 2021). The packaged fit reproduces
    BioAge's ``kdm0`` to under 0.001 years mean absolute difference.

    The columns and units are BioAge's, and must be supplied as such:
    ``fev`` (FEV1, mL), ``sbp`` (mmHg), ``totchol`` (mg/dL), ``hba1c`` (% NGSP),
    ``albumin`` (g/dL), ``creat`` (mg/dL), ``lncrp`` (``log(1 + CRP)`` with CRP
    in mg/dL, BioAge's transform), ``alp`` (U/L) and ``bun`` (mg/dL).

    ``sex_col`` takes ``male``/``female``, ``M``/``F`` or NHANES's 1/2.
    ``max_missing=0`` reproduces BioAge's shipped column exactly, missing rows
    included; the default 2 is the rule in BioAge's current ``kdm_calc``.
    """
    if sex_col not in df.columns:
        raise DataError(f"no {sex_col!r} column; kdm0 is fitted separately by sex")
    labels = df[sex_col].map(lambda v: _SEX.get(str(v).strip().lower().removesuffix(".0")))
    unknown = sorted({str(v) for v, s in zip(df[sex_col], labels) if s is None})
    if unknown:
        raise DataError(f"sex values FALCONAge cannot map to male/female: {unknown[:6]}. "
                        "Use male/female, M/F or NHANES's 1/2.")
    refs = load_kdm_bioage()
    out = pd.Series(np.nan, index=df.index, name="kdm")
    for sex, ref in refs.items():
        rows = labels == sex
        if rows.any():
            out[rows] = kdm(df[rows], ref, age_col=age_col, max_missing=max_missing)
    return out


def bioage_hd_scale(hd_values: pd.Series) -> pd.DataFrame:
    """BioAge's two cohort-relative forms of homeostatic dysregulation.

    BioAge's ``hd_calc`` divides the Mahalanobis distance by its standard
    deviation in the scored cohort and reports ``log(hd) / sd(log(hd))`` beside
    it. :func:`hd` returns the raw distance, which is comparable between
    cohorts scored against the same reference; these two are not, because the
    divisor is the cohort's own spread. Use them to compare with BioAge output.
    """
    v = pd.to_numeric(hd_values, errors="coerce")
    logv = np.log(v)
    return pd.DataFrame({"hd": v / v.std(ddof=1), "hd_log": logv / logv.std(ddof=1)},
                        index=v.index)


# ---------------------------------------------------------------------------
# Does the scored cohort live where the reference did?
# ---------------------------------------------------------------------------
#: A value shared by at least this share of reference rows at the reference's
#: minimum is a reporting floor, not a continuous measurement. A continuous
#: marker puts about one row in n at its minimum; NHANES III puts 63.9% of CRP
#: values at 0.21 mg/dL, the assay's detection floor.
FLOOR_SHARE = 0.01

#: Report a marker when at least this share of the scored cohort lies outside
#: the reference's range. A reporting threshold, not a statistic.
OUT_OF_RANGE_WARN = 0.05


def marker_ranges(reference: pd.DataFrame, markers: list[str]) -> pd.DataFrame:
    """Each marker's reference distribution: min, 1st, 50th, 99th percentile, max.

    ``share_at_min`` is the share of rows sitting exactly at the minimum, which
    is how a detection floor shows itself in the data.
    """
    rows = {}
    for m in markers:
        v = pd.to_numeric(reference[m], errors="coerce").dropna().to_numpy(dtype=np.float64)
        if v.size == 0:
            continue
        lo = float(v.min())
        rows[m] = {"min": lo, "p01": float(np.percentile(v, 1)),
                   "p50": float(np.median(v)), "p99": float(np.percentile(v, 99)),
                   "max": float(v.max()),
                   "share_at_min": float(np.isclose(v, lo, rtol=0, atol=1e-9 * max(1.0, abs(lo))).mean())}
    return pd.DataFrame.from_dict(rows, orient="index")


def reference_range_check(df: pd.DataFrame, ref) -> list[str]:
    """Where this cohort's markers leave the reference, stated as facts.

    Read off the reference's own recorded distribution, per marker:

    * **Values below the reference's reporting floor.** Either the cohort was
      measured on a more sensitive assay than the reference (hs-CRP against
      NHANES III, whose CRP assay read nothing below 0.21 mg/dL), in which case
      :func:`censor_to_reference` reads both the same way, or the unit differs.
      The data cannot tell those apart, so the message names both.
    * **A median outside the reference's 1st to 99th percentile.** Nearly always
      a unit mismatch: albumin in g/dL against a g/L reference sits there, and
      both are clinically normal in their own unit.
    * **Many values outside the reference's range**, where there is no floor to
      explain them: the model is extrapolating.
    """
    ranges = getattr(ref, "ranges", None)
    if ranges is None or ranges.empty:
        return []
    out = []
    for m in ref.markers:
        if m not in df.columns or m not in ranges.index:
            continue
        v = pd.to_numeric(df[m], errors="coerce").dropna().to_numpy(dtype=np.float64)
        if v.size == 0:
            continue
        r = ranges.loc[m]
        med = float(np.median(v))
        below, above = float((v < r["min"]).mean()), float((v > r["max"]).mean())
        floored = r["share_at_min"] >= FLOOR_SHARE
        said_floor = False
        if floored and below >= OUT_OF_RANGE_WARN:
            said_floor = True
            out.append(
                f"{m}: {below:.0%} of values fall below {r['min']:.4g}, the reference's "
                f"reporting floor ({r['share_at_min']:.0%} of the reference sits at it). "
                "Check the unit first; if it matches, this cohort was measured more "
                "sensitively than the reference and censor_to_reference() reads both "
                "the same way.")
        elif below + above >= OUT_OF_RANGE_WARN:
            out.append(
                f"{m}: {below + above:.0%} of values lie outside the reference's range "
                f"({r['min']:.4g} to {r['max']:.4g}); the model is extrapolating there.")
        if not r["p01"] <= med <= r["p99"] and not (said_floor and med < r["min"]):
            out.append(
                f"{m}: cohort median {med:.4g} lies outside the reference's 1st to 99th "
                f"percentile ({r['p01']:.4g} to {r['p99']:.4g}); check the unit first.")
    return out


def censor_to_reference(df: pd.DataFrame, ref, markers: list[str] | None = None
                        ) -> tuple[pd.DataFrame, dict[str, int]]:
    """Set values below a reference's reporting floor to the floor.

    A reference whose assay could not read below a limit records every lower
    value as the limit, and a model fitted on it has seen nothing below. Values
    under that limit from a more sensitive assay are therefore read on the
    reference's scale only after being set to it. Only markers whose reference
    minimum is a floor (``share_at_min >= FLOOR_SHARE``) are touched unless
    ``markers`` names others; values above the reference range are real
    measurements and are left alone.

    Returns the censored copy and, per marker, how many values moved.
    """
    ranges = getattr(ref, "ranges", None)
    if ranges is None or ranges.empty:
        raise AnalysisError("this reference has no recorded ranges; refit it with fit_kdm or fit_hd")
    chosen = markers if markers is not None else [
        m for m in ranges.index if ranges.at[m, "share_at_min"] >= FLOOR_SHARE]
    out, moved = df.copy(), {}
    for m in chosen:
        if m not in out.columns:
            continue
        floor = float(ranges.at[m, "min"])
        v = pd.to_numeric(out[m], errors="coerce")
        hit = v < floor
        moved[m] = int(hit.sum())
        out[m] = v.where(~hit, floor)
    return out, moved


# ---------------------------------------------------------------------------
# Homeostatic dysregulation
# ---------------------------------------------------------------------------
@dataclass
class HDReference:
    markers: list[str]
    centre: np.ndarray
    inv_cov: np.ndarray
    n_reference: int
    description: str = ""
    #: Each marker's reference distribution; see :func:`marker_ranges`.
    ranges: pd.DataFrame | None = None


def fit_hd(reference: pd.DataFrame, markers: list[str]) -> HDReference:
    """Estimate the healthy reference centre and covariance.

    The reference should be the healthy young subset, not the whole cohort.
    Kwon and Belsky ship ``NHANES3_HDTrain`` for exactly this, and it is not
    interchangeable with ``NHANES3``: fitting the centre on everybody makes the
    average unhealthy person the definition of normal.
    """
    ref = reference[markers].dropna()
    if len(ref) < len(markers) + 5:
        raise AnalysisError(
            f"HD reference has {len(ref)} complete rows for {len(markers)} markers; "
            "the covariance is singular or nearly so. Supply more rows or fewer markers."
        )
    m = ref.to_numpy(dtype=np.float64)
    centre = m.mean(axis=0)
    cov = np.cov(m, rowvar=False)
    # pinv, not inv: clinical panels contain near-collinear pairs (total and LDL
    # cholesterol, urea and creatinine) and a plain inverse turns that into an
    # enormous distance for one sample and a silent NaN for the next.
    return HDReference(list(markers), centre, np.linalg.pinv(cov), len(ref),
                       "pseudo-inverse covariance", marker_ranges(reference, markers))


def hd(df: pd.DataFrame, ref: HDReference) -> pd.Series:
    """Homeostatic dysregulation: Mahalanobis distance from the reference."""
    missing = [m for m in ref.markers if m not in df.columns]
    if missing:
        raise DataError("HD needs the reference's markers; missing: " + ", ".join(missing))
    d = df[ref.markers].to_numpy(dtype=np.float64) - ref.centre
    # einsum rather than a loop: this is the one clinical clock that is O(n·p²).
    m2 = np.einsum("ij,jk,ik->i", d, ref.inv_cov, d)
    return pd.Series(np.sqrt(np.maximum(m2, 0.0)), index=df.index, name="hd")


FORMULAS = {"phenoage": phenoage, "kdm": kdm, "hd": hd}


@dataclass
class ClinicalClock:
    """Adapter that gives the three formulas the same interface as a LinearClock.

    KDM and HD need a reference cohort and PhenoAge does not, so the reference
    is passed at predict time rather than stored on the model. That keeps the
    model object free of data and makes "which reference produced this number"
    a property of the run, which is where the manifest can see it.
    """

    #: Nine markers is not a workload for a device; see the module docstring.
    #: Declared rather than implicit so :func:`falconage.models.effective_spec`
    #: can tell the caller what these clocks actually computed in.
    CPU_ONLY = True

    clock: object
    reference: object | None = None

    def predict(self, data, spec=None, *, reference=None, **kw):
        """Score. ``spec`` is accepted for interface uniformity and unused.

        Unused is not the same as ignored: ``CPU_ONLY`` above tells the scoring
        loop so, and the manifest records ``cpu`` for this clock rather than
        whatever the run asked for.
        """
        from ..core.errors import AnalysisError

        ref = reference if reference is not None else self.reference
        df = data.X.join(data.obs, how="left", rsuffix="_obs")
        name = self.clock.formula

        if name == "phenoage":
            return phenoage(df), None
        if ref is None:
            raise AnalysisError(
                f"{self.clock.id} needs a reference cohort.\n"
                "  It has no fixed coefficients: "
                + ("the per-marker regressions on age are refitted on the "
                   "reference" if name == "kdm" else
                   "the centre and covariance come from the reference")
                + ".\n  Pass reference=fa.models.clinical.fit_"
                + f"{name}(reference_df, markers).\n"
                "  test/data/clinical/ carries the NHANES extracts the published "
                "papers used."
            )
        if name == "kdm":
            return kdm(df, ref), None
        if name == "hd":
            return hd(df, ref), None
        raise AnalysisError(f"unknown clinical formula {name!r}")
