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
# unit, so the numbers below can be checked line by line against Table 1.
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
    "alkaline_phosphatase": 0.00188,
    "white_blood_cell_count": 0.0554,
    "age": 0.0804,
}
PHENOAGE_INTERCEPT = -19.9067

#: Gompertz calibration. gamma is the shape; t=120 months is the horizon the
#: mortality score is expressed at; the rest fall out of the published inversion.
PHENOAGE_GAMMA = 0.0076927
PHENOAGE_T = 120.0

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


def phenoage(df: pd.DataFrame) -> pd.Series:
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
    """
    missing = [m for m in PHENOAGE_UNITS if m not in df.columns]
    if missing:
        raise DataError(
            "clinical PhenoAge needs all ten markers; missing: " + ", ".join(missing)
            + "\n  It is a closed form with no imputation step -- the published "
              "model has no term for an absent marker, and substituting a cohort "
              "mean invents one."
        )

    x = df.copy()
    crp = np.asarray(x["crp"], dtype=np.float64)
    if np.nanmin(crp) <= 0:
        raise DataError(
            "CRP must be strictly positive: PhenoAge takes its logarithm. "
            "Values reported as 0 are usually below the assay's detection limit; "
            "substitute the limit itself rather than zero."
        )
    x["log_crp"] = np.log(crp)

    xb = np.full(len(x), PHENOAGE_INTERCEPT, dtype=np.float64)
    for marker, beta in PHENOAGE_COEF.items():
        xb += beta * np.asarray(x[marker], dtype=np.float64)

    # Mortality score at 120 months under the Gompertz hazard, then inverted
    # back onto the age scale.
    mortality = 1.0 - np.exp(-np.exp(xb) * (np.exp(PHENOAGE_GAMMA * PHENOAGE_T) - 1.0)
                             / PHENOAGE_GAMMA)
    mortality = np.clip(mortality, 1e-12, 1.0 - 1e-12)
    return pd.Series(
        141.50225 + np.log(-0.00553 * np.log(1.0 - mortality)) / 0.090165,
        index=df.index, name="phenoage",
    )


# ---------------------------------------------------------------------------
# Klemera-Doubal
# ---------------------------------------------------------------------------
@dataclass
class KDMReference:
    """Per-biomarker regressions on chronological age, fitted on a reference.

    Follows Klemera and Doubal 2006 as implemented in the BioAge R package,
    including the ``s_R`` correction. Every symbol below is theirs:

    ``k``, ``q``, ``s``
        slope, intercept and residual standard deviation of each biomarker
        regressed on chronological age.
    ``r_char``
        the characteristic correlation, a ``|k/s|``-weighted mean of the
        per-marker correlations with age.
    ``s_r``
        the variance the estimator would have if biological and chronological
        age were the same thing. Subtracting it from the observed variance is
        what stops KDM collapsing onto chronological age when the biomarkers
        carry little information -- omit it and every KDM paper's headline
        finding becomes an artefact of the age range.
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
    s_ba2: float | None = None   # None: estimated from the scored cohort

    def describe(self) -> pd.DataFrame:
        return pd.DataFrame({"slope": self.k, "intercept": self.q,
                             "resid_sd": self.s, "cor_with_age": self.r},
                            index=self.markers)


def fit_kdm(reference: pd.DataFrame, markers: list[str], age_col: str = "age",
            s_ba2: float | None = None) -> KDMReference:
    """Fit the KDM reference regressions.

    ``s_ba2`` is normally left ``None`` and estimated from the cohort being
    scored, as the reference implementation does. Pass a value to hold it fixed
    when scoring several cohorts that must be comparable -- otherwise each gets
    its own and the numbers are not on the same scale.
    """
    ref = reference.dropna(subset=[age_col, *markers])
    if len(ref) < 30:
        raise AnalysisError(
            f"KDM reference has {len(ref)} complete rows; fewer than 30 makes the "
            "per-marker regressions meaningless. Supply a larger reference cohort."
        )
    age = ref[age_col].to_numpy(dtype=np.float64)
    k, q, s, r = [], [], [], []
    degenerate = []
    for m in markers:
        y = ref[m].to_numpy(dtype=np.float64)
        slope, intercept = np.polyfit(age, y, 1)
        resid = y - (slope * age + intercept)
        sd = float(np.std(resid, ddof=2))
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
        r.append(abs(float(np.corrcoef(age, y)[0, 1])))

    if degenerate:
        raise AnalysisError(
            "KDM cannot use a marker that does not vary: "
            + ", ".join(f"{m!r}" for m in degenerate)
            + f"\n  Each has zero residual spread across the {len(ref)} reference "
            "rows, so its contribution to the estimate is a division by zero.\n"
            "  This is nearly always a data problem rather than a biological one "
            "-- a unit conversion that\n  collapsed the column, or one value "
            "filled down. Check the column, then either fix it or\n  leave it out "
            "of `markers=`; KDM is defined for any panel size."
        )

    k, q, s, r = (np.asarray(v, dtype=np.float64) for v in (k, q, s, r))

    ks = np.abs(k / s)
    r_char = float(np.sum(ks * np.sqrt(r)) / np.sum(ks))
    lo, hi = float(age.min()), float(age.max())
    s_r = float(((1 - r_char**2) / r_char**2) * ((hi - lo) ** 2 / (12 * len(markers)))) \
        if r_char > 0 else np.inf

    return KDMReference(list(markers), k, q, s, r, r_char, s_r, len(ref), (lo, hi), s_ba2)


def kdm(df: pd.DataFrame, ref: KDMReference, age_col: str = "age") -> pd.Series:
    """Klemera-Doubal biological age in years."""
    missing = [m for m in ref.markers if m not in df.columns]
    if missing:
        raise DataError("KDM needs the reference's markers; missing: " + ", ".join(missing))
    x = df[ref.markers].to_numpy(dtype=np.float64)
    age = df[age_col].to_numpy(dtype=np.float64)

    num = np.nansum((x - ref.q) * ref.k / ref.s**2, axis=1)
    den = float(np.nansum((ref.k / ref.s) ** 2))

    # BA_E, the estimate before the chronological-age term, rescaled for any
    # marker this sample is missing so a partial panel is not silently shrunk
    # toward zero.
    n_obs = np.isfinite(x).sum(axis=1)
    ba_e = (num / den) * (len(ref.markers) / np.maximum(n_obs, 1))

    s_ba2 = ref.s_ba2
    if s_ba2 is None:
        d = ba_e - age
        s_ba2 = float(np.nanmean((d - np.nanmean(d)) ** 2) - ref.s_r)
    if not np.isfinite(s_ba2) or s_ba2 <= 0:
        # The biomarkers carry no information beyond age. Klemera and Doubal's
        # correction is larger than the observed spread, and the honest answer
        # is the biomarker estimate itself rather than a division by a negative
        # variance that silently flips the sign of the age term.
        return pd.Series(ba_e, index=df.index, name="kdm")

    return pd.Series((num + age / s_ba2) / (den + 1.0 / s_ba2),
                     index=df.index, name="kdm")


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
                       "pseudo-inverse covariance")


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
