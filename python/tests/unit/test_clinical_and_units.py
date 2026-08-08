"""Clinical clocks and the refusal to guess units."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import falconage as fa
from falconage.core import units
from falconage.core.errors import DataError, UnitConversionError, UnitsNotDeclaredError
from falconage.models import clinical


# ---------------------------------------------------------------------------
# units
# ---------------------------------------------------------------------------
def test_units_must_be_declared():
    with pytest.raises(UnitsNotDeclaredError) as exc:
        units.require_units(None, ["albumin", "creatinine", "glucose"])
    msg = str(exc.value)
    assert "will not infer" in msg
    assert "units=" in msg, "the error must show the caller what to pass"


def test_partial_units_are_rejected():
    with pytest.raises(UnitsNotDeclaredError, match="missing entries"):
        units.require_units({"albumin": "g/L"}, ["albumin", "glucose"])


def test_the_albumin_trap():
    """4.2 g/dL and 42 g/L are the same measurement; both are clinically normal;
    a coefficient applied to the wrong one is out by a factor of ten."""
    assert units.convert(4.2, "g/dL", "g/L") == pytest.approx(42.0)
    assert units.check_plausible("albumin", [4.2]) is not None   # implausible in g/L
    assert units.check_plausible("albumin", [42.0]) is None


def test_creatinine_molar_conversion():
    # 1 mg/dL = 88.4017 umol/L for a molar mass of 113.12 g/mol
    assert units.convert(1.0, "mg/dL", "umol/L") == pytest.approx(88.4017)


def test_undefined_conversion_is_an_error_not_a_guess():
    with pytest.raises(UnitConversionError, match="enumerated rather than derived"):
        units.convert(1.0, "furlongs", "g/L")


def test_gestational_days_to_weeks():
    """The corpus's cord blood series records days; every gestational clock
    predicts weeks."""
    assert units.convert(280.0, "days", "weeks") == pytest.approx(40.0)


# ---------------------------------------------------------------------------
# PhenoAge
# ---------------------------------------------------------------------------
def test_phenoage_coefficients_are_the_published_ten():
    assert len(clinical.PHENOAGE_COEF) == 10
    assert clinical.PHENOAGE_COEF["age"] == pytest.approx(0.0804)
    assert clinical.PHENOAGE_INTERCEPT == pytest.approx(-19.9067)


def test_phenoage_tracks_age_and_responds_to_biomarkers(synthetic_clinical):
    v = clinical.phenoage(synthetic_clinical.X)
    age = synthetic_clinical.X["age"]
    assert v.notna().all()
    assert v.corr(age) > 0.9, "PhenoAge is dominated by its age term, as published"
    assert 10 < v.mean() < 110

    worse = synthetic_clinical.X.copy()
    worse["crp"] = worse["crp"] * 10          # a full log unit of inflammation
    assert clinical.phenoage(worse).mean() > v.mean()


def test_phenoage_refuses_a_missing_marker(synthetic_clinical):
    df = synthetic_clinical.X.drop(columns=["albumin"])
    with pytest.raises(DataError, match="missing: albumin"):
        clinical.phenoage(df)


def test_phenoage_refuses_zero_crp(synthetic_clinical):
    df = synthetic_clinical.X.copy()
    df.loc[df.index[0], "crp"] = 0.0
    with pytest.raises(DataError, match="strictly positive"):
        clinical.phenoage(df)


def test_phenoage_in_wrong_units_gives_a_visibly_different_answer(synthetic_clinical):
    """Not a defect -- a demonstration of why the units module refuses to guess."""
    right = clinical.phenoage(synthetic_clinical.X)
    wrong = synthetic_clinical.X.copy()
    wrong["albumin"] = wrong["albumin"] / 10.0       # g/L read as g/dL
    assert abs(clinical.phenoage(wrong).mean() - right.mean()) > 1.0


# ---------------------------------------------------------------------------
# KDM and HD
# ---------------------------------------------------------------------------
MARKERS = ["albumin", "creatinine", "glucose", "crp", "lymphocyte_percent",
           "mean_cell_volume", "red_cell_distribution_width",
           "alkaline_phosphatase", "white_blood_cell_count"]


def test_kdm_recovers_age_on_its_own_reference(synthetic_clinical):
    """Scoring the reference cohort against itself must give back something
    close to chronological age -- that is what the estimator is for.

    The absolute threshold is deliberately loose. Across thirty seeds of this
    cohort the correlation runs 0.80 to 0.93, so anything at or above about 0.9
    is a threshold on the seed rather than on the estimator, and it fails the
    day the fixture is regenerated. The previous version asserted 0.85 and did
    exactly that.

    The tight assertion is the third one, and it is the one that says KDM
    works: combining nine weak markers has to beat the best single marker,
    which correlates about 0.38 here. That held in 30/30 draws with a margin
    never below 0.39, and it is the property that would actually break if the
    estimator regressed.
    """
    df = synthetic_clinical.X
    ref = clinical.fit_kdm(df, MARKERS)
    ba = clinical.kdm(df, ref)

    r = ba.corr(df["age"])
    assert r > 0.75, f"KDM correlated {r:.3f} with age on its own reference"
    assert abs(float((ba - df["age"]).median())) < 5.0

    best_single = max(abs(df[m].corr(df["age"])) for m in MARKERS)
    assert r > best_single + 0.2, (
        f"KDM ({r:.3f}) barely beat the best single marker ({best_single:.3f}); "
        "combining the panel is supposed to be worth something")


def test_kdm_refuses_a_marker_that_does_not_vary(synthetic_clinical):
    """A constant column is a data error, and it used to poison the answer.

    Zero residual spread makes k/s an infinity, corrcoef of a constant a NaN,
    and r_char a NaN -- after which nansum carries on and returns a plausible
    number from a reference that is mostly not-a-number. Refusing is the only
    safe behaviour, and the message has to name the column.
    """
    from falconage.core.errors import AnalysisError

    df = synthetic_clinical.X.copy()
    df["white_blood_cell_count"] = 6.5

    with pytest.raises(AnalysisError, match="does not vary"):
        clinical.fit_kdm(df, MARKERS)


def test_kdm_needs_a_reference_large_enough_to_regress(synthetic_clinical):
    from falconage.core.errors import AnalysisError

    with pytest.raises(AnalysisError, match="fewer than 30"):
        clinical.fit_kdm(synthetic_clinical.X.head(10), MARKERS)


def test_hd_is_zero_at_the_reference_centre(synthetic_clinical):
    """A sample sitting exactly at the reference mean has no dysregulation."""
    df = synthetic_clinical.X
    ref = clinical.fit_hd(df, MARKERS)
    centre = pd.DataFrame([ref.centre], columns=MARKERS, index=["centre"])
    assert clinical.hd(centre, ref).iloc[0] == pytest.approx(0.0, abs=1e-8)


def test_hd_grows_with_distance(synthetic_clinical):
    df = synthetic_clinical.X
    ref = clinical.fit_hd(df, MARKERS)
    near = pd.DataFrame([ref.centre], columns=MARKERS, index=["a"])
    far = pd.DataFrame([ref.centre * 1.5], columns=MARKERS, index=["b"])
    assert clinical.hd(far, ref).iloc[0] > clinical.hd(near, ref).iloc[0]


def test_hd_uses_a_pseudo_inverse_so_collinear_panels_still_work(synthetic_clinical):
    """Clinical panels contain near-collinear pairs; a plain inverse blows up."""
    df = synthetic_clinical.X.copy()
    df["albumin_copy"] = df["albumin"]
    ref = clinical.fit_hd(df, MARKERS + ["albumin_copy"])
    out = clinical.hd(df, ref)
    assert np.isfinite(out).all()


def test_clinical_clock_without_a_reference_says_what_to_pass(synthetic_clinical):
    from falconage.core.errors import AnalysisError

    with pytest.raises(AnalysisError, match="needs a reference cohort"):
        fa.score(synthetic_clinical, clocks=["kdm"])


def test_scoring_clinical_end_to_end(synthetic_clinical):
    ref = clinical.fit_kdm(synthetic_clinical.X, MARKERS)
    res = fa.score(synthetic_clinical, clocks=["phenoage", "kdm"], reference=ref)
    assert list(res.scores.columns) == ["phenoage", "kdm"]
    assert res.scores.notna().all().all()
