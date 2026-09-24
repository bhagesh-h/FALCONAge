"""Clinical clocks and the refusal to guess units."""

from __future__ import annotations

from pathlib import Path

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


KDM0_FIXTURE = Path(__file__).resolve().parents[1] / "data" / "nhanes3_kdm0_fixture.csv.gz"
KDM0_MARKERS = ["fev", "sbp", "totchol", "hba1c", "albumin", "creat", "lncrp", "alp", "bun"]


def test_kdm_reproduces_bioage_kdm0():
    """The reference implementation, reproduced to its own rounding.

    BioAge trains ``kdm0`` by sex on NHANES III aged 30 to 75, non-pregnant, and
    its shipped column is complete-case (every row with a value carries all nine
    markers), which ``max_missing=0`` reproduces. FALCONAge 1.0.0 was 1.77 years
    out on these rows, most of it from taking the square root of a correlation
    in ``r_char``; see python/tests/data/SOURCE.md for the fixture.
    """
    d = pd.read_csv(KDM0_FIXTURE)
    parts = []
    for sex in (1, 2):
        rows = d[d["gender"] == sex]
        ref = clinical.fit_kdm(rows, KDM0_MARKERS)
        parts.append(pd.DataFrame({"kdm": clinical.kdm(rows, ref, max_missing=0),
                                   "kdm0": rows["kdm0"]}))
    out = pd.concat(parts)

    assert (out["kdm"].isna() == out["kdm0"].isna()).all(), "missing pattern differs"
    both = out.dropna()
    diff = (both["kdm"] - both["kdm0"]).abs()
    assert len(both) == 9583
    assert diff.mean() < 0.005, f"mean |difference| {diff.mean():.4f} years"
    assert diff.max() < 0.02, f"max |difference| {diff.max():.4f} years"


def test_r_char_weights_the_correlation_not_its_square_root():
    """BioAge: r1 = |k/s| * sqrt(R^2), and sqrt(R^2) is |correlation|.

    Two markers built so that their correlations with age are known; r_char
    must be the |k/s|-weighted mean of those correlations.
    """
    rng = np.random.default_rng(3)
    age = rng.uniform(30, 75, 400)
    df = pd.DataFrame({"age": age,
                       "a": 2.0 * age + rng.normal(0, 20, 400),
                       "b": -0.5 * age + rng.normal(0, 30, 400)})
    ref = clinical.fit_kdm(df, ["a", "b"])

    corr = np.array([abs(np.corrcoef(age, df[m])[0, 1]) for m in ("a", "b")])
    w = np.abs(ref.k / ref.s)
    assert ref.r == pytest.approx(corr)
    assert ref.r_char == pytest.approx(float(np.sum(w * corr) / np.sum(w)))
    assert ref.r_char != pytest.approx(float(np.sum(w * np.sqrt(corr)) / np.sum(w)))


def test_kdm_follows_bioage_on_missing_markers(synthetic_clinical):
    """More than two markers missing is NaN, BioAge's rule; None lifts it."""
    df = synthetic_clinical.X
    ref = clinical.fit_kdm(df, MARKERS)
    one = df.head(1).copy()
    one.loc[:, MARKERS[:3]] = np.nan

    assert np.isnan(clinical.kdm(one, ref).iloc[0])
    assert np.isfinite(clinical.kdm(one, ref, max_missing=3).iloc[0])
    assert np.isfinite(clinical.kdm(one, ref, max_missing=None).iloc[0])


def test_kdm_projects_with_the_reference_s_ba2(synthetic_clinical):
    """s_BA^2 is fitted on the reference and reused, as BioAge projects.

    Scoring a different cohort must use the reference's value, not one
    re-estimated on the cohort being scored.
    """
    df = synthetic_clinical.X
    ref = clinical.fit_kdm(df.iloc[:300], MARKERS)
    other = df.iloc[300:]
    x = other[MARKERS].to_numpy(dtype=float)
    num = np.sum((x - ref.q) * ref.k / ref.s**2, axis=1)
    den = float(np.sum((ref.k / ref.s) ** 2))
    expected = (num + other["age"].to_numpy() / ref.s_ba2) / (den + 1.0 / ref.s_ba2)

    assert ref.s_ba2 > 0
    assert clinical.kdm(other, ref).to_numpy() == pytest.approx(expected)


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


# ---------------------------------------------------------------------------
# Harmonisation: HbA1c units, and a cohort measured unlike its reference
# ---------------------------------------------------------------------------

def test_hba1c_converts_by_the_ngsp_master_equation():
    """NGSP = 0.09148 x IFCC + 2.152 (ngsp.org; Hoelzel et al. 2004)."""
    assert units.convert(48.0, "mmol/mol", "%", marker="hba1c") == pytest.approx(
        0.09148 * 48.0 + 2.152)
    back = units.convert(6.5, "%", "mmol/mol", marker="hba1c")
    assert back == pytest.approx(10.93 * 6.5 - 23.50)
    # The two published directions are not exact inverses; the round trip is close.
    assert units.convert(back, "mmol/mol", "%", marker="hba1c") == pytest.approx(6.5, abs=0.01)


def test_the_hba1c_equation_is_not_a_general_unit_rule():
    with pytest.raises(UnitConversionError):
        units.convert(48.0, "mmol/mol", "%")


def test_prepare_clinical_converts_ifcc_hba1c(synthetic_clinical):
    X = synthetic_clinical.X[["age"]].assign(hba1c=[48.0] * len(synthetic_clinical.X))
    d = fa.FalconData(X=X, obs=synthetic_clinical.obs, modality="clinical_chemistry")
    out = fa.prepare_clinical(d, units={"hba1c": "mmol/mol", "age": "years"},
                              target={"hba1c": "%"})
    assert out.X["hba1c"].iloc[0] == pytest.approx(0.09148 * 48.0 + 2.152)
    assert "hba1c: mmol/mol -> %" in out.uns["unit_conversions"]


def _nhanes_crp_reference():
    d = pd.read_csv(KDM0_FIXTURE)
    return d[d["gender"] == 1], clinical.fit_kdm(d[d["gender"] == 1], KDM0_MARKERS)


def test_the_nhanes_crp_floor_is_found_in_the_data():
    """63.9% of NHANES III CRP values sit at the 0.21 mg/dL detection floor."""
    _, ref = _nhanes_crp_reference()
    assert ref.ranges.at["lncrp", "share_at_min"] > 0.5
    assert np.expm1(ref.ranges.at["lncrp", "min"]) == pytest.approx(0.21, abs=1e-3)
    assert ref.ranges.at["sbp", "share_at_min"] < clinical.FLOOR_SHARE


def test_a_more_sensitive_assay_is_flagged_and_censored():
    rows, ref = _nhanes_crp_reference()
    cohort = rows.head(200).copy()
    floor = ref.ranges.at["lncrp", "min"]
    cohort["lncrp"] = np.log1p(np.linspace(0.02, 0.20, 200))    # hs-CRP, all below 0.21

    msgs = clinical.reference_range_check(cohort, ref)
    assert any(m.startswith("lncrp:") and "censor_to_reference" in m for m in msgs)

    fixed, moved = clinical.censor_to_reference(cohort, ref)
    assert moved["lncrp"] == 200 and (fixed["lncrp"] == floor).all()
    assert not any(m.startswith("lncrp:") for m in clinical.reference_range_check(fixed, ref))
    # A marker without a floor is not touched by default.
    assert "sbp" not in moved


def test_a_unit_mismatch_is_flagged_by_its_median(synthetic_clinical):
    df = synthetic_clinical.X
    ref = clinical.fit_kdm(df, MARKERS)
    wrong = df.assign(albumin=df["albumin"] / 10.0)          # g/dL against a g/L reference
    msgs = clinical.reference_range_check(wrong, ref)
    assert any(m.startswith("albumin:") and "unit" in m for m in msgs)
    assert clinical.reference_range_check(df, ref) == []


def test_score_reports_reference_range_problems(synthetic_clinical):
    ref = clinical.fit_kdm(synthetic_clinical.X, MARKERS)
    X = synthetic_clinical.X.assign(albumin=synthetic_clinical.X["albumin"] / 10.0)
    d = fa.FalconData(X=X, obs=synthetic_clinical.obs, modality="clinical_chemistry")
    res = fa.score(d, clocks=["kdm"], reference=ref)
    assert any(r["category"] == "reference_range" for r in res.manifest.warnings)


# ---------------------------------------------------------------------------
# BioAge compatibility
# ---------------------------------------------------------------------------

def test_kdm_bioage_reproduces_kdm0_from_the_packaged_fit():
    d = pd.read_csv(KDM0_FIXTURE)
    k = clinical.kdm_bioage(d, sex_col="gender", max_missing=0)
    assert (k.isna() == d["kdm0"].isna()).all()
    both = pd.DataFrame({"k": k, "k0": d["kdm0"]}).dropna()
    assert len(both) == 9583
    assert (both["k"] - both["k0"]).abs().mean() < 0.005


def test_kdm_bioage_reads_sex_as_text_or_nhanes_code():
    d = pd.read_csv(KDM0_FIXTURE).head(300)
    by_code = clinical.kdm_bioage(d, sex_col="gender")
    by_text = clinical.kdm_bioage(d.assign(gender=d["gender"].map({1: "M", 2: "female"})),
                                  sex_col="gender")
    assert by_code.equals(by_text)
    with pytest.raises(DataError, match="cannot map"):
        clinical.kdm_bioage(d.assign(gender="x"), sex_col="gender")


def test_the_packaged_kdm_bioage_file_is_current():
    """The JSON is a derived artefact; refitting the fixture must give it back."""
    import importlib.util

    tool = Path(__file__).resolve().parents[2] / "tools" / "build_kdm_bioage.py"
    spec = importlib.util.spec_from_file_location("build_kdm_bioage", tool)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.TARGET.read_text() == mod.render(mod.build())


def test_phenoage_log1p_crp_is_bioage_s_transform(synthetic_clinical):
    df = synthetic_clinical.X
    assert clinical.phenoage(df, crp_transform="log1p").to_numpy() == pytest.approx(
        clinical.phenoage(df.assign(crp=1.0 + df["crp"])).to_numpy())
    assert (clinical.phenoage(df, crp_transform="log1p") > clinical.phenoage(df)).all()
    with pytest.raises(DataError, match="crp_transform"):
        clinical.phenoage(df, crp_transform="ln")


def test_bioage_hd_scale_divides_by_the_cohort_spread(synthetic_clinical):
    ref = clinical.fit_hd(synthetic_clinical.X, MARKERS)
    out = clinical.bioage_hd_scale(clinical.hd(synthetic_clinical.X, ref))
    assert out["hd"].std(ddof=1) == pytest.approx(1.0)
    assert out["hd_log"].std(ddof=1) == pytest.approx(1.0)
