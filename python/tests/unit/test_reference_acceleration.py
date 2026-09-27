"""Acceleration against an external reference population (method="reference").

The known-answer design: a reference population whose clock follows a known
line in age, with a slope below one as a clock that regresses toward the mean
age of its training population does, and an older study cohort scored on the
same line plus a known shift. The reference method has to recover the shift.
The within-cohort residual cannot, because it is centred on the study by
construction, and ``predicted - chronological`` gets the sign wrong, because at
these ages the regression toward the mean outweighs a shift of three years.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import falconage as fa
from falconage.core.errors import AnalysisError, IllegalOperationError
from falconage.core.manifest import RunManifest
from falconage.models import clinical

CLOCK = "kdm"          # age_years, so every convention is legal for it
SLOPE = 0.6            # a clock pulled toward its training mean
SHIFT = 3.0            # what the study carries beyond the reference
INTERCEPT = {"F": 20.0, "M": 23.0}


def _result(scores: dict, obs: pd.DataFrame, weights: dict | None = None):
    manifest = RunManifest()
    manifest.weights = weights or {}
    return fa.FalconResult(scores=pd.DataFrame(scores, index=obs.index), obs=obs,
                           manifest=manifest, registry=fa.registry.load())


def _population(n: int, lo: float, hi: float, shift: float, seed: int,
                prefix: str, sexes=("F", "M"), noise: float = 1.0):
    rng = np.random.default_rng(seed)
    age = rng.uniform(lo, hi, n)
    sex = np.array([sexes[i % len(sexes)] for i in range(n)])
    b0 = np.array([INTERCEPT[s] for s in sex])
    y = b0 + SLOPE * age + shift + rng.normal(0.0, noise, n)
    obs = pd.DataFrame({"age": age, "sex": sex},
                       index=[f"{prefix}{i:04d}" for i in range(n)])
    return _result({CLOCK: y}, obs)


@pytest.fixture
def reference():
    return _population(3000, 30.0, 80.0, 0.0, seed=11, prefix="R")


@pytest.fixture
def study():
    return _population(400, 60.0, 75.0, SHIFT, seed=12, prefix="S")


# ---------------------------------------------------------------------------
# the answer
# ---------------------------------------------------------------------------
def test_reference_recovers_a_known_shift_the_within_cohort_residual_cannot(
        reference, study):
    by_ref = fa.acceleration(study, method="reference", reference=reference)[CLOCK]
    by_res = fa.acceleration(study, method="residual")[CLOCK]
    by_abs = fa.acceleration(study, method="absolute")[CLOCK]

    # Noise SD 1 on 400 samples: the mean is within about 0.05 of the truth,
    # plus the reference line's own error, smaller still on 3,000.
    assert by_ref.mean() == pytest.approx(SHIFT, abs=0.25)
    assert by_ref.groupby(study.obs["sex"]).mean().to_numpy() == pytest.approx(
        [SHIFT, SHIFT], abs=0.3)
    # Centred on the cohort, so the shift the whole cohort shares is gone.
    assert abs(by_res.mean()) < 1e-8
    # Regression toward the mean: every person here is three years "older"
    # than the reference, and predicted minus chronological says younger.
    assert by_abs.mean() < -1.0


def test_the_residual_is_the_reference_line_applied_to_the_study(reference, study):
    """Exactly score minus a line fitted by np.polyfit in the reference stratum."""
    acc = fa.acceleration(study, method="reference", reference=reference)[CLOCK]
    ref_obs, ref_y = reference.obs, reference.scores[CLOCK]
    for sex in ("F", "M"):
        r = ref_obs["sex"] == sex
        slope, intercept = np.polyfit(ref_obs.loc[r, "age"], ref_y[r], 1)
        s = study.obs["sex"] == sex
        expected = study.scores.loc[s, CLOCK] - (slope * study.obs.loc[s, "age"] + intercept)
        assert acc[s].to_numpy() == pytest.approx(expected.to_numpy(), abs=1e-10)


def test_matching_on_sex_fits_each_sex_its_own_line(reference, study):
    """Pooled over sex, the line splits the intercept gap between the sexes and
    hands half of it to each as acceleration. Matched, both get the true shift."""
    pooled = fa.acceleration(study, method="reference", reference=reference,
                             match=("age",))[CLOCK]
    matched = fa.acceleration(study, method="reference", reference=reference)[CLOCK]
    sex = study.obs["sex"]

    gap = INTERCEPT["M"] - INTERCEPT["F"]
    assert (pooled[sex == "M"].mean() - pooled[sex == "F"].mean()) == pytest.approx(
        gap, abs=0.4)
    assert (matched[sex == "M"].mean() - matched[sex == "F"].mean()) == pytest.approx(
        0.0, abs=0.4)


def test_a_study_used_as_its_own_reference_is_within_group_by_sex(study):
    """The two conventions differ only in whose data the line comes from."""
    own = fa.acceleration(study, method="reference", reference=study)[CLOCK]
    within = fa.acceleration(study, method="within_group", group="sex")[CLOCK]
    assert own.to_numpy() == pytest.approx(within.to_numpy(), abs=1e-10)


def test_the_fit_is_recorded_on_the_frame(reference, study):
    acc = fa.acceleration(study, method="reference", reference=reference)
    assert acc.attrs["method"] == "reference"
    assert acc.attrs["match"] == ["age", "sex"]
    assert acc.attrs["adjusted_for"] == []
    fits = {f["stratum"]: f for f in acc.attrs["reference_fit"]}
    assert set(fits) == {"sex=f", "sex=m"}
    for sex, f in fits.items():
        assert f["clock"] == CLOCK
        assert f["n_reference"] == 1500
        assert f["slope"] == pytest.approx(SLOPE, abs=0.02)
        assert f["resid_sd"] == pytest.approx(1.0, abs=0.1)
        assert 30.0 <= f["age_min"] < f["age_max"] <= 80.0


def test_levels_are_compared_as_text_ignoring_case(reference, study):
    lower = study.obs.assign(sex=study.obs["sex"].str.lower())
    relabelled = _result({CLOCK: study.scores[CLOCK]}, lower)
    a = fa.acceleration(study, method="reference", reference=reference)[CLOCK]
    b = fa.acceleration(relabelled, method="reference", reference=reference)[CLOCK]
    assert a.to_numpy() == pytest.approx(b.to_numpy())


def test_a_missing_match_value_gives_nan_for_that_sample_only(reference, study):
    obs = study.obs.copy()
    obs.iloc[0, obs.columns.get_loc("sex")] = np.nan
    acc = fa.acceleration(_result({CLOCK: study.scores[CLOCK]}, obs),
                          method="reference", reference=reference)[CLOCK]
    assert np.isnan(acc.iloc[0])
    assert acc.iloc[1:].notna().all()


def test_reference_through_the_scoring_pipeline(synthetic_clinical):
    """KDM fitted once, both populations scored with it by fa.score."""
    x = synthetic_clinical.X
    fitted = clinical.fit_kdm(x, ["albumin", "creatinine", "glucose", "crp",
                                  "lymphocyte_percent", "mean_cell_volume",
                                  "red_cell_distribution_width",
                                  "alkaline_phosphatase"])
    ref_res = fa.score(synthetic_clinical, clocks=["phenoage", "kdm"], reference=fitted)
    old = x.index[(x["age"] >= 60) & (x["age"] <= 80)]
    study_data = fa.FalconData(X=x.loc[old], obs=synthetic_clinical.obs.loc[old],
                               modality="clinical_chemistry", units=synthetic_clinical.units)
    study_res = fa.score(study_data, clocks=["phenoage", "kdm"], reference=fitted)

    acc = fa.acceleration(study_res, method="reference", reference=ref_res)
    assert list(acc.columns) == ["phenoage", "kdm"]
    for cid in acc.columns:
        for sex in ("F", "M"):
            r = ref_res.obs["sex"] == sex
            slope, intercept = np.polyfit(ref_res.obs.loc[r, "age"],
                                          ref_res.scores.loc[r, cid], 1)
            s = study_res.obs["sex"] == sex
            expected = (study_res.scores.loc[s, cid]
                        - (slope * study_res.obs.loc[s, "age"] + intercept))
            assert acc.loc[s, cid].to_numpy() == pytest.approx(expected.to_numpy())


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------
def test_no_reference_is_refused(study):
    with pytest.raises(AnalysisError, match="needs reference="):
        fa.acceleration(study, method="reference")


def test_a_reference_without_the_clock_is_refused(reference, study):
    other = _result({"phenoage": reference.scores[CLOCK]}, reference.obs)
    with pytest.raises(AnalysisError, match="not scored on kdm"):
        fa.acceleration(study, method="reference", reference=other, clocks=[CLOCK])
    # Not naming clocks does not drop the one the reference lacks.
    with pytest.raises(AnalysisError, match="not scored on kdm"):
        fa.acceleration(study, method="reference", reference=other)


def test_a_reference_without_a_match_column_is_refused(reference, study):
    bare = _result({CLOCK: reference.scores[CLOCK]}, reference.obs.drop(columns="sex"))
    with pytest.raises(AnalysisError, match="the reference has no 'sex' column"):
        fa.acceleration(study, method="reference", reference=bare)


def test_a_study_without_a_match_column_is_refused(reference, study):
    bare = _result({CLOCK: study.scores[CLOCK]}, study.obs.drop(columns="sex"))
    with pytest.raises(AnalysisError, match="this result has no 'sex' column"):
        fa.acceleration(bare, method="reference", reference=reference)


def test_match_must_include_age(reference, study):
    with pytest.raises(AnalysisError, match="does not include 'age'"):
        fa.acceleration(study, method="reference", reference=reference, match=("sex",))


def test_ages_outside_the_reference_are_refused_with_the_range(study):
    narrow = _population(600, 40.0, 60.0, 0.0, seed=13, prefix="N")
    with pytest.raises(AnalysisError, match="rather than extrapolate") as err:
        fa.acceleration(study, method="reference", reference=narrow)
    ages = narrow.obs["age"][narrow.obs["sex"] == "F"]
    assert f"{ages.min():.4g} to {ages.max():.4g}" in str(err.value)


def test_a_stratum_the_reference_lacks_is_refused(study):
    women = _population(600, 30.0, 80.0, 0.0, seed=14, prefix="W", sexes=("F",))
    with pytest.raises(AnalysisError, match="no scored sample with sex=m"):
        fa.acceleration(study, method="reference", reference=women)


def test_differently_coded_sex_is_refused_not_guessed(reference, study):
    coded = reference.obs.assign(sex=reference.obs["sex"].map({"M": 1, "F": 2}))
    other = _result({CLOCK: reference.scores[CLOCK]}, coded)
    with pytest.raises(AnalysisError, match="recode one side"):
        fa.acceleration(study, method="reference", reference=other)


def test_a_stratum_too_small_to_fit_a_line_is_refused(study):
    tiny = _population(4, 30.0, 80.0, 0.0, seed=15, prefix="T")
    obs = tiny.obs.copy()
    obs["age"] = [30.0, 80.0, 31.0, 79.0]
    tiny = _result({CLOCK: tiny.scores[CLOCK]}, obs)
    with pytest.raises(AnalysisError, match="needs at least 3"):
        fa.acceleration(study, method="reference", reference=tiny)


def test_different_coefficients_are_refused(reference, study):
    a = _result({CLOCK: study.scores[CLOCK]}, study.obs,
                weights={CLOCK: {"sha256": "a" * 64}})
    b = _result({CLOCK: reference.scores[CLOCK]}, reference.obs,
                weights={CLOCK: {"sha256": "b" * 64}})
    with pytest.raises(AnalysisError, match="different coefficients"):
        fa.acceleration(a, method="reference", reference=b)


def test_adjust_is_refused_with_reference(reference, study):
    obs = study.obs.assign(mono=np.linspace(0, 1, len(study.obs)))
    s = _result({CLOCK: study.scores[CLOCK]}, obs)
    with pytest.raises(AnalysisError, match="adjust= needs method='residual'"):
        fa.acceleration(s, method="reference", reference=reference, adjust=["mono"])


@pytest.mark.parametrize("cid", ["hd", "yingdamage", "dunedinpoam38"])
def test_scales_without_acceleration_are_refused(reference, study, cid):
    """HD has no age axis, a pace is a rate, and an age_years_relative clock
    has no origin that travels between datasets. The last admits the
    within-cohort residual and must still be refused a reference line."""
    reg = fa.registry.load()
    assert "acceleration" not in reg.get(cid).legal_operations
    s = _result({cid: study.scores[CLOCK]}, study.obs)
    r = _result({cid: reference.scores[CLOCK]}, reference.obs)
    with pytest.raises(IllegalOperationError):
        fa.acceleration(s, method="reference", reference=r, clocks=[cid])


def test_a_relative_clock_keeps_its_within_cohort_residual(study):
    s = _result({"yingdamage": study.scores[CLOCK]}, study.obs)
    assert fa.acceleration(s, method="residual", clocks=["yingdamage"]).shape == (400, 1)
