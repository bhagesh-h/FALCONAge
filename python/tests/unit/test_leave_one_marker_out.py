"""Which marker of a clinical clock carries an association.

Two kinds of known answer. For PhenoAge, whose weights are fixed, holding a
marker at a reference mean moves every score by exactly its weight times the
distance to that mean, divided by the Gompertz inversion's 0.090165, because
PhenoAge is affine in its linear predictor; a reference whose mean differs by a
chosen amount therefore gives a mean change that can be written down in
advance. For KDM and HD, which are refitted, each row must equal a refit done
by hand. Then the question the function exists for: an outcome built from one
marker alone must be attributed to that marker.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import falconage as fa
from falconage.core.errors import AnalysisError
from falconage.core.manifest import RunManifest
from falconage.models import clinical

MARKERS = ["albumin", "creatinine", "glucose", "crp", "lymphocyte_percent",
           "mean_cell_volume", "red_cell_distribution_width",
           "alkaline_phosphatase", "white_blood_cell_count"]
#: Years of PhenoAge per unit of the linear predictor: the Gompertz inversion.
PER_XB = 1.0 / 0.090165


def _with_outcome(data, marker: str, seed: int, noise: float = 0.5):
    """A copy of the cohort whose ``outcome`` is built from one marker only.

    From the marker's residual on age, so the outcome is unrelated to age and
    to every other marker, and the only way a clock can predict it is through
    this marker.
    """
    x = data.X
    slope, intercept = np.polyfit(x["age"], x[marker], 1)
    resid = x[marker] - (slope * x["age"] + intercept)
    rng = np.random.default_rng(seed)
    obs = data.obs.copy()
    obs["outcome"] = resid / resid.std() + rng.normal(0.0, noise, len(x))
    return fa.FalconData(X=x, obs=obs, modality=data.modality, units=data.units)


def _result(scores: pd.Series, obs: pd.DataFrame, clock: str):
    return fa.FalconResult(scores=pd.DataFrame({clock: scores}), obs=obs,
                           manifest=RunManifest(), registry=fa.registry.load())


# ---------------------------------------------------------------------------
# PhenoAge: a marker held at the reference mean
# ---------------------------------------------------------------------------
def test_phenoage_moves_by_the_weight_times_the_distance_to_the_reference_mean(
        synthetic_clinical):
    res = fa.score(synthetic_clinical, clocks=["phenoage"])
    x = synthetic_clinical.X
    # A reference whose glucose is 0.4 mmol/L higher and whose CRP is 1.5 times
    # as high, and otherwise the same people. Holding glucose at the reference
    # mean raises every study glucose by 0.4 on average, so PhenoAge rises by
    # 0.1953 * 0.4 / 0.090165 years; CRP enters as log, so by
    # 0.0954 * log(1.5) / 0.090165.
    ref = x.assign(glucose=x["glucose"] + 0.4, crp=x["crp"] * 1.5)
    tab = fa.leave_one_marker_out(res, "phenoage", data=synthetic_clinical, reference=ref)

    coef = clinical.PHENOAGE_COEF
    assert tab.loc["glucose", "mean_change"] == pytest.approx(
        coef["glucose"] * 0.4 * PER_XB, rel=1e-6)
    assert tab.loc["crp", "mean_change"] == pytest.approx(
        coef["log_crp"] * np.log(1.5) * PER_XB, rel=1e-6)
    for m in ("albumin", "creatinine", "mean_cell_volume"):
        assert tab.loc[m, "mean_change"] == pytest.approx(0.0, abs=1e-9)
        spread = np.abs(coef[m] * (x[m] - x[m].mean())) * PER_XB
        assert tab.loc[m, "mean_abs_change"] == pytest.approx(spread.mean(), rel=1e-6)

    assert tab.attrs["held_at"]["glucose"] == pytest.approx(x["glucose"].mean() + 0.4)
    assert tab.attrs["held_at"]["crp"] == pytest.approx(
        1.5 * float(np.exp(np.log(x["crp"]).mean())))


def test_phenoage_against_its_own_cohort_changes_nothing_on_average(synthetic_clinical):
    res = fa.score(synthetic_clinical, clocks=["phenoage"])
    tab = fa.leave_one_marker_out(res, "phenoage", data=synthetic_clinical,
                                  reference=synthetic_clinical.X)
    assert list(tab.index[:1]) == [fa.analysis.FULL_PANEL]
    assert set(tab.index[1:]) == set(MARKERS)
    assert tab["mean_change"].to_numpy() == pytest.approx(np.zeros(len(tab)), abs=1e-9)
    assert (tab.loc[MARKERS, "r_spearman"] < 1.0).all()


def test_phenoage_attributes_an_outcome_to_the_marker_that_carries_it(synthetic_clinical):
    """Across twenty outcome seeds: full-panel p below 4e-5, glucose first with
    attenuation at least 0.86, no other marker above 0.07."""
    data = _with_outcome(synthetic_clinical, "glucose", seed=41)
    res = fa.score(data, clocks=["phenoage"])
    tab = fa.leave_one_marker_out(res, "phenoage", data=data, reference=data.X,
                                  test="outcome")

    assert tab.loc[fa.analysis.FULL_PANEL, "p"] < 1e-3
    assert tab.index[1] == "glucose"
    assert tab.loc["glucose", "attenuation"] > 0.7
    assert tab.loc[fa.analysis.FULL_PANEL, "partial_r"] > 0.2
    others = tab.drop(index=[fa.analysis.FULL_PANEL, "glucose"])
    assert (others["attenuation"] < 0.3).all()
    # Holding glucose fixed while it carries the outcome leaves nothing to find.
    assert tab.loc["glucose", "p"] > 0.01


def test_phenoage_can_test_a_chosen_subset(synthetic_clinical):
    res = fa.score(synthetic_clinical, clocks=["phenoage"])
    tab = fa.leave_one_marker_out(res, "phenoage", data=synthetic_clinical,
                                  reference=synthetic_clinical.X,
                                  markers=["crp", "glucose"])
    assert set(tab.index) == {fa.analysis.FULL_PANEL, "crp", "glucose"}


# ---------------------------------------------------------------------------
# KDM and HD: refitted without the marker
# ---------------------------------------------------------------------------
def test_kdm_rows_are_refits_on_the_remaining_markers(synthetic_clinical):
    x = synthetic_clinical.X
    res = fa.score(synthetic_clinical, clocks=["kdm"],
                   reference=clinical.fit_kdm(x, MARKERS))
    tab = fa.leave_one_marker_out(res, "kdm", data=synthetic_clinical, reference=x,
                                  markers=MARKERS)

    full = clinical.kdm(x, clinical.fit_kdm(x, MARKERS))
    for m in ("glucose", "red_cell_distribution_width"):
        rest = [k for k in MARKERS if k != m]
        by_hand = clinical.kdm(x, clinical.fit_kdm(x, rest))
        assert tab.loc[m, "mean_change"] == pytest.approx(float((by_hand - full).mean()))
        assert tab.loc[m, "r_spearman"] == pytest.approx(
            float(full.corr(by_hand, method="spearman")))
    assert tab.attrs["removal"] == "refitted on the remaining markers"


def test_kdm_attributes_an_outcome_to_the_marker_that_carries_it(synthetic_clinical):
    """Creatinine has the largest |k/s| in this cohort, so KDM carries it most.

    Across twenty outcome seeds the full-panel p stayed below 1e-5, creatinine
    was ranked first every time with attenuation at least 0.73, and no other
    marker exceeded 0.11; the thresholds below leave room on each side.
    """
    data = _with_outcome(synthetic_clinical, "creatinine", seed=42)
    x = data.X
    res = fa.score(data, clocks=["kdm"], reference=clinical.fit_kdm(x, MARKERS))
    tab = fa.leave_one_marker_out(res, "kdm", data=data, reference=x,
                                  markers=MARKERS, test="outcome")

    assert tab.loc[fa.analysis.FULL_PANEL, "p"] < 1e-4
    assert tab.index[1] == "creatinine"
    assert tab.loc["creatinine", "attenuation"] > 0.6
    assert tab.loc["creatinine", "p"] > 0.01
    others = tab.drop(index=[fa.analysis.FULL_PANEL, "creatinine"])
    assert (others["attenuation"] < 0.3).all()


def test_kdm_fitted_by_sex_is_refitted_by_sex(synthetic_clinical):
    """A result scored against one fit per sex, as BioAge's kdm0 is."""
    x = synthetic_clinical.X.join(synthetic_clinical.obs[["sex"]])
    by_sex = pd.Series(np.nan, index=x.index)
    for sex in ("F", "M"):
        rows = x[x["sex"] == sex]
        by_sex[rows.index] = clinical.kdm(rows, clinical.fit_kdm(rows, MARKERS))
    res = _result(by_sex, synthetic_clinical.obs, "kdm")

    tab = fa.leave_one_marker_out(res, "kdm", data=synthetic_clinical,
                                  reference=synthetic_clinical, markers=MARKERS,
                                  sex_col="sex")
    rest = [k for k in MARKERS if k != "creatinine"]
    by_hand = pd.Series(np.nan, index=x.index)
    for sex in ("F", "M"):
        rows = x[x["sex"] == sex]
        by_hand[rows.index] = clinical.kdm(rows, clinical.fit_kdm(rows, rest))
    assert tab.loc["creatinine", "mean_change"] == pytest.approx(
        float((by_hand - by_sex).mean()))

    # The same result read as if it had one pooled fit is a different clock.
    with pytest.raises(AnalysisError, match="does not reproduce"):
        fa.leave_one_marker_out(res, "kdm", data=synthetic_clinical,
                                reference=synthetic_clinical, markers=MARKERS)


def test_hd_reports_ranks_but_not_differences(synthetic_clinical):
    """A Mahalanobis distance shrinks with every dimension removed, so its raw
    change measures the panel size; relative_score admits rank and correlation
    only, and the table follows the scale."""
    x = synthetic_clinical.X
    res = fa.score(synthetic_clinical, clocks=["hd"], reference=clinical.fit_hd(x, MARKERS))
    tab = fa.leave_one_marker_out(res, "hd", data=synthetic_clinical, reference=x,
                                  markers=MARKERS)
    assert tab["mean_change"].isna().all()
    assert tab["mean_abs_change"].isna().all()
    assert tab.loc[MARKERS, "r_spearman"].between(-1.0, 0.9999).all()

    full = clinical.hd(x, clinical.fit_hd(x, MARKERS))
    rest = [k for k in MARKERS if k != "albumin"]
    by_hand = clinical.hd(x, clinical.fit_hd(x, rest))
    assert tab.loc["albumin", "r_spearman"] == pytest.approx(
        float(full.corr(by_hand, method="spearman")))


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------
def test_a_clock_that_is_not_clinical_is_refused(synthetic_clinical):
    res = _result(synthetic_clinical.X["age"], synthetic_clinical.obs, "horvath2013")
    with pytest.raises(AnalysisError, match="defined for the clinical clocks"):
        fa.leave_one_marker_out(res, "horvath2013", data=synthetic_clinical,
                                reference=synthetic_clinical.X)


def test_a_clock_not_in_the_result_is_refused(synthetic_clinical):
    res = fa.score(synthetic_clinical, clocks=["phenoage"])
    with pytest.raises(AnalysisError, match="not scored in this result"):
        fa.leave_one_marker_out(res, "kdm", data=synthetic_clinical,
                                reference=synthetic_clinical.X, markers=MARKERS)


def test_no_reference_is_refused_and_says_why(synthetic_clinical):
    res = fa.score(synthetic_clinical, clocks=["phenoage"])
    with pytest.raises(AnalysisError, match="holding it at a constant"):
        fa.leave_one_marker_out(res, "phenoage", data=synthetic_clinical, reference=None)


def test_kdm_without_its_panel_is_refused(synthetic_clinical):
    x = synthetic_clinical.X
    res = fa.score(synthetic_clinical, clocks=["kdm"], reference=clinical.fit_kdm(x, MARKERS))
    with pytest.raises(AnalysisError, match="no fixed panel"):
        fa.leave_one_marker_out(res, "kdm", data=synthetic_clinical, reference=x)


def test_a_reference_other_than_the_scoring_one_is_refused(synthetic_clinical):
    x = synthetic_clinical.X
    res = fa.score(synthetic_clinical, clocks=["kdm"],
                   reference=clinical.fit_kdm(x.iloc[:120], MARKERS))
    with pytest.raises(AnalysisError, match="does not reproduce"):
        fa.leave_one_marker_out(res, "kdm", data=synthetic_clinical, reference=x,
                                markers=MARKERS)


def test_a_missing_test_column_is_refused(synthetic_clinical):
    res = fa.score(synthetic_clinical, clocks=["phenoage"])
    with pytest.raises(AnalysisError, match="no 'plaque' column"):
        fa.leave_one_marker_out(res, "phenoage", data=synthetic_clinical,
                                reference=synthetic_clinical.X, test="plaque")


def test_a_text_test_column_is_refused_rather_than_read_as_missing(synthetic_clinical):
    """associate() reads text as missing; a table of NaN would be the silent result."""
    obs = synthetic_clinical.obs.assign(
        plaque=np.where(np.arange(len(synthetic_clinical.obs)) % 3, "No", "Yes"))
    res = _result(clinical.phenoage(synthetic_clinical.X), obs, "phenoage")
    with pytest.raises(AnalysisError, match="Code it as a number first"):
        fa.leave_one_marker_out(res, "phenoage", data=synthetic_clinical,
                                reference=synthetic_clinical.X, test="plaque")


def test_an_hd_panel_too_small_to_lose_a_marker_is_refused(synthetic_clinical):
    x = synthetic_clinical.X
    two = ["albumin", "creatinine"]
    res = fa.score(synthetic_clinical, clocks=["hd"], reference=clinical.fit_hd(x, two))
    with pytest.raises(AnalysisError, match="at least 3"):
        fa.leave_one_marker_out(res, "hd", data=synthetic_clinical, reference=x,
                                markers=two)


def test_an_unknown_phenoage_marker_is_refused(synthetic_clinical):
    res = fa.score(synthetic_clinical, clocks=["phenoage"])
    with pytest.raises(AnalysisError, match="PhenoAge has no marker 'bmi'"):
        fa.leave_one_marker_out(res, "phenoage", data=synthetic_clinical,
                                reference=synthetic_clinical.X, markers=["bmi"])


def test_a_reference_without_the_marker_is_refused(synthetic_clinical):
    res = fa.score(synthetic_clinical, clocks=["phenoage"])
    with pytest.raises(AnalysisError, match="no mean to hold it at"):
        fa.leave_one_marker_out(res, "phenoage", data=synthetic_clinical,
                                reference=synthetic_clinical.X.drop(columns="crp"))


def test_sex_col_is_refused_for_phenoage_rather_than_ignored(synthetic_clinical):
    res = fa.score(synthetic_clinical, clocks=["phenoage"])
    with pytest.raises(AnalysisError, match="PhenoAge is not refitted"):
        fa.leave_one_marker_out(res, "phenoage", data=synthetic_clinical,
                                reference=synthetic_clinical.X, sex_col="sex")


def test_data_missing_scored_samples_is_refused(synthetic_clinical):
    res = fa.score(synthetic_clinical, clocks=["phenoage"])
    with pytest.raises(AnalysisError, match="not in data="):
        fa.leave_one_marker_out(res, "phenoage", data=synthetic_clinical.X.iloc[10:],
                                reference=synthetic_clinical.X)
