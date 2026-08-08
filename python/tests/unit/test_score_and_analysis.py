"""Scoring, the manifest, and the downstream statistics."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import falconage as fa
from falconage.core.errors import (
    FeatureCoverageError,
    IllegalOperationError,
    WeightsUnavailableError,
)


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def test_score_compatible_runs_the_tier_a_clocks(synthetic_betas):
    res = fa.score(synthetic_betas, clocks="compatible")
    assert res.scores.shape[0] == synthetic_betas.n_samples
    assert res.scores.shape[1] >= 15
    assert res.scores.notna().all().all()
    assert set(res.scores.columns) >= {"horvath2013", "hannum", "dnamphenoage"}


def test_score_records_a_complete_manifest(synthetic_betas, tmp_path):
    res = fa.score(synthetic_betas, clocks=["horvath2013", "hannum"])
    m = res.manifest
    assert m.falconage_version == fa.__version__
    assert m.dtype == "float64"
    assert set(m.weights) == {"horvath2013", "hannum"}
    assert len(m.weights["horvath2013"]["sha256"]) == 64
    assert m.finished_utc

    p = m.write(tmp_path / "run_manifest.json")
    doc = json.loads(p.read_text())
    assert doc["weights"]["horvath2013"]["provenance"].startswith("Horvath 2013")


def test_explicit_scaffold_request_raises_rather_than_skipping(synthetic_betas):
    """A named clock that cannot run is an error; a silent skip leaves a column
    quietly missing from the results table."""
    with pytest.raises(WeightsUnavailableError):
        fa.score(synthetic_betas, clocks=["grimage2"])


def test_compatible_skips_scaffolds_and_says_why(synthetic_betas):
    res = fa.score(synthetic_betas, clocks="compatible")
    assert "grimage2" in res.skipped
    assert "scaffold" in res.skipped["grimage2"]


def test_coverage_floor_is_enforced(synthetic_betas):
    thin = synthetic_betas.subset(features=list(synthetic_betas.features[:200]))
    with pytest.raises(FeatureCoverageError, match="below the"):
        fa.score(thin, clocks=["horvath2013"], min_coverage=0.9)


def test_coverage_is_recorded_per_clock(synthetic_betas):
    res = fa.score(synthetic_betas, clocks=["horvath2013"])
    cov = res.coverage["horvath2013"]
    assert cov["coverage"] == pytest.approx(1.0)
    assert cov["n_imputed"] == 0


def test_long_form_carries_the_scale(synthetic_betas):
    res = fa.score(synthetic_betas, clocks=["horvath2013", "dnamtl"])
    lf = res.long()
    assert set(lf["scale_type"]) == {"age_years", "telomere_kb"}
    assert lf.shape[0] == synthetic_betas.n_samples * 2


def test_results_round_trip_to_disk(synthetic_betas, tmp_path):
    res = fa.score(synthetic_betas, clocks=["horvath2013", "hannum"])
    written = res.write(tmp_path)
    assert {"scores", "scores_wide", "qc", "manifest"} <= set(written)
    back = pd.read_csv(written["scores_wide"], index_col=0)
    np.testing.assert_allclose(back["horvath2013"].to_numpy(),
                               res.scores["horvath2013"].to_numpy(), rtol=1e-12)


def test_h5ad_round_trip_preserves_everything(synthetic_betas, tmp_path):
    pytest.importorskip("anndata")
    p = synthetic_betas.write_h5ad(tmp_path / "d.h5ad")
    back = fa.FalconData.read_h5ad(p)
    assert back.modality == synthetic_betas.modality
    assert back.platform == synthetic_betas.platform
    np.testing.assert_allclose(back.X.to_numpy(), synthetic_betas.X.to_numpy(), rtol=1e-12)


# ---------------------------------------------------------------------------
# acceleration
# ---------------------------------------------------------------------------
def test_the_three_acceleration_conventions_differ(synthetic_betas):
    res = fa.score(synthetic_betas, clocks=["horvath2013"])
    absolute = fa.acceleration(res, method="absolute")["horvath2013"]
    residual = fa.acceleration(res, method="residual")["horvath2013"]
    assert residual.mean() == pytest.approx(0.0, abs=1e-9), "residuals centre at zero"
    assert not np.allclose(absolute, residual)


def test_within_group_acceleration_centres_each_group(synthetic_betas):
    res = fa.score(synthetic_betas, clocks=["horvath2013"])
    acc = fa.acceleration(res, method="within_group", group="condition")
    for _, g in acc["horvath2013"].groupby(res.obs["condition"]):
        assert g.mean() == pytest.approx(0.0, abs=1e-8)


def test_acceleration_refuses_a_pace_clock_when_named(synthetic_betas):
    res = fa.score(synthetic_betas, clocks=["dunedinpoam38"])
    with pytest.raises(IllegalOperationError, match="already a rate"):
        fa.acceleration(res, clocks=["dunedinpoam38"])


def test_acceleration_filters_silently_when_not_named(synthetic_betas):
    """Not naming clocks means 'the ones this makes sense for'."""
    res = fa.score(synthetic_betas, clocks=["horvath2013", "dunedinpoam38"])
    acc = fa.acceleration(res)
    assert list(acc.columns) == ["horvath2013"]


def test_acceleration_needs_an_age_column(synthetic_betas):
    from falconage.core.errors import AnalysisError

    d = fa.FalconData(X=synthetic_betas.X, obs=pd.DataFrame(index=synthetic_betas.X.index),
                      modality="dna_methylation")
    res = fa.score(d, clocks=["horvath2013"])
    with pytest.raises(AnalysisError, match="needs chronological age"):
        fa.acceleration(res)


# ---------------------------------------------------------------------------
# association, survival, reliability, benchmark
# ---------------------------------------------------------------------------
def test_associate_returns_bh_corrected_p(synthetic_betas):
    res = fa.score(synthetic_betas, clocks=["horvath2013", "hannum", "lin"])
    out = fa.associate(res, outcome="age", covariates=())
    assert set(out.columns) >= {"beta", "se", "p", "q"}
    assert (out["q"] >= out["p"] - 1e-12).all(), "BH q is never below its p"


def test_cox_hazard_finds_a_planted_signal(rng):
    """A synthetic survival dataset where the score genuinely predicts the event."""
    n = 300
    score = rng.normal(size=n)
    t = rng.exponential(scale=np.exp(-0.8 * score))
    e = (t < np.quantile(t, 0.7)).astype(int)
    obs = pd.DataFrame({"time": t, "event": e}, index=[f"s{i}" for i in range(n)])

    class Stub:
        pass

    r = Stub()
    r.scores = pd.DataFrame({"planted": score}, index=obs.index)
    r.obs = obs
    out = fa.cox_hazard(r, time_col="time", event_col="event")
    assert out.loc["planted", "hr"] > 1.2
    assert out.loc["planted", "p"] < 1e-4


def test_icc_is_one_for_identical_repeats():
    df = pd.DataFrame({"subject": ["a", "a", "b", "b", "c", "c"],
                       "value": [1.0, 1.0, 5.0, 5.0, 9.0, 9.0]})
    assert fa.icc(df, "subject", "value") == pytest.approx(1.0, abs=1e-9)


def test_icc_is_near_zero_when_repeats_are_noise(rng):
    df = pd.DataFrame({"subject": np.repeat(list("abcdefghij"), 2),
                       "value": rng.normal(size=20)})
    assert fa.icc(df, "subject", "value") < 0.5


def test_pool_icc_uses_fisher_z():
    from falconage.analysis import pool_icc

    # tanh(mean(atanh([0.9, 0.5]))) is above the plain mean of 0.7
    assert pool_icc([0.9, 0.5]) > 0.7


def test_benchmark_detects_a_planted_acceleration(synthetic_betas):
    """Shift the case group's methylation towards older values and the AA2 test
    must find it."""
    X = synthetic_betas.X.copy()
    case = synthetic_betas.obs["condition"] == "CASE"
    reg = fa.registry.load()
    feats, coefs = reg.coefficients("horvath2013")
    for f, c in zip(feats, coefs):
        if f in X.columns and abs(c) > 1e-6:
            X.loc[case, f] = np.clip(X.loc[case, f] + 0.06 * np.sign(c), 0.001, 0.999)

    d = fa.FalconData(X=X, obs=synthetic_betas.obs, modality="dna_methylation",
                      platform="450K")
    res = fa.score(d, clocks=["horvath2013", "hannum"])
    b = fa.run_benchmark(res, condition_col="condition", control="HC",
                         dataset_col="dataset")
    assert b.summary().loc["horvath2013", "AA2"] == 1
    row = b.per_dataset.query("clock == 'horvath2013'").iloc[0]
    assert row["delta"] > 0 and row["q"] < 0.05


def test_benchmark_total_discounts_a_biased_clock():
    """The MedE discount is what stops a clock that over-predicts everybody from
    sweeping AA1. Checked as arithmetic on the published formula."""
    aa2, aa1, medae, mede = 3, 4, 8.0, 4.0
    total = aa2 + aa1 * (1 - max(0.0, mede) / medae)
    assert total == pytest.approx(5.0)
    # a clock with no bias keeps the whole AA1 credit
    assert aa2 + aa1 * (1 - max(0.0, -4.0) / medae) == pytest.approx(7.0)


def test_benchmark_excludes_non_age_scales(synthetic_betas):
    res = fa.score(synthetic_betas, clocks=["horvath2013", "dnamtl", "zhangmortality"])
    b = fa.run_benchmark(res, condition_col="condition", control="HC")
    assert "dnamtl" not in b.summary().index
    assert "zhangmortality" not in b.summary().index


def test_agreement_is_a_square_correlation_matrix(synthetic_betas):
    res = fa.score(synthetic_betas, clocks=["horvath2013", "hannum", "lin"])
    m = fa.agreement(res)
    assert m.shape == (3, 3)
    np.testing.assert_allclose(np.diag(m), 1.0)


def test_combine_keeps_per_dataset_coverage(synthetic_betas):
    a = fa.score(synthetic_betas, clocks=["horvath2013"])
    thin = synthetic_betas.subset(features=list(synthetic_betas.features)[:-100])
    b = fa.score(thin, clocks=["horvath2013"], min_coverage=0.5)
    c = fa.combine([a, b], keys=["full", "thin"])
    assert c.scores.shape[0] == a.scores.shape[0] * 2
    assert c.coverage["horvath2013"]["coverage"] <= a.coverage["horvath2013"]["coverage"]


# ---------------------------------------------------------------------------
# cell-composition adjustment
# ---------------------------------------------------------------------------

def test_cell_composition_is_empty_without_deconvolution_clocks(synthetic_betas):
    """Absence of an adjustment is data, not an exception."""
    res = fa.score(synthetic_betas, clocks=["horvath2013", "hannum"])
    assert fa.cell_composition(res).empty


def test_adjust_needs_something_to_adjust_with(synthetic_betas):
    from falconage.core.errors import AnalysisError

    res = fa.score(synthetic_betas, clocks=["horvath2013"])
    with pytest.raises(AnalysisError, match="deconvolution clocks"):
        fa.acceleration(res, adjust="cell_composition")


def test_adjust_is_refused_on_methods_that_cannot_honour_it(synthetic_betas):
    from falconage.core.errors import AnalysisError

    res = fa.score(synthetic_betas, clocks=["horvath2013"])
    with pytest.raises(AnalysisError, match="needs method='residual'"):
        fa.acceleration(res, method="absolute", adjust=["age"])


def test_adjusting_removes_the_covariate_it_is_given(synthetic_betas):
    """The property that makes this worth having.

    Build a covariate that the score genuinely depends on, then confirm the
    adjusted acceleration is uncorrelated with it while the unadjusted one is
    not. This is the confounding the deconvolution clocks exist to remove.
    """
    res = fa.score(synthetic_betas, clocks=["horvath2013"])
    rng = np.random.default_rng(20260808)

    y = res.scores["horvath2013"]
    confounder = 0.4 * (y - y.mean()) / y.std() + rng.normal(0, 1, len(y))
    res.obs["mono"] = confounder.to_numpy()

    plain = fa.acceleration(res)["horvath2013"]
    fixed = fa.acceleration(res, adjust=["mono"])["horvath2013"]

    r_plain = abs(np.corrcoef(plain, confounder)[0, 1])
    r_fixed = abs(np.corrcoef(fixed, confounder)[0, 1])
    assert r_plain > 0.2, "the confounder was not actually confounding"
    assert r_fixed < 1e-8, "regressing it out should leave no correlation"


def test_the_adjustment_is_recorded_on_the_frame(synthetic_betas):
    """An adjusted acceleration is a different quantity and must say so."""
    res = fa.score(synthetic_betas, clocks=["horvath2013"])
    res.obs["mono"] = np.linspace(0.1, 0.3, res.scores.shape[0])
    assert fa.acceleration(res).attrs["adjusted_for"] == []
    assert fa.acceleration(res, adjust=["mono"]).attrs["adjusted_for"] == ["mono"]


def test_interpretation_carries_scale_reliability_and_caveats(synthetic_betas):
    """1.6: what a reader needs is in the object, not only on the website."""
    res = fa.score(synthetic_betas, clocks=["horvath2013", "yingcausage"],
                   min_coverage=0.0)
    t = res.interpretation()
    assert t.loc["horvath2013", "scale_type"] == "age_years"
    assert "acceleration" in t.loc["horvath2013", "legal_operations"]
    # The paper-vs-coefficients disagreement travels with the number.
    assert "586" in t.loc["yingcausage", "caveats"]
    assert "not a diagnostic" in fa.FalconResult.CAVEAT


# ---------------------------------------------------------------------------
# probe loss, before scoring
# ---------------------------------------------------------------------------

def test_probe_loss_names_tier_b_and_c_instead_of_omitting_them(synthetic_betas):
    """A clock silently absent from this table reads as 'fine'."""
    t = fa.probe_loss(synthetic_betas, clocks=["horvath2013", "grimage2"])
    assert set(t.index) == {"horvath2013", "grimage2"}
    assert t.loc["grimage2", "heaviest_absent"] == "coefficients not available"
    assert np.isnan(t.loc["grimage2", "mass_coverage"])


def test_probe_loss_separates_count_from_weight(synthetic_betas):
    """The EPICv2 case: same probes lost by count, very different by weight."""
    reg = fa.registry.load()
    feats, coefs = reg.coefficients("horvath2013")
    heaviest = [f for _, f in sorted(zip(np.abs(coefs), feats), reverse=True)[:12]]

    X = synthetic_betas.X.drop(columns=heaviest, errors="ignore")
    d = fa.FalconData(X=X, obs=synthetic_betas.obs,
                      modality="dna_methylation", platform="450K")

    row = fa.probe_loss(d, clocks=["horvath2013"]).loc["horvath2013"]
    assert row["coverage"] > row["mass_coverage"], (
        "dropping the heaviest probes must cost more weight than count")
    assert row["heaviest_absent"]


def test_probe_loss_is_ordered_worst_first(synthetic_betas):
    t = fa.probe_loss(synthetic_betas, clocks="scoreable")
    mass = t["mass_coverage"].dropna().to_numpy()
    assert (np.diff(mass) >= -1e-12).all(), "should ascend from worst"


def test_acceleration_both_returns_the_two_conventions_side_by_side(synthetic_betas):
    """Documented in six places before it existed; now it exists."""
    res = fa.score(synthetic_betas, clocks=["horvath2013", "hannum"])
    both = fa.acceleration(res, method="both")

    for cid in ("horvath2013", "hannum"):
        assert f"{cid}_absolute" in both.columns
        assert f"{cid}_residual" in both.columns

    plain_abs = fa.acceleration(res, method="absolute")["horvath2013"]
    plain_res = fa.acceleration(res, method="residual")["horvath2013"]
    np.testing.assert_allclose(both["horvath2013_absolute"], plain_abs, rtol=0, atol=0)
    np.testing.assert_allclose(both["horvath2013_residual"], plain_res, rtol=0, atol=0)

    # They are genuinely different, which is the reason to show both.
    assert not np.allclose(both["horvath2013_absolute"],
                           both["horvath2013_residual"])


def test_an_unknown_method_names_the_ones_that_work(synthetic_betas):
    from falconage.core.errors import AnalysisError

    res = fa.score(synthetic_betas, clocks=["horvath2013"])
    with pytest.raises(AnalysisError, match="both"):
        fa.acceleration(res, method="typo")
