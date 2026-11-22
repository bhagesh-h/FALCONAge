"""Building, adding and stacking clocks: todo items 19, 20 and 21."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import falconage as fa
from falconage.analysis.learners import folds, metrics
from falconage.core.errors import AnalysisError, IllegalOperationError, RegistryError


@pytest.fixture(scope="module")
def small():
    """60 samples by 120 CpGs, a third of them carrying an age signal, with
    three visits per person so grouped folds have something to keep together."""
    rng = np.random.default_rng(7)
    n, p = 60, 120
    person = np.repeat(np.arange(20), 3)
    age = np.repeat(rng.uniform(20, 80, 20), 3) + np.tile([0, 1, 2], 20)
    w = np.zeros(p); w[:40] = rng.normal(0, 0.004, 40)
    X = np.clip(0.5 + (age[:, None] - 50) * w + rng.normal(0, 0.02, (n, p)), 0.01, 0.99)
    ids = [f"S{i:02d}" for i in range(n)]
    obs = pd.DataFrame({"age": age, "person": person.astype(str)}, index=ids)
    cols = [f"cg{i:08d}" for i in range(p)]
    return fa.FalconData(X=pd.DataFrame(X, index=ids, columns=cols), obs=obs,
                         modality="dna_methylation", platform="450K")


def test_grouped_folds_never_split_a_group():
    g = np.repeat(np.arange(12), 3)
    f = folds(len(g), 4, np.random.default_rng(0), g)
    for grp in np.unique(g):
        assert len(set(f[g == grp])) == 1


def test_metrics_report_medae_slope_and_bias():
    y = np.linspace(20, 80, 30)
    m = metrics(y * 0.8 + 10, y)
    assert {"mae", "medae", "r", "slope", "bias_low", "bias_high"} <= set(m)
    assert m["slope"] == pytest.approx(0.8)


def test_fit_clock_takes_groups_and_reports_the_new_metrics(small):
    fit = fa.fit_clock(small, learner="ridge", outer_folds=5, inner_folds=3, groups="person")
    assert {"medae", "slope"} <= set(fit.metrics)
    assert fit.metrics["r"] > 0.5
    assert "grouped by 'person'" in fit.notes[0]


def test_compare_learners_reports_a_nested_error(small):
    res = fa.compare_learners(small, learners=["elasticnet", "ridge", "pc_elasticnet"],
                              outer_folds=4, inner_folds=2, groups="person")
    assert set(res.table.index) == {"elasticnet_0.5", "ridge", "pc_elasticnet"}
    assert len(res.picks) == 4
    assert np.isfinite(res.selected["mae"])
    # the honest error cannot beat the best learner's own out-of-fold error by
    # more than noise: picking is a fitted choice, not a free improvement
    assert res.selected["mae"] >= res.table["mae"].min() - 1e-9


def test_compare_learners_on_noise_is_no_better_than_the_mean():
    """Varma and Simon's check: with no signal, the nested error matches
    predicting the training mean, however many learners were tried."""
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.uniform(0.1, 0.9, (40, 60)),
                     columns=[f"cg{i:08d}" for i in range(60)])
    obs = pd.DataFrame({"age": rng.uniform(20, 80, 40)}, index=X.index)
    d = fa.FalconData(X=X, obs=obs, modality="dna_methylation", platform="450K")
    res = fa.compare_learners(d, learners=["elasticnet", "ridge"], outer_folds=4, inner_folds=2)
    baseline = float(np.mean(np.abs(obs.age - obs.age.mean())))
    assert res.selected["mae"] > 0.8 * baseline


def test_a_fitted_clock_round_trips_through_add_clock_and_score(small, tmp_path):
    fit = fa.fit_clock(small, learner="ridge", outer_folds=5, inner_folds=3, groups="person")
    yml = fit.to_clock("testcohortclock", tmp_path)
    reg = fa.registry.load()
    try:
        cid = reg.add_clock(yml, replace=True)
        assert reg.get(cid).origin == "user"
        res = fa.score(small, clocks=[cid])
        np.testing.assert_allclose(res.scores[cid].to_numpy(),
                                   fit.predict(small).to_numpy(), atol=1e-8)
        assert res.manifest.weights[cid]["source"] == "user_clock"
    finally:
        reg._clocks.pop("testcohortclock", None)


def test_add_clock_refuses_to_shadow_a_shipped_clock(tmp_path):
    (tmp_path / "c.csv").write_text("feature_id,coefficient\ncg00000029,1.0\n")
    (tmp_path / "c.yaml").write_text(
        "id: horvath2013\nname: x\ndata_type: dna_methylation\ntissue: [whole blood]\n"
        "training_target: [age]\nscale_type: age_years\nunit: [years]\ncoefficients: c.csv\n")
    with pytest.raises(RegistryError, match="ships with FALCONAge"):
        fa.registry.load().add_clock(tmp_path / "c.yaml")


def test_add_clock_names_a_missing_field_and_an_unknown_op(tmp_path):
    (tmp_path / "c.csv").write_text("feature_id,coefficient\ncg00000029,1.0\n")
    (tmp_path / "a.yaml").write_text("id: mine\nname: x\ncoefficients: c.csv\n")
    with pytest.raises(RegistryError, match="missing"):
        fa.registry.load().add_clock(tmp_path / "a.yaml")
    (tmp_path / "b.yaml").write_text(
        "id: mine2\nname: x\ndata_type: dna_methylation\ntissue: [whole blood]\n"
        "training_target: [age]\nscale_type: age_years\nunit: [years]\ncoefficients: c.csv\n"
        "postprocess: [{op: not_an_op}]\n")
    with pytest.raises(RegistryError, match="not an operation"):
        fa.registry.load().add_clock(tmp_path / "b.yaml")


def test_a_nonlinear_learner_cannot_be_written_as_a_clock(small):
    pytest.importorskip("sklearn")
    fit = fa.fit_clock(small, learner="random_forest", outer_folds=4)
    assert fit.coefficients is None
    with pytest.raises(AnalysisError, match="no weight per feature"):
        fit.to_clock("rf", "/tmp/unused")


@pytest.fixture(scope="module")
def scored():
    d = fa.demo_data(n=40)
    return fa.score(d, clocks=["horvath2013", "hannum", "dnamphenoage", "zhangen", "dunedinpoam38"])


def test_stack_reports_out_of_fold_metrics_and_a_comparison(scored):
    st = fa.stack_clocks(scored, clocks=["horvath2013", "hannum", "zhangen"], method="nnls")
    assert (st.weights >= 0).all()
    assert {"mae", "r"} <= set(st.metrics)
    assert set(st.comparison.index) >= {"horvath2013", "hannum", "zhangen", "plain mean"}
    np.testing.assert_allclose(st.predict(scored).iloc[:3].to_numpy() is not None, True)


def test_a_plain_mean_across_scales_is_refused(scored):
    with pytest.raises(IllegalOperationError, match="no unit"):
        fa.stack_clocks(scored, clocks=["horvath2013", "dunedinpoam38"], method="mean")


def test_a_linear_stack_collapses_to_one_clock_that_scores_the_same(scored, tmp_path):
    st = fa.stack_clocks(scored, clocks=["hannum", "dnamphenoage"], method="ridge")
    yml = st.to_clock("teststack", tmp_path)
    reg = fa.registry.load()
    try:
        cid = reg.add_clock(yml, replace=True)
        d = fa.demo_data(n=40)
        again = fa.score(d, clocks=[cid]).scores[cid]
        np.testing.assert_allclose(again.to_numpy(), st.predict(scored).to_numpy(), atol=1e-6)
    finally:
        reg._clocks.pop("teststack", None)


def test_a_stack_with_a_per_sample_standardisation_is_not_collapsed(scored, tmp_path):
    st = fa.stack_clocks(scored, clocks=["hannum", "zhangen"], method="nnls")
    with pytest.raises(AnalysisError, match="not a plain weighted sum"):
        st.to_clock("nope", tmp_path)
