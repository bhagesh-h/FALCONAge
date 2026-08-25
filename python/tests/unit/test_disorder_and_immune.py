"""Disorder readouts, repertoire structure, mosaic dispersion, variance split.

Same discipline as test_uncertainty: wherever a closed form exists, the test
asserts against the closed form rather than against a recorded output. Three of
these functions have an exact answer that can be written down -- entropy at a
fixed beta, the Simpson identity behind the clonality simulation, and the
balanced-design variance components -- and those are the ones worth pinning,
because a regression in any of them would otherwise look like a plausible number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import falconage as fa
from falconage.core.errors import AnalysisError, DataError
from falconage.immune import effective_clones, zipf_clone_sizes
from falconage.uncertainty import VarianceError


def _betas(values, n_features=200, n_samples=10, seed=0, obs=None):
    rng = np.random.default_rng(seed)
    if np.isscalar(values):
        X = np.full((n_samples, n_features), float(values))
    else:
        X = np.asarray(values, dtype=float)
    ids = [f"s{i}" for i in range(X.shape[0])]
    feats = [f"cg{i:08d}" for i in range(X.shape[1])]
    frame = pd.DataFrame(X, index=ids, columns=feats)
    return fa.FalconData(
        X=frame,
        obs=pd.DataFrame(obs if obs is not None else {}, index=ids),
        modality="dna_methylation")


# ---------------------------------------------------------------------------
# entropy
# ---------------------------------------------------------------------------

def test_entropy_is_one_at_half():
    """Every site at beta 0.5 is maximum disorder, so S = 1 exactly."""
    out = fa.entropy(_betas(0.5))
    assert np.allclose(out["entropy"], 1.0)


@pytest.mark.parametrize("beta", [0.0, 1.0])
def test_entropy_is_zero_when_fully_committed(beta):
    """A methylome with no undecided site carries no entropy."""
    out = fa.entropy(_betas(beta))
    assert np.allclose(out["entropy"], 0.0, atol=1e-7)


def test_entropy_matches_the_closed_form():
    """S = [b log b + (1-b) log(1-b)] / log(0.5) for a constant matrix."""
    b = 0.2
    expect = (b * np.log(b) + (1 - b) * np.log(1 - b)) / np.log(0.5)
    out = fa.entropy(_betas(b))
    assert np.allclose(out["entropy"], expect)
    assert (out["n_sites"] == 200).all()


def test_entropy_refuses_m_values():
    """M-values pass every shape check and give a number; that is the trap."""
    data = _betas(0.5)
    data.X.iloc[0, 0] = 4.2
    with pytest.raises(DataError, match=r"beta values in \[0, 1\]"):
        fa.entropy(data)


def test_entropy_refuses_non_methylation():
    data = _betas(0.5)
    data.modality = "clinical_chemistry"
    with pytest.raises(DataError, match="not on clinical_chemistry"):
        fa.entropy(data)


def test_complete_sites_default_drops_partially_covered_columns():
    data = _betas(0.5)
    data.X.iloc[0, :50] = np.nan
    assert (fa.entropy(data)["n_sites"] == 150).all()
    loose = fa.entropy(data, complete_sites=False)
    assert loose["n_sites"].iloc[0] == 150 and loose["n_sites"].iloc[1] == 200


# ---------------------------------------------------------------------------
# drift
# ---------------------------------------------------------------------------

def test_drift_is_zero_for_identical_samples():
    """Ten identical samples sit exactly on their own leave-one-out centroid."""
    out = fa.drift(_betas(0.4))
    assert np.allclose(out["drift"], 0.0)


def test_drift_leave_one_out_uses_the_other_samples_only():
    """Three samples, one site: the LOO centre for s0 is mean(s1, s2)."""
    X = np.array([[0.1], [0.5], [0.9]])
    out = fa.drift(_betas(X))
    assert out["drift"].iloc[0] == pytest.approx(abs(0.1 - 0.7))
    assert out["drift"].iloc[1] == pytest.approx(abs(0.5 - 0.5))
    assert out["drift"].iloc[2] == pytest.approx(abs(0.9 - 0.3))


def test_drift_against_a_supplied_reference():
    X = np.array([[0.2], [0.4], [0.6]])
    data = _betas(X)
    ref = pd.Series({data.X.columns[0]: 0.5})
    out = fa.drift(data, reference=ref)
    assert np.allclose(out["drift"], [0.3, 0.1, 0.1])


def test_drift_needs_three_samples_without_a_reference():
    with pytest.raises(AnalysisError, match="at least 3 samples"):
        fa.drift(_betas(np.array([[0.2], [0.4]])))


# ---------------------------------------------------------------------------
# variable_sites and the barometer
# ---------------------------------------------------------------------------

def _widening(n=60, n_sites=40, seed=1):
    """Sites whose spread grows with age, plus flat sites to be rejected."""
    rng = np.random.default_rng(seed)
    age = np.linspace(20, 80, n)
    X = np.empty((n, n_sites))
    for j in range(n_sites):
        scale = (0.002 + 0.06 * (age - 20) / 60) if j < 20 else np.full(n, 0.03)
        X[:, j] = np.clip(0.5 + rng.normal(0, scale), 0, 1)
    return _betas(X, obs={"age": age})


def test_variable_sites_finds_the_widening_half():
    table = fa.variable_sites(_widening(), bins=3)
    rising = table["rising"].to_numpy()
    assert rising[:20].sum() >= 15, "most widening sites should be detected"
    assert rising[20:].sum() <= 2, "flat sites should mostly be rejected"


def test_variable_sites_direction_is_separate_from_significance():
    """A site whose variance falls is just as significant and must not count."""
    rng = np.random.default_rng(3)
    age = np.linspace(20, 80, 60)
    narrowing = np.clip(0.5 + rng.normal(0, 0.06 - 0.055 * (age - 20) / 60), 0, 1)
    table = fa.variable_sites(_betas(narrowing.reshape(-1, 1), obs={"age": age}))
    assert table["direction"].iloc[0] == -1
    assert not table["rising"].iloc[0]


def test_barometer_is_a_group_statistic_with_its_counts():
    data = _widening()
    data.obs["arm"] = ["a"] * 30 + ["b"] * 30
    out = fa.noise_barometer(data, group="arm")
    assert list(out.index) == ["a", "b"]
    assert (out["n_samples"] == 30).all()
    # Group b is the older half, where the widening sites are wider.
    assert out.loc["b", "barometer"] > out.loc["a", "barometer"]


def test_barometer_says_so_when_nothing_rises():
    rng = np.random.default_rng(5)
    age = np.linspace(20, 80, 40)
    X = np.clip(0.5 + rng.normal(0, 0.03, size=(40, 30)), 0, 1)
    with pytest.raises(AnalysisError, match="variance rose significantly"):
        fa.noise_barometer(_betas(X, obs={"age": age}))


# ---------------------------------------------------------------------------
# repertoire structure
# ---------------------------------------------------------------------------

def _clones(spec):
    rows = [{"sample_id": s, "count": c} for s, counts in spec.items() for c in counts]
    return pd.DataFrame(rows)


def test_effective_clones_matches_inverse_simpson():
    assert effective_clones([1, 1, 1, 1]) == pytest.approx(4.0)
    assert effective_clones([97, 1, 1, 1]) == pytest.approx(
        1.0 / ((0.97 ** 2) + 3 * (0.01 ** 2)))


def test_zipf_sizes_normalise_and_concentrate():
    flat = zipf_clone_sizes(100, alpha=0.0)
    steep = zipf_clone_sizes(100, alpha=1.5)
    assert flat.sum() == pytest.approx(1.0) and steep.sum() == pytest.approx(1.0)
    assert effective_clones(flat) == pytest.approx(100.0)
    assert effective_clones(steep) < 20


def test_diversity_of_an_even_repertoire():
    """Four equal clones: Shannon = log 4, evenness 1, clonality 0."""
    out = fa.repertoire_diversity(_clones({"A": [10, 10, 10, 10]}))
    assert out.loc["A", "richness"] == 4
    assert out.loc["A", "shannon"] == pytest.approx(np.log(4))
    assert out.loc["A", "evenness"] == pytest.approx(1.0)
    assert out.loc["A", "clonality"] == pytest.approx(0.0)
    assert out.loc["A", "effective_clones"] == pytest.approx(4.0)


def test_clonality_rises_as_the_repertoire_concentrates():
    out = fa.repertoire_diversity(_clones({
        "even": [10] * 10,
        "skewed": [901] + [11] * 9,
    }))
    assert out.loc["skewed", "clonality"] > out.loc["even", "clonality"]
    assert out.loc["skewed", "effective_clones"] < out.loc["even", "effective_clones"]
    # Richness is identical, which is the whole point: it misses the structure.
    assert out.loc["skewed", "richness"] == out.loc["even", "richness"]


def test_chao1_adds_a_term_for_unseen_clones():
    """Bias-corrected Chao1: S_obs + f1(f1-1) / (2(f2+1)).

    Two singletons, no doubletons: 3 + 2*1/(2*1) = 4.
    Three singletons, no doubletons: 4 + 3*2/(2*1) = 7.
    The f2+1 denominator is why the f2 = 0 case is defined at all, which the
    classic f1^2 / 2*f2 form is not.
    """
    assert fa.repertoire_diversity(_clones({"A": [1, 1, 5]})).loc["A", "chao1"] == 4.0
    assert fa.repertoire_diversity(_clones({"A": [1, 1, 1, 5]})).loc["A", "chao1"] == 7.0
    # One doubleton moves the denominator: 4 + 2*1/(2*2) = 4.5.
    assert fa.repertoire_diversity(_clones({"A": [1, 1, 2, 5]})).loc["A", "chao1"] == 4.5


def test_chao1_refuses_non_integer_counts():
    out = fa.repertoire_diversity(_clones({"A": [1.5, 2.5, 3.5]}))
    assert np.isnan(out.loc["A", "chao1"])


def test_rarefaction_equalises_depth():
    out = fa.repertoire_diversity(
        _clones({"deep": [50] * 40, "shallow": [5] * 40}), rarefy="min")
    assert out["n_reads"].nunique() == 1
    assert out.attrs["rarefied_to"] == 200


def test_rarefying_up_is_refused():
    with pytest.raises(AnalysisError, match="shallower than the rarefaction depth"):
        fa.repertoire_diversity(_clones({"A": [1, 1]}), rarefy=1000)


# ---------------------------------------------------------------------------
# the clonality simulation
# ---------------------------------------------------------------------------

def _reference(n_sites=300, seed=7):
    rng = np.random.default_rng(seed)
    feats = [f"cg{i:08d}" for i in range(n_sites)]
    return pd.DataFrame(
        {"CD8mem": rng.uniform(0.2, 0.8, n_sites),
         "Neu": rng.uniform(0.2, 0.8, n_sites)}, index=feats)


def test_simulation_holds_cell_fractions_fixed():
    sim = fa.simulate_clonality(
        _reference(), {"CD8mem": 0.3, "Neu": 0.7}, clonal_types=["CD8mem"],
        clone_sizes=[zipf_clone_sizes(n) for n in (2, 200)], n_replicates=3)
    assert sim.n_samples == 6
    assert sim.obs["frac_CD8mem"].nunique() == 1
    assert sim.obs["frac_Neu"].nunique() == 1
    assert sorted(sim.obs["n_clones"].unique()) == [2, 200]


def test_simulated_spread_follows_one_over_root_neff():
    """The derivation says Var = sigma^2 * f^2 / N_eff. Check the ratio.

    Two clone counts a hundredfold apart should give per-site standard
    deviations ten-fold apart. Tolerance is loose because this is a finite
    sample of a random draw, but the exponent is what is being tested and a
    wrong model would miss by far more than the tolerance.
    """
    ref = _reference(n_sites=4000)
    fracs = {"CD8mem": 1.0, "Neu": 0.0}
    spreads = {}
    for n in (10, 1000):
        sim = fa.simulate_clonality(
            ref, fracs, clonal_types=["CD8mem"],
            clone_sizes=[zipf_clone_sizes(n)], n_replicates=1, sigma=0.02, seed=11)
        # Departure of the single simulated sample from the noiseless mixture.
        spreads[n] = float(np.std(sim.X.to_numpy()[0] - ref["CD8mem"].to_numpy()))
    assert spreads[10] / spreads[1000] == pytest.approx(10.0, rel=0.25)


def test_simulation_refuses_fractions_that_do_not_sum_to_one():
    with pytest.raises(DataError, match="sum to 0.9"):
        fa.simulate_clonality(_reference(), {"CD8mem": 0.3, "Neu": 0.6},
                              clonal_types=["CD8mem"],
                              clone_sizes=[zipf_clone_sizes(5)])


def test_simulation_output_is_scoreable_shape():
    sim = fa.simulate_clonality(
        _reference(), {"CD8mem": 0.5, "Neu": 0.5}, clonal_types=["CD8mem"],
        clone_sizes=[zipf_clone_sizes(50)], n_replicates=2, age=60.0)
    assert sim.modality == "dna_methylation"
    assert sim.obs["age"].eq(60.0).all()
    assert sim.X.to_numpy().min() >= 0.0 and sim.X.to_numpy().max() <= 1.0
    assert sim.uns["falconage_simulation"]["kind"] == "clonality"


# ---------------------------------------------------------------------------
# mosaic
# ---------------------------------------------------------------------------

def _cell_ages(ages, widths, grid=(20.0, 80.0)):
    frame = pd.DataFrame({"age": ages, "interval_width": widths},
                         index=[f"c{i}" for i in range(len(ages))])
    frame.attrs["grid"] = grid
    return frame


def test_mosaic_calls_pure_noise_unremarkable():
    """Cells drawn around one true age, with honest widths, must not look mosaic."""
    rng = np.random.default_rng(0)
    se = 4.0
    ages = 50 + rng.normal(0, se, 200)
    widths = np.full(200, se * 2 * 1.96)
    out = fa.mosaic(_cell_ages(ages, widths))
    assert out.loc["all", "p_excess"] > 0.05
    assert out.loc["all", "sd_biological"] < 2.0


def test_mosaic_detects_spread_beyond_the_noise():
    """A genuine bimodal mixture, measured precisely, is not explained by noise."""
    rng = np.random.default_rng(1)
    ages = np.concatenate([rng.normal(30, 1, 100), rng.normal(70, 1, 100)])
    widths = np.full(200, 1.0 * 2 * 1.96)
    out = fa.mosaic(_cell_ages(ages, widths))
    assert out.loc["all", "p_excess"] < 0.01
    assert out.loc["all", "sd_biological"] > 15.0


def test_mosaic_counts_cells_pinned_to_the_grid_edge():
    ages = np.concatenate([np.full(30, 20.0), np.full(30, 50.0)])
    out = fa.mosaic(_cell_ages(ages, np.full(60, 4.0)), min_cells=10)
    assert out.loc["all", "n_at_grid_edge"] == 30


def test_mosaic_groups_and_respects_min_cells():
    ages = np.concatenate([np.full(30, 40.0), np.full(5, 60.0)])
    frame = _cell_ages(ages, np.full(35, 4.0))
    frame["tissue"] = ["a"] * 30 + ["b"] * 5
    out = fa.mosaic(frame, group="tissue", min_cells=20)
    assert list(out.index) == ["a"]


# ---------------------------------------------------------------------------
# variance components
# ---------------------------------------------------------------------------

class _FakeResult:
    def __init__(self, scores, obs):
        self.scores, self.obs = scores, obs


def _balanced(n_subjects=12, n_occasions=2, n_reps=2, *,
              trait=9.0, state=4.0, tech=1.0, seed=0, age=None):
    """A balanced nested design with known variance components."""
    rng = np.random.default_rng(seed)
    rows, index = [], []
    for i in range(n_subjects):
        a = rng.normal(0, np.sqrt(trait))
        for j in range(n_occasions):
            s = rng.normal(0, np.sqrt(state))
            for k in range(n_reps):
                rows.append({"subject": f"p{i}", "occasion": f"o{j}",
                             "value": 50 + a + s + rng.normal(0, np.sqrt(tech)),
                             "age": (age[i] if age is not None else 50.0)})
                index.append(f"p{i}_o{j}_r{k}")
    frame = pd.DataFrame(rows, index=index)
    return _FakeResult(scores=frame[["value"]].rename(columns={"value": "clockA"}),
                       obs=frame[["subject", "occasion", "age"]])


def test_variance_components_recovers_known_components():
    """Large balanced design, generous tolerance: the estimator must be unbiased.

    Moment estimators are noisy at this size, so the assertion is on the
    ordering and the rough magnitude rather than on three decimals. A swapped
    stratum -- the classic bug here -- would fail this by a wide margin.
    """
    res = _balanced(n_subjects=200, trait=9.0, state=4.0, tech=1.0, seed=3)
    vc = fa.variance_components(res, subject_col="subject", occasion_col="occasion")
    row = vc.table.loc["clockA"]
    assert row["var_trait"] == pytest.approx(9.0, rel=0.35)
    assert row["var_state"] == pytest.approx(4.0, rel=0.35)
    assert row["var_tech"] == pytest.approx(1.0, rel=0.25)
    assert row["icc"] == pytest.approx(9 / 14, rel=0.2)


def test_variance_components_without_occasions_is_icc_1_1():
    res = _balanced(n_subjects=60, n_occasions=1, n_reps=3, seed=4)
    vc = fa.variance_components(res, subject_col="subject")
    assert np.isnan(vc.table.loc["clockA", "var_state"])
    assert vc.design["nested"] is False


def test_age_adjustment_exposes_cohort_range_inflation():
    """A clock that only tracks age looks perfectly reliable until age is removed."""
    rng = np.random.default_rng(6)
    ages = np.linspace(20, 80, 60)
    rows, index = [], []
    for i, a in enumerate(ages):
        for k in range(2):
            rows.append({"subject": f"p{i}", "occasion": "o0", "age": a,
                         "clockA": a + rng.normal(0, 1.0)})
            index.append(f"p{i}_r{k}")
    frame = pd.DataFrame(rows, index=index)
    res = _FakeResult(frame[["clockA"]], frame[["subject", "occasion", "age"]])
    vc = fa.variance_components(res, subject_col="subject")
    row = vc.table.loc["clockA"]
    assert row["icc"] > 0.95, "raw ICC is inflated by the 60-year age span"
    assert row["icc_age_adjusted"] < row["icc"] - 0.5


def test_replicates_needed_is_infinite_when_state_dominates():
    """No number of re-scans fixes day-to-day biological variation."""
    res = _balanced(n_subjects=80, trait=1.0, state=9.0, tech=0.5, seed=8)
    vc = fa.variance_components(res, subject_col="subject", occasion_col="occasion")
    assert np.isinf(vc.replicates_needed(0.9).loc["clockA"])


def test_variance_components_refuses_two_subjects():
    res = _balanced(n_subjects=2)
    with pytest.raises(VarianceError, match="fewer than three people"):
        fa.variance_components(res, subject_col="subject", occasion_col="occasion")


# ---------------------------------------------------------------------------
# coefficient mass
# ---------------------------------------------------------------------------

def test_coefficient_mass_weights_by_magnitude_not_count():
    """The whole reason for the statistic: heavy features decide behaviour."""
    reg = fa.registry.load()
    cid = next(c for c in reg.list() if reg.has_coefficient_vector(c))
    feats, weights = reg.coefficients(cid)
    order = np.argsort(-np.abs(np.asarray(weights, dtype=float).ravel()))
    heaviest = [str(feats[i]) for i in order[:5]]

    out = fa.coefficient_mass({"heavy": heaviest}, registry=reg, clocks=[cid])
    row = out.loc[cid]
    assert row["heavy_n"] == 5
    assert row["heavy_frac_mass"] > row["heavy_frac_sites"], (
        "five of the largest weights must hold more mass than five of any five")


def test_coefficient_mass_reports_what_it_skipped():
    reg = fa.registry.load()
    untraced = [c.clock_id if hasattr(c, "clock_id") else c.name
                for c in reg.filter(availability="untraced")][:3]
    bundled = next(c for c in reg.list() if reg.has_coefficient_vector(c))
    out = fa.coefficient_mass(["cg00000029"], registry=reg,
                              clocks=[bundled, *untraced])
    assert set(out.attrs["skipped"]) >= set(untraced)
    assert bundled in out.index


def test_coefficient_mass_counts_unmatched_annotation_ids():
    """The check that catches an EPIC v2 suffix list meeting bare probe ids."""
    reg = fa.registry.load()
    cid = next(c for c in reg.list() if reg.has_coefficient_vector(c))
    out = fa.coefficient_mass({"nonsense": ["not_a_probe_1", "not_a_probe_2"]},
                              registry=reg, clocks=[cid])
    assert out.attrs["unmatched"]["nonsense"] == 2
    assert out.loc[cid, "nonsense_frac_mass"] == 0.0
