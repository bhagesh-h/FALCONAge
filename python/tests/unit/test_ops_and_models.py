"""Arithmetic, asserted against hand computation rather than another package.

Every constant here is traceable to a published paper. Where a test says a
number, that number was worked out from the formula in the paper, not read off
another implementation's output -- which is the whole point of section 10.1's
oracle ranking.
"""

from __future__ import annotations

import numpy as np
import pytest

import falconage as fa
from falconage.core.backend import resolve
from falconage.core.errors import ScoringError
from falconage.models import ops


# ---------------------------------------------------------------------------
# Horvath's transform
# ---------------------------------------------------------------------------
def test_anti_log_linear_matches_the_published_definition():
    """F(y) = 21·e^y − 1 for y<0, 21·y + 20 for y>=0, with adult_age=20."""
    y = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    got = ops.anti_log_linear(y, adult_age=20.0)
    want = np.where(y < 0, 21.0 * np.exp(y) - 1.0, 21.0 * y + 20.0)
    np.testing.assert_allclose(got, want, rtol=0, atol=0)
    # At y=0 both branches give 20 -- the join is continuous, which is why a
    # Horvath prediction of exactly 0 is twenty years and not one.
    assert ops.anti_log_linear(np.array([0.0]))[0] == pytest.approx(20.0)


def test_anti_log_linear_inverts_log_linear():
    x = np.array([0.5, 5.0, 20.0, 45.0, 90.0])
    np.testing.assert_allclose(ops.anti_log_linear(ops.log_linear(x)), x, rtol=1e-12)


def test_horvath_worked_example():
    """A linear predictor of 1.0 must give 41 years.

    21 * 1.0 + 20 = 41. Hand arithmetic, not a fixture: if the intercept or the
    adult-age constant ever changes this test says so immediately.
    """
    assert ops.anti_log_linear(np.array([1.0]))[0] == pytest.approx(41.0)


# ---------------------------------------------------------------------------
# the rest of the catalogue
# ---------------------------------------------------------------------------
def test_beta_m_roundtrip():
    b = np.array([0.01, 0.25, 0.5, 0.75, 0.99])
    np.testing.assert_allclose(ops.m_to_beta(ops.beta_to_m(b)), b, rtol=1e-9)


def test_beta_to_m_survives_exact_zero_and_one():
    """Real normalised arrays contain exact 0 and 1; an infinity there
    propagates through the whole dot product and returns NaN for the sample."""
    m = ops.beta_to_m(np.array([0.0, 1.0]))
    assert np.isfinite(m).all()


def test_simplex_projection_returns_a_composition():
    v = np.array([[0.6, 0.5, -0.2], [1.5, -0.4, 0.1]])
    p = ops.simplex_projection(v)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-12)
    assert (p >= 0).all()


def test_simplex_projection_is_not_clip_and_renormalise():
    """The two give different answers, and the projection is the correct one."""
    v = np.array([[0.9, 0.5, -0.4]])
    proj = ops.simplex_projection(v)[0]
    naive = np.clip(v[0], 0, None)
    naive = naive / naive.sum()
    assert not np.allclose(proj, naive)


def test_rank_normalize_is_scale_free():
    x = np.array([[1.0, 5.0, 3.0, 2.0]])
    np.testing.assert_allclose(ops.rank_normalize(x), ops.rank_normalize(x * 1000 + 7))


def test_cox_to_years_grimage_constants():
    """The four GrimAge2 constants, applied to a linear predictor equal to the
    cohort mean, must return the cohort mean age."""
    out = ops.cox_to_years(np.array([15.370829484122]),
                           cox_mean=15.370829484122, cox_std=1.09534876966487,
                           age_mean=66.0943807965085, age_std=9.05974444998421)
    assert out[0] == pytest.approx(66.0943807965085)


def test_apply_chain_runs_in_order():
    x = np.array([1.0])
    chain = ({"op": "add", "value": 1.0}, {"op": "multiply", "value": 10.0})
    assert ops.apply_chain(x, chain, ops.POSTPROCESS)[0] == pytest.approx(20.0)
    # order matters, and the reverse proves the chain is not being sorted
    rev = ({"op": "multiply", "value": 10.0}, {"op": "add", "value": 1.0})
    assert ops.apply_chain(x, rev, ops.POSTPROCESS)[0] == pytest.approx(11.0)


def test_unknown_op_raises_rather_than_skipping():
    with pytest.raises(ScoringError, match="unknown operation"):
        ops.apply_chain(np.array([1.0]), ({"op": "nope"},), ops.POSTPROCESS)


def test_describe_chain_is_readable():
    c = ({"op": "add", "value": 0.696}, {"op": "anti_log_linear", "adult_age": 20})
    assert ops.describe_chain(c) == "add(value=0.696) -> anti_log_linear(adult_age=20)"


# ---------------------------------------------------------------------------
# alignment and imputation
# ---------------------------------------------------------------------------
def test_alignment_never_fills_with_zero(synthetic_betas):
    """Zero is a real, extreme beta. Filling an absent probe with it shifts any
    clock with a large coefficient on that probe by whole years."""
    feats = list(synthetic_betas.features[:50]) + ["cg99999999", "cg99999998"]
    al = fa.models.align(synthetic_betas, feats, imputation="reference")
    assert al.n_imputed == 2 * synthetic_betas.n_samples
    assert not np.any(al.matrix == 0.0)
    assert al.coverage == pytest.approx(50 / 52)


def test_alignment_none_leaves_nan(synthetic_betas):
    al = fa.models.align(synthetic_betas, list(synthetic_betas.features[:10]) + ["cgX"],
                         imputation="none")
    assert np.isnan(al.matrix[:, -1]).all()


def test_all_nan_column_counts_as_absent(synthetic_betas):
    """Present as a column, absent as a measurement -- GEO series matrices carry
    these routinely."""

    X = synthetic_betas.X.copy()
    dead = X.columns[0]
    X[dead] = np.nan
    d = fa.FalconData(X=X, obs=synthetic_betas.obs, modality="dna_methylation")
    al = fa.models.align(d, [dead, X.columns[1]])
    assert not al.present[0]
    assert al.coverage == pytest.approx(0.5)


def test_mass_coverage_needs_coefficients_to_exist(synthetic_betas):
    """Without weights there is no honest answer, so the field stays None
    rather than defaulting to 1.0 and reading as 'all the weight is here'."""
    al = fa.models.align(synthetic_betas, list(synthetic_betas.X.columns[:4]))
    assert al.mass_coverage is None
    assert al.missing_mass == []


def test_mass_coverage_separates_heavy_from_negligible_absences(synthetic_betas):
    """The case feature count cannot see.

    Two datasets, each missing exactly one of four features, so both report
    75% feature coverage. One drops the feature carrying 97% of the weight and
    the other drops a rounding error. A single coverage number calls these
    identical; they are not.
    """
    feats = list(synthetic_betas.X.columns[:4])
    coefs = np.array([10.0, 0.1, 0.1, 0.1])

    def cover(drop):
        X = synthetic_betas.X.drop(columns=[feats[drop]])
        d = fa.FalconData(X=X, obs=synthetic_betas.obs,
                          modality="dna_methylation")
        return fa.models.align(d, feats, coefficients=coefs)

    heavy, light = cover(0), cover(3)

    assert heavy.coverage == pytest.approx(0.75)
    assert light.coverage == pytest.approx(0.75)

    assert heavy.mass_coverage == pytest.approx(0.3 / 10.3)
    assert light.mass_coverage == pytest.approx(10.2 / 10.3)
    assert heavy.missing_mass[0][0] == feats[0]


def test_the_mass_floor_rejects_what_the_feature_floor_would_pass(synthetic_betas):
    """A clock can clear the count and still have lost the probes it leans on."""
    from falconage.core.errors import FeatureCoverageError

    reg = fa.registry.load()
    m = fa.models.LinearClock.from_registry(reg, "horvath2013")

    # Drop the ten heaviest probes: a rounding error by count, a large share of
    # the model by weight.
    heaviest = [f for _, f in sorted(
        zip(np.abs(m.coefficients), m.features), reverse=True)[:10]]
    X = synthetic_betas.X.drop(columns=heaviest, errors="ignore")
    d = fa.FalconData(X=X, obs=synthetic_betas.obs,
                      modality="dna_methylation", platform="450K")

    al = fa.models.align(d, m.features, coefficients=m.coefficients)
    assert al.coverage > al.mass_coverage    # the whole point

    spec = resolve("cpu")
    floor = 0.5 * (al.coverage + al.mass_coverage)   # between the two measures
    with pytest.raises(FeatureCoverageError, match="total .coefficient."):
        m.predict(d, spec, min_coverage=floor)


def test_feature_order_does_not_change_the_score(synthetic_betas):
    """A clock's answer must not depend on the column order of the input."""
    spec = resolve("cpu")
    m = fa.models.LinearClock.from_registry(fa.registry.load(), "horvath2013")
    a, _ = m.predict(synthetic_betas, spec)

    shuffled = synthetic_betas.X.sample(frac=1.0, axis=1, random_state=1)
    d2 = fa.FalconData(X=shuffled, obs=synthetic_betas.obs,
                       modality="dna_methylation", platform="450K")
    b, _ = m.predict(d2, spec)
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), rtol=0, atol=0)


def test_duplicating_a_sample_duplicates_its_score(synthetic_betas):
    import pandas as pd

    spec = resolve("cpu")
    m = fa.models.LinearClock.from_registry(fa.registry.load(), "hannum")
    base, _ = m.predict(synthetic_betas, spec)

    X = pd.concat([synthetic_betas.X, synthetic_betas.X.iloc[[0]].rename(index={
        synthetic_betas.X.index[0]: "DUP"})])
    obs = pd.concat([synthetic_betas.obs, synthetic_betas.obs.iloc[[0]].rename(
        index={synthetic_betas.obs.index[0]: "DUP"})])
    d = fa.FalconData(X=X, obs=obs, modality="dna_methylation", platform="450K")
    dup, _ = m.predict(d, spec)
    assert dup["DUP"] == pytest.approx(dup[synthetic_betas.X.index[0]], abs=1e-12)


# ---------------------------------------------------------------------------
# the synthetic-weights trick for scaffolds
# ---------------------------------------------------------------------------
def test_scaffold_architecture_is_testable_without_real_coefficients(tmp_path):
    """A tier C clock given a one-hot coefficient vector must return exactly the
    matching feature's value put through its postprocess chain.

    This is how a scaffold is verified: the architecture is checked
    algebraically, and only the numbers are missing.
    """
    import pandas as pd

    reg = fa.registry.ClockRegistry.from_yaml()
    n = reg.get("dunedinpace").n_features or 20_000
    feats = [f"cg{i:08d}" for i in range(n)]

    w = tmp_path / "synthetic.csv"
    coefs = np.zeros(n)
    coefs[7] = 2.0
    w.write_text("feature_id,coefficient\n"
                 + "\n".join(f"{f},{c}" for f, c in zip(feats, coefs)) + "\n")
    reg.register_local_weights("dunedinpace", w)

    X = pd.DataFrame(np.full((2, n), 0.25), index=["a", "b"], columns=feats)
    X.iloc[0, 7] = 0.5
    d = fa.FalconData(X=X, obs=pd.DataFrame(index=["a", "b"]),
                      modality="dna_methylation")
    model = fa.models.build(reg, "dunedinpace")
    got, _ = model.predict(d, resolve("cpu"))
    assert got["a"] == pytest.approx(1.0)   # 0.5 * 2.0, identity postprocess
    assert got["b"] == pytest.approx(0.5)   # 0.25 * 2.0
