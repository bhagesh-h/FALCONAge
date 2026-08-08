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


# ---------------------------------------------------------------------------
# the published output transforms
# ---------------------------------------------------------------------------

def test_every_documented_postprocess_op_is_dispatchable():
    """The op table in the science page is the spec; this is the inventory."""
    from falconage.models import ops

    documented = {
        "add", "multiply", "divide_by", "anti_log_linear", "log_linear",
        "expit", "exp", "clip", "cox_to_years", "anti_logp2", "anti_log_log",
        "one_minus", "days_to_weeks", "days_to_months", "scale_and_shift",
        "petkovich_blood", "stubbs_multitissue", "mortality_to_phenoage",
        "anti_log", "sigmoid", "add_constant",
    }
    assert documented <= set(ops.POSTPROCESS)


def test_the_aliases_are_the_same_function_not_a_copy():
    """A copied implementation is one that can drift; an alias cannot."""
    from falconage.models import ops

    assert ops.POSTPROCESS["anti_log"] is ops.POSTPROCESS["exp"]
    assert ops.POSTPROCESS["sigmoid"] is ops.POSTPROCESS["expit"]
    assert ops.POSTPROCESS["add_constant"] is ops.POSTPROCESS["add"]


@pytest.mark.parametrize("op,kwargs,x,want", [
    ("anti_logp2", {}, 0.0, -1.0),                     # e^0 - 2
    ("anti_log_log", {}, 0.0, np.exp(-1.0)),           # e^(-e^0)
    ("one_minus", {}, 0.25, 0.75),
    ("days_to_weeks", {}, 280.0, 40.0),                # a term pregnancy
    ("days_to_months", {}, 61.0, 2.0),
    ("scale_and_shift", {"scale": 2.0, "offset": 3.0}, 1.0, 8.0),   # (1+3)*2
])
def test_transform_values(op, kwargs, x, want):
    from falconage.models import ops

    got = ops.POSTPROCESS[op](np.array([x]), **kwargs)
    assert float(got[0]) == pytest.approx(want)


def test_petkovich_is_pinned_at_zero_rather_than_nan():
    """A fractional power of a negative base is a NaN that spreads."""
    from falconage.models import ops

    out = ops.POSTPROCESS["petkovich_blood"](np.array([-99.0, 0.0]))
    assert np.isfinite(out).all()
    assert out[0] == 0.0


def test_stubbs_is_not_monotone_and_that_is_the_published_model():
    from falconage.models import ops

    f = ops.POSTPROCESS["stubbs_multitissue"]
    vertex = -1.2424 / (2 * 0.1207)
    lo, hi = f(np.array([vertex - 2.0])), f(np.array([vertex + 2.0]))
    assert float(lo[0]) == pytest.approx(float(hi[0]), rel=1e-9)


def test_mortality_to_phenoage_uses_the_corrected_constant():
    """0.090165, not 0.09165. The two differ by years on the same input."""
    from falconage.models import ops

    f = ops.POSTPROCESS["mortality_to_phenoage"]
    x = np.array([-7.0, -5.0, -3.0])
    got = f(x)

    gamma = 0.0076927
    m = 1.0 - np.exp(-np.exp(x) * (np.exp(120.0 * gamma) - 1.0) / gamma)
    want = 141.50225 + np.log(-0.00553 * np.log(1.0 - m)) / 0.090165
    np.testing.assert_allclose(got, want, rtol=0, atol=0)

    # Monotone: more mortality risk is more phenotypic age, always.
    assert np.all(np.diff(got) > 0)

    # And the wrong constant is not quietly equivalent.
    wrong = 141.50225 + np.log(-0.00553 * np.log(1.0 - m)) / 0.09165
    assert abs(float((got - wrong)[0])) > 1.0


def test_mortality_to_phenoage_survives_the_extremes():
    """ln(1 - m) diverges at m = 1; a very sick sample is not an infinity."""
    from falconage.models import ops

    out = ops.POSTPROCESS["mortality_to_phenoage"](np.array([-40.0, 0.0, 40.0]))
    assert np.isfinite(out).all()


# ---------------------------------------------------------------------------
# PC clocks
# ---------------------------------------------------------------------------

def _synthetic_rotation(features, n_components=5, seed=20260808):
    """A rotation with the shape of a real one and none of the coefficients."""
    from falconage.models.pc import PCRotation

    rng = np.random.default_rng(seed)
    n = len(features)
    return PCRotation(
        features=list(features),
        centre=rng.uniform(0.2, 0.8, n),
        rotation=rng.normal(0, 1 / np.sqrt(n), (n, n_components)),
        coefficients=rng.normal(0, 2.0, n_components),
    )


def test_pc_rotation_rejects_mismatched_shapes():
    from falconage.core.errors import RegistryError
    from falconage.models.pc import PCRotation

    with pytest.raises(RegistryError, match="coefficients"):
        PCRotation(features=["a", "b"], centre=np.zeros(2),
                   rotation=np.zeros((2, 3)), coefficients=np.zeros(2))
    with pytest.raises(RegistryError, match="rotation has"):
        PCRotation(features=["a", "b"], centre=np.zeros(2),
                   rotation=np.zeros((3, 2)), coefficients=np.zeros(2))


def test_pc_clock_computes_the_published_form(synthetic_betas):
    """((x - centre) @ rotation) @ coefficients, asserted against numpy."""
    from falconage.models.pc import PCLinearClock

    reg = fa.registry.load()
    feats = list(synthetic_betas.features[:200])
    rot = _synthetic_rotation(feats)
    m = PCLinearClock(clock=reg.get("pchorvath2013"), rotation=rot)

    got, al = m.predict(synthetic_betas, resolve("cpu"))

    x = synthetic_betas.X[feats].to_numpy(dtype=np.float64)
    want = ((x - rot.centre) @ rot.rotation) @ rot.coefficients
    # pchorvath2013 carries Horvath's output transform, so apply it too.
    want = ops.apply_chain(want, reg.get("pchorvath2013").postprocess,
                           ops.POSTPROCESS)
    np.testing.assert_allclose(got.to_numpy(), want, rtol=1e-12)
    assert al.coverage == pytest.approx(1.0)


def test_pc_clock_reports_no_coefficient_mass(synthetic_betas):
    """A PC clock has no per-probe weight, so claiming a mass share would be
    attributing a component back to probes -- the very thing PCA removes."""
    from falconage.models.pc import PCLinearClock

    reg = fa.registry.load()
    feats = list(synthetic_betas.features[:200])
    m = PCLinearClock(clock=reg.get("pchorvath2013"),
                      rotation=_synthetic_rotation(feats))
    _, al = m.predict(synthetic_betas, resolve("cpu"))
    assert al.mass_coverage is None


def test_pc_rotation_roundtrips_through_npz(tmp_path, synthetic_betas):
    from falconage.models.pc import read_rotation

    feats = list(synthetic_betas.features[:50])
    rot = _synthetic_rotation(feats, n_components=4)
    p = tmp_path / "r.npz"
    np.savez(p, features=np.array(rot.features), centre=rot.centre,
             rotation=rot.rotation, coefficients=rot.coefficients)

    back = read_rotation(p)
    assert back.features == rot.features
    assert back.n_components == 4
    np.testing.assert_allclose(back.rotation, rot.rotation)


def test_npz_missing_an_array_says_which(tmp_path):
    from falconage.core.errors import RegistryError
    from falconage.models.pc import read_rotation

    p = tmp_path / "bad.npz"
    np.savez(p, features=np.array(["a"]), centre=np.zeros(1))
    with pytest.raises(RegistryError, match="rotation"):
        read_rotation(p)


def test_pc_clock_enforces_the_feature_floor(synthetic_betas):
    from falconage.core.errors import FeatureCoverageError
    from falconage.models.pc import PCLinearClock

    reg = fa.registry.load()
    feats = list(synthetic_betas.features[:100]) + [f"cg9999{i:04d}" for i in range(100)]
    m = PCLinearClock(clock=reg.get("pchorvath2013"),
                      rotation=_synthetic_rotation(feats))
    with pytest.raises(FeatureCoverageError, match="below the"):
        m.predict(synthetic_betas, resolve("cpu"), min_coverage=0.8)
