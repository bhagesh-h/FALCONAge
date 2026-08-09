"""BMIQ, and the beta-mixture fit under it.

The fit is checked against mixtures drawn with known parameters, which is the
only way to know an EM is right rather than merely converged. The normalisation
is checked on the property it exists to produce: type II probes ending up on the
type I scale, and type I probes untouched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import falconage as fa
from falconage.core.errors import DataError
from falconage.preprocess.bmiq import bmiq, fit_beta_mixture


def _draw(rng, n, weights, params):
    """Sample from a three-state beta mixture with known parameters."""
    which = rng.choice(3, size=n, p=weights)
    out = np.empty(n)
    for k, (a, b) in enumerate(params):
        m = which == k
        out[m] = rng.beta(a, b, size=int(m.sum()))
    return np.clip(out, 1e-4, 1 - 1e-4)


# ---------------------------------------------------------------------------
# the mixture fit
# ---------------------------------------------------------------------------

def test_the_em_recovers_parameters_it_was_given(rng):
    """The check that separates a correct EM from a converged one."""
    weights = np.array([0.45, 0.10, 0.45])
    params = [(2.0, 20.0), (8.0, 8.0), (20.0, 2.0)]
    x = _draw(rng, 40_000, weights, params)

    fit = fit_beta_mixture(x)
    assert fit.converged
    order = np.argsort(fit.means)
    got = fit.means[order]
    want = [a / (a + b) for a, b in params]
    assert np.allclose(got, want, atol=0.03), (got, want)
    assert np.allclose(fit.weights[order], weights, atol=0.05)


def test_the_states_come_out_in_the_order_their_names_imply(rng):
    """Components are not exchangeable here: state 0 must mean unmethylated.
    A random start can converge to a relabelled solution that then maps the
    wrong state onto the wrong one, and the output looks plausible."""
    x = _draw(rng, 20_000, [0.4, 0.2, 0.4], [(2.0, 20.0), (8.0, 8.0), (20.0, 2.0)])
    fit = fit_beta_mixture(x)
    assert list(fit.means) == sorted(fit.means), fit.means


def test_the_fit_is_reproducible(rng):
    x = _draw(rng, 10_000, [0.4, 0.2, 0.4], [(2.0, 20.0), (8.0, 8.0), (20.0, 2.0)])
    a = fit_beta_mixture(x)
    b = fit_beta_mixture(x)
    assert np.allclose(a.means, b.means) and a.iterations == b.iterations


def test_hard_assignment_separates_the_modes(rng):
    x = _draw(rng, 20_000, [0.45, 0.10, 0.45], [(2.0, 30.0), (8.0, 8.0), (30.0, 2.0)])
    fit = fit_beta_mixture(x)
    st = fit.state(x)
    order = np.argsort(fit.means)
    assert x[st == order[0]].mean() < 0.2
    assert x[st == order[2]].mean() > 0.8


def test_a_handful_of_probes_is_not_a_mixture(rng):
    with pytest.raises(DataError, match="not a fit"):
        fit_beta_mixture(rng.uniform(0, 1, size=40))


# ---------------------------------------------------------------------------
# the normalisation
# ---------------------------------------------------------------------------

def _two_type_data(rng, n_i=4_000, n_ii=8_000, n_samples=3):
    """Type I and type II probes drawn from deliberately offset mixtures.

    Type II is compressed toward the middle, which is the real artefact BMIQ
    exists to remove.
    """
    feats = [f"cg{i:08d}" for i in range(n_i + n_ii)]
    rows = []
    for _ in range(n_samples):
        xi = _draw(rng, n_i, [0.45, 0.10, 0.45],
                   [(2.0, 30.0), (8.0, 8.0), (30.0, 2.0)])
        xii = _draw(rng, n_ii, [0.45, 0.10, 0.45],
                    [(4.0, 12.0), (8.0, 8.0), (12.0, 4.0)])   # compressed
        rows.append(np.concatenate([xi, xii]))
    X = pd.DataFrame(rows, index=[f"S{i}" for i in range(n_samples)], columns=feats)
    types = pd.Series(["I"] * n_i + ["II"] * n_ii, index=feats)
    d = fa.FalconData(X=X, obs=pd.DataFrame(index=X.index),
                      modality="dna_methylation", platform="450K")
    return d, types


def test_type_i_probes_are_left_exactly_alone(rng):
    """They are the reference. It also means BMIQ never moves a clock built
    purely on type I probes, which is worth being able to state."""
    d, types = _two_type_data(rng)
    out = bmiq(d, probe_type=types)
    i_cols = types.index[types == "I"]
    assert np.array_equal(d.X[i_cols].to_numpy(), out.X[i_cols].to_numpy())


def test_type_ii_probes_move_toward_the_type_i_scale(rng):
    d, types = _two_type_data(rng)
    out = bmiq(d, probe_type=types)
    i_cols = types.index[types == "I"]
    ii_cols = types.index[types == "II"]

    ref = d.X[i_cols].to_numpy().ravel()
    before = d.X[ii_cols].to_numpy().ravel()
    after = out.X[ii_cols].to_numpy().ravel()

    # The compression is the artefact: type II starts with a narrower spread
    # than type I and should end up closer to it.
    gap_before = abs(before.std() - ref.std())
    gap_after = abs(after.std() - ref.std())
    assert gap_after < gap_before, (gap_before, gap_after)


def test_output_stays_in_the_beta_range(rng):
    d, types = _two_type_data(rng)
    out = bmiq(d, probe_type=types)
    v = out.X.to_numpy()
    assert np.nanmin(v) >= 0.0 and np.nanmax(v) <= 1.0


def test_it_is_fitted_per_sample_not_once_across_the_cohort(rng):
    """The type I/II offset is a property of a chip and a run. Fitting once
    across samples would smear one array's dye batch onto every other."""
    d, types = _two_type_data(rng, n_samples=3)
    out = bmiq(d, probe_type=types)
    per = out.uns["bmiq"]["per_sample"]
    assert len(per) == 3
    assert all(r["status"] == "ok" for r in per)
    # Three independent fits give three different sets of component means.
    means = [tuple(r["type_ii_means"]) for r in per]
    assert len(set(means)) == 3


def test_the_run_records_what_was_fitted(rng):
    d, types = _two_type_data(rng)
    rec = bmiq(d, probe_type=types).uns["bmiq"]
    assert rec["n_type_i"] == 4_000 and rec["n_type_ii"] == 8_000
    assert "Teschendorff" in rec["reference"]


def test_a_clock_sized_subset_is_refused(rng):
    """BMIQ fits a mixture to each probe type and needs a full array. On 353
    probes there is nothing to fit, and a fit that succeeds on that many is
    describing noise."""
    d, types = _two_type_data(rng, n_i=200, n_ii=200)
    with pytest.raises(DataError, match="nothing to fit"):
        bmiq(d, probe_type=types)


def test_it_says_what_it_needs_when_the_platform_is_unknown(rng):
    d, _ = _two_type_data(rng)
    bare = fa.FalconData(X=d.X, obs=d.obs, modality="dna_methylation")
    with pytest.raises(DataError, match="type I and which are type II"):
        bmiq(bare)
