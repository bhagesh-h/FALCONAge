"""Clocks that summarise a probe set rather than weighting it."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import falconage as fa
from falconage.core.errors import FeatureCoverageError, ScoringError
from falconage.models.aggregation import AggregationClock, is_aggregation, parse_statistic

#: Six, not the five the model_type tally suggests at a glance: `hypoclock` is
#: declared "mean aggregation" without the word methylation, and it is one --
#: the mean beta over 678 solo-WCGW CpGs, inverted by its `one_minus`
#: postprocess. Detection reads the word "aggregation", which caught it; the
#: first version of this list did not.
AGGREGATORS = {"epitoc1", "epicmithyper", "stemtoc", "stemtocvitro", "reedbmi",
               "hypoclock"}


# ---------------------------------------------------------------------------
# reading the statistic out of what the registry already declares
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_type,expect", [
    ("mean methylation aggregation", ("mean", None)),
    ("95th-percentile methylation aggregation", ("quantile", 0.95)),
    ("90th percentile methylation aggregation", ("quantile", 0.90)),
    ("weighted methylation aggregation", ("weighted_mean", None)),
])
def test_the_statistic_is_read_from_the_declared_model_type(model_type, expect):
    assert parse_statistic(model_type) == expect


def test_an_unreadable_model_type_refuses_rather_than_defaulting_to_the_mean():
    """A percentile clock scored as a mean returns a plausible number on the
    right scale that measures a different thing."""
    with pytest.raises(ScoringError, match="Recognised:"):
        parse_statistic("elastic net regression")


def test_the_registry_agrees_on_which_clocks_these_are(registry):
    got = {c.id for c in registry if is_aggregation(c)}
    assert got == AGGREGATORS


def test_every_one_of_them_parses(registry):
    """A statistic that cannot be read is a clock that cannot be scored the day
    somebody registers its probe list."""
    for cid in AGGREGATORS:
        stat, q = parse_statistic(registry.get(cid).model_type)
        assert stat in {"mean", "quantile", "weighted_mean"}
        assert (q is None) or (0 < q < 1)


def test_they_are_all_still_scaffolds(registry):
    """No probe list ships for any of them, so this is architecture, exactly
    like PCLinearClock. The point is that registering one now works."""
    for cid in AGGREGATORS:
        assert not registry.has_coefficients(cid)


# ---------------------------------------------------------------------------
# the arithmetic
# ---------------------------------------------------------------------------

def _clock(registry, cid):
    return registry.get(cid)


def _data(rng, feats, n=12):
    ids = [f"S{i:03d}" for i in range(n)]
    X = pd.DataFrame(rng.uniform(0.05, 0.95, size=(n, len(feats))),
                     index=ids, columns=feats)
    return fa.FalconData(X=X, obs=pd.DataFrame(index=ids),
                         modality="dna_methylation", platform="450K")


def test_the_mean_form_is_the_mean(registry, rng):
    feats = [f"cg{i:08d}" for i in range(40)]
    d = _data(rng, feats)
    m = AggregationClock(clock=_clock(registry, "epitoc1"), features=feats,
                         coefficients=np.ones(len(feats)), statistic="mean")
    from falconage.core.backend import resolve

    got, _ = m.predict(d, resolve("cpu", None, requires_fp64=False), min_coverage=0.0)
    assert np.allclose(got.to_numpy(), d.X[feats].to_numpy().mean(axis=1))


def test_the_percentile_form_is_taken_across_probes_within_a_sample(registry, rng):
    """Not across samples. The premise of a percentile clock is about the
    spread inside one methylome."""
    feats = [f"cg{i:08d}" for i in range(200)]
    d = _data(rng, feats)
    m = AggregationClock(clock=_clock(registry, "stemtoc"), features=feats,
                         coefficients=np.ones(len(feats)), statistic="quantile", q=0.95)
    from falconage.core.backend import resolve

    got, _ = m.predict(d, resolve("cpu", None, requires_fp64=False), min_coverage=0.0)
    want = np.nanquantile(d.X[feats].to_numpy(), 0.95, axis=1)
    assert np.allclose(got.to_numpy(), want)
    # And it is emphatically not the mean.
    assert not np.allclose(got.to_numpy(), d.X[feats].to_numpy().mean(axis=1))


def test_the_weighted_form_uses_the_weights(registry, rng):
    feats = [f"cg{i:08d}" for i in range(30)]
    d = _data(rng, feats)
    w = rng.uniform(0.1, 3.0, size=len(feats))
    m = AggregationClock(clock=_clock(registry, "reedbmi"), features=feats,
                         coefficients=w, statistic="weighted_mean")
    from falconage.core.backend import resolve

    got, _ = m.predict(d, resolve("cpu", None, requires_fp64=False), min_coverage=0.0)
    want = (d.X[feats].to_numpy() * w).sum(axis=1) / w.sum()
    assert np.allclose(got.to_numpy(), want)


def test_the_mean_form_ignores_the_weight_column(registry, rng):
    """For a mean or a percentile clock the published definition is over the
    probe set. A weight column in the registered file is a fact nobody put
    there, and it must not change the answer."""
    feats = [f"cg{i:08d}" for i in range(40)]
    d = _data(rng, feats)
    from falconage.core.backend import resolve

    spec = resolve("cpu", None, requires_fp64=False)
    ones = AggregationClock(clock=_clock(registry, "epitoc1"), features=feats,
                            coefficients=np.ones(len(feats)), statistic="mean")
    junk = AggregationClock(clock=_clock(registry, "epitoc1"), features=feats,
                            coefficients=rng.uniform(0.1, 5, len(feats)),
                            statistic="mean")
    a, _ = ones.predict(d, spec, min_coverage=0.0)
    b, _ = junk.predict(d, spec, min_coverage=0.0)
    assert np.array_equal(a.to_numpy(), b.to_numpy())


def test_a_partial_probe_set_is_refused(registry, rng):
    """An aggregate over a fraction of the probe set is an aggregate over a
    different probe set -- there is no intercept to absorb the difference."""
    feats = [f"cg{i:08d}" for i in range(100)]
    d = _data(rng, feats[:30])
    m = AggregationClock(clock=_clock(registry, "epitoc1"), features=feats,
                         coefficients=np.ones(len(feats)), statistic="mean")
    from falconage.core.backend import resolve

    with pytest.raises(FeatureCoverageError, match="different probe set"):
        m.predict(d, resolve("cpu", None, requires_fp64=False), min_coverage=0.8)


def test_a_zero_weight_file_is_refused(registry, rng):
    feats = [f"cg{i:08d}" for i in range(20)]
    d = _data(rng, feats)
    m = AggregationClock(clock=_clock(registry, "reedbmi"), features=feats,
                         coefficients=np.zeros(len(feats)), statistic="weighted_mean")
    from falconage.core.backend import resolve

    with pytest.raises(ScoringError, match="non-zero weights"):
        m.predict(d, resolve("cpu", None, requires_fp64=False), min_coverage=0.0)


# ---------------------------------------------------------------------------
# and end to end, through the registry, with a registered probe list
# ---------------------------------------------------------------------------

def test_registering_a_probe_list_makes_one_scoreable(fresh_registry, rng, tmp_path):
    """The whole point. A published probe set is a two-column CSV with weight
    1.0, which is the file register_local_weights already validates."""
    import dataclasses

    reg = fresh_registry
    feats = [f"cg{i:08d}" for i in range(60)]
    p = tmp_path / "epitoc1.csv"
    p.write_text("feature_id,coefficient\n"
                 + "\n".join(f"{f},1.0" for f in feats) + "\n", encoding="utf-8")

    # n_features is None in the registry for these; declare it so the file
    # validator has something to check against.
    reg._clocks["epitoc1"] = dataclasses.replace(reg.get("epitoc1"),
                                                 n_features=len(feats))
    reg.register_local_weights("epitoc1", p)
    assert reg.has_coefficients("epitoc1")

    d = _data(rng, feats)
    obs = d.obs.copy()
    obs["tissue"] = "whole blood"
    d = fa.FalconData(X=d.X, obs=obs, modality=d.modality, platform=d.platform)

    res = fa.score(d, clocks=["epitoc1"], min_coverage=0.0, registry=reg)
    assert res.scores.shape == (d.n_samples, 1)
    assert np.allclose(res.scores["epitoc1"].to_numpy(),
                       d.X[feats].to_numpy().mean(axis=1))
    # And the scale it was declared on still governs what may be done with it.
    assert reg.get("epitoc1").scale_type == "divisions"
    assert "acceleration" not in reg.get("epitoc1").legal_operations
