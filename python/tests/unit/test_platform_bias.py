"""Probe loss, priced in the clock's own unit.

``probe_loss`` already said how much of a model is absent. These cover the
column that says what that costs, and the two rows that make the whole table
believable: a clock that loses nothing shifts by exactly zero.
"""

from __future__ import annotations

import numpy as np
import pytest

import falconage as fa
from falconage.preprocess import BIAS_WARN, _load_platform_bias

TABLE = _load_platform_bias()
needs_table = pytest.mark.skipif(
    not TABLE, reason="platform_bias.csv absent; run python/tools/build_platform_bias.py")


@needs_table
def test_the_table_covers_the_platforms_users_actually_have():
    plats = {p for _, p in TABLE}
    assert {"EPICv1", "EPICv2"} <= plats


@needs_table
def test_a_clock_that_loses_nothing_shifts_by_exactly_zero():
    """The internal check on the whole measurement.

    ``vidalbralo`` has eight probes and keeps all eight on both platforms. A
    non-zero shift there would mean the masking is picking up something other
    than probe loss.
    """
    for plat in ("EPICv1", "EPICv2"):
        row = TABLE[("vidalbralo", plat)]
        assert row["probes_retained"] == row["probes_total"]
        assert row["median_shift"] == 0.0


@needs_table
def test_a_clock_spreading_weight_thinly_degrades_gracefully():
    """The published prediction, tested against our own measurement.

    PC clocks absorb probe loss because the rotation spreads the weight; the
    same argument applies to a BLUP clock over 319,607 probes. ``zhangblup``
    loses 32,764 probes on EPICv2 -- ten percent of the model -- and moves by
    less than a quarter of a year, while ``hannum`` loses seven of seventy-one
    and moves by several years.
    """
    blup = TABLE[("zhangblup", "EPICv2")]
    hannum = TABLE[("hannum", "EPICv2")]
    lost_blup = 1 - blup["probes_retained"] / blup["probes_total"]
    lost_hannum = 1 - hannum["probes_retained"] / hannum["probes_total"]
    assert lost_blup > lost_hannum, "the BLUP clock loses a larger fraction"
    assert abs(blup["median_shift"]) < abs(hannum["median_shift"]) / 10


@needs_table
def test_the_intervals_bracket_the_medians():
    for (cid, plat), row in TABLE.items():
        assert row["ci_lo"] <= row["median_shift"] <= row["ci_hi"], (cid, plat)


@needs_table
def test_probe_loss_reports_the_years(synthetic_betas):
    d = fa.FalconData(X=synthetic_betas.X, obs=synthetic_betas.obs,
                      modality="dna_methylation", platform="EPICv2")
    tab = fa.probe_loss(d, clocks="scoreable")
    assert "bias_years" in tab.columns and "bias_ci" in tab.columns
    row = tab.loc["hannum"]
    assert row["bias_years"] == pytest.approx(TABLE[("hannum", "EPICv2")]["median_shift"])
    assert "to" in row["bias_ci"]


@needs_table
def test_an_unmeasured_platform_says_nothing_rather_than_zero(synthetic_betas):
    """An absent row means 'not measured', never 'no bias'."""
    d = fa.FalconData(X=synthetic_betas.X, obs=synthetic_betas.obs,
                      modality="dna_methylation", platform="27K")
    tab = fa.probe_loss(d, clocks="scoreable")
    assert tab["bias_years"].isna().all()


@needs_table
def test_score_warns_when_the_shift_is_large(synthetic_betas):
    obs = synthetic_betas.obs.copy()
    obs["tissue"] = "whole blood"
    d = fa.FalconData(X=synthetic_betas.X, obs=obs, modality="dna_methylation",
                      platform="EPICv2")
    res = fa.score(d, clocks=["hannum", "vidalbralo"], min_coverage=0.0)
    w = {x["clock"]: x["message"] for x in res.manifest.warnings
         if x.get("category") == "platform_bias"}
    assert "hannum" in w and "median" in w["hannum"]
    assert "vidalbralo" not in w, "a clock that loses nothing must stay quiet"
    assert abs(TABLE[("hannum", "EPICv2")]["median_shift"]) >= BIAS_WARN


@needs_table
def test_the_shift_is_reported_and_never_applied(synthetic_betas):
    """The design decision, asserted. An automatic offset would be a second
    number nobody can trace."""
    obs = synthetic_betas.obs.copy()
    obs["tissue"] = "whole blood"
    plain = fa.FalconData(X=synthetic_betas.X, obs=obs,
                          modality="dna_methylation", platform="450K")
    v2 = fa.FalconData(X=synthetic_betas.X, obs=obs,
                       modality="dna_methylation", platform="EPICv2")
    a = fa.score(plain, clocks=["hannum"], min_coverage=0.0).scores["hannum"]
    b = fa.score(v2, clocks=["hannum"], min_coverage=0.0).scores["hannum"]
    assert np.array_equal(a.to_numpy(), b.to_numpy()), (
        "declaring a platform must change the warnings, not the numbers")
