"""Study design: how many samples, and did the difference hold up.

Both of these answer a question a laboratory asks out loud. ``power`` is asked
before any array is run; ``consensus`` is asked after, and its job is to say no.
"""

from __future__ import annotations

import numpy as np
import pytest

import falconage as fa
from falconage.core.errors import AnalysisError


def _blood(data, **cols):
    obs = data.obs.copy()
    obs["tissue"] = "whole blood"
    for k, v in cols.items():
        obs[k] = v
    return fa.FalconData(X=data.X, obs=obs, modality=data.modality,
                         platform=data.platform)


# ---------------------------------------------------------------------------
# power
# ---------------------------------------------------------------------------

def test_power_matches_the_textbook_formula():
    """n = 2 (z_{1-a/2} + z_b)^2 sd^2 / delta^2, computed twice."""
    from scipy.stats import norm

    p = fa.power("hannum", effect=2.0, sd=5.0)
    z = norm.ppf(0.975) + norm.ppf(0.80)
    assert p.n_per_group == int(np.ceil(2 * z ** 2 * 25.0 / 4.0))
    assert p.n_total == 2 * p.n_per_group


def test_a_smaller_effect_costs_the_square():
    """Halving the effect quadruples the sample size. The reason a guessed
    effect size is a guessed study."""
    big = fa.power("hannum", effect=4.0, sd=5.0).n_per_group
    small = fa.power("hannum", effect=2.0, sd=5.0).n_per_group
    assert small == pytest.approx(4 * big, rel=0.02)


def test_power_refuses_to_default_the_sd():
    with pytest.raises(AnalysisError, match="sd squared"):
        fa.power("hannum", effect=1.0)


def test_power_reads_the_sd_off_a_pilot(synthetic_betas):
    d = _blood(synthetic_betas)
    res = fa.score(d, clocks=["hannum"], min_coverage=0.0)
    p = fa.power("hannum", effect=1.0, result=res)
    assert p.sd == pytest.approx(float(res.scores["hannum"].std(ddof=1)))
    assert "cohort SD" in p.assumptions


def test_reliability_splits_the_sample_size_into_signal_and_noise():
    """The arithmetic behind '3-16 replicates, or 1-2 with a PC clock'.

    At ICC 0.9 a tenth of the observed variance is the instrument, so a tenth
    of the sample size is buying nothing but an average of it.
    """
    p = fa.power("horvath2013", effect=1.0, sd=5.0, icc=0.9)
    assert p.n_if_perfectly_measured < p.n_per_group
    assert p.n_if_perfectly_measured == pytest.approx(0.9 * p.n_per_group, rel=0.02)


def test_replicates_trade_arrays_against_recruitment():
    one = fa.power("horvath2013", effect=1.0, sd=5.0, icc=0.6, replicates=1)
    four = fa.power("horvath2013", effect=1.0, sd=5.0, icc=0.6, replicates=4)
    assert four.n_per_group < one.n_per_group
    assert four.n_per_group >= one.n_if_perfectly_measured, (
        "averaging replicates approaches the noise-free n, never beats it")


def test_the_measured_icc_comes_from_the_pilot_when_there_is_one(synthetic_betas):
    d = _blood(synthetic_betas)
    res = fa.score(d, clocks=["horvath2013"], min_coverage=0.0)
    fa.technical_se(res, d)
    p = fa.power("horvath2013", effect=1.0, result=res)
    assert p.icc is not None
    assert "measured on this cohort" in p.icc_source


def test_a_cohort_narrower_than_its_own_noise_says_so(synthetic_betas):
    """Not clipped to a comfortable zero.

    The synthetic fixture drifts each probe by 0.0012 per year, which is far
    less true signal than a real cohort carries, so Horvath's propagated
    measurement error exceeds the spread between its samples. That is a real
    state of affairs -- it is what a study with too narrow an age range looks
    like -- and the honest output is a negative implied ICC and a refusal to
    quote a reliability-adjusted n, not an ICC of 0.0 reported without comment.
    """
    d = _blood(synthetic_betas)
    res = fa.score(d, clocks=["horvath2013"], min_coverage=0.0)
    fa.technical_se(res, d)
    p = fa.power("horvath2013", effect=1.0, result=res)
    assert p.icc <= 0
    assert "as large as the spread" in p.icc_source
    assert p.n_if_perfectly_measured is None
    # And the plain sample size is still returned -- it does not depend on ICC.
    assert p.n_per_group > 0


def test_replicates_do_nothing_without_a_reliability_figure():
    """Honest: with no ICC there is no way to know how much of sd is noise."""
    a = fa.power("hannum", effect=1.0, sd=4.0, replicates=1)
    b = fa.power("hannum", effect=1.0, sd=4.0, replicates=8)
    assert a.n_per_group == b.n_per_group
    assert a.icc is None and a.icc_source == "not established"


def test_detectable_effect_inverts_power():
    n = fa.power("hannum", effect=2.0, sd=5.0).n_per_group
    got = fa.detectable_effect("hannum", n, sd=5.0)
    assert got == pytest.approx(2.0, rel=0.02)


# ---------------------------------------------------------------------------
# consensus
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def scored_arms(synthetic_betas):
    """A scored cohort with two arms that are balanced on age.

    Alternating rather than splitting in half. The fixture's age runs linearly
    with the row index, so the obvious first-half/second-half split makes the
    control arm young and the treated arm old -- and a pace clock, which is
    tested on its raw score because a residual is not defined for a rate,
    correctly reports that difference. The test would then be measuring the
    age confound it built, not the protocol.
    """
    d = _blood(synthetic_betas)
    res = fa.score(d, clocks="compatible")
    res.obs = res.obs.copy()
    res.obs["arm"] = ["ctrl" if i % 2 else "treat" for i in range(res.scores.shape[0])]
    return res


def _shift(res, clocks, amount):
    import copy

    out = copy.copy(res)
    out.scores = res.scores.copy()
    treat = (res.obs["arm"] == "treat").to_numpy()
    for cid in clocks:
        out.scores.loc[treat, cid] = out.scores.loc[treat, cid] + amount
    return out


def _by_generation(res):
    gens: dict[str, list[str]] = {}
    for cid in res.scores.columns:
        gens.setdefault(res.registry.get(cid).generation, []).append(cid)
    return gens


def test_one_clock_moving_is_called_a_false_positive(scored_arms):
    """The published signature: five of six intervention datasets had exactly
    one significant clock, first-generation every time, and four of the five
    lost it under correction (PMC11526921)."""
    rep = fa.consensus(_shift(scored_arms, ["horvath2013"], 12.0), "arm",
                       reference="ctrl")
    assert rep.verdict == "unsupported"
    assert "single significant clock" in rep.why
    assert rep.table.loc["horvath2013", "sig_bonferroni"]


def test_only_age_trained_clocks_moving_is_also_unsupported(scored_arms):
    first = _by_generation(scored_arms).get("first", [])
    assert len(first) >= 2, "fixture must carry more than one first-generation clock"
    rep = fa.consensus(_shift(scored_arms, first, 15.0), "arm", reference="ctrl")
    assert rep.verdict == "unsupported"
    assert "only age-trained clocks" in rep.why


def test_a_change_across_generations_is_supported(scored_arms):
    gens = _by_generation(scored_arms)
    assert "first" in gens and "second" in gens, "fixture must span generations"
    move = gens["first"][:2] + gens["second"][:2]
    rep = fa.consensus(_shift(scored_arms, move, 15.0), "arm", reference="ctrl")
    assert rep.verdict == "supported", rep.why
    assert "generations" in rep.why


def test_nothing_moving_is_unsupported_and_says_so(scored_arms):
    rep = fa.consensus(scored_arms, "arm", reference="ctrl")
    assert rep.verdict == "unsupported"
    assert "nothing survives correction" in rep.why, rep.why


def test_the_verdict_always_carries_its_counts(scored_arms):
    """A verdict without its arithmetic is an oracle."""
    res = _shift(scored_arms, ["horvath2013"], 12.0)
    rep = fa.consensus(res, "arm", reference="ctrl")
    assert "of" in rep.why and "Bonferroni" in rep.why
    assert rep.n_tests == res.scores.shape[1]


def test_bonferroni_corrects_over_the_tests_actually_run(scored_arms):
    res = _shift(scored_arms, ["horvath2013"], 12.0)
    rep = fa.consensus(res, "arm", reference="ctrl")
    t = rep.table
    assert np.allclose(t["p_bonferroni"], np.clip(t["p"] * rep.n_tests, 0, 1))


def test_a_pace_clock_is_tested_on_its_score_not_a_residual(scored_arms):
    """Subtracting chronological age from a rate is the units error LEGAL_OPS
    exists to prevent, and it must not sneak back in here."""
    rep = fa.consensus(scored_arms, "arm", reference="ctrl")
    reg = scored_arms.registry
    for cid, row in rep.table.iterrows():
        legal = "acceleration" in reg.get(cid).legal_operations
        assert row["basis"] == ("residual" if legal else "score"), cid


def test_consensus_needs_exactly_two_groups(scored_arms):
    import copy

    res = copy.copy(scored_arms)
    res.obs = res.obs.copy()
    res.obs["arm"] = ["a", "b", "c"] * (res.scores.shape[0] // 3)
    with pytest.raises(AnalysisError, match="compares exactly two"):
        fa.consensus(res, "arm")
