"""Measurement error, propagated.

The arithmetic here is simple enough that most of these tests assert an exact
number against a hand-computed one. That is the point: an interval nobody can
reproduce by hand is another opaque output, and the whole reason for the feature
is that the score already was one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import falconage as fa
from falconage.models import ops
from falconage.uncertainty import UncertaintyError


# ---------------------------------------------------------------------------
# the derivative table
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,kwargs,x", [
    ("add", {"value": 3.0}, [0.1, -2.0, 5.0]),
    ("multiply", {"value": -1.5}, [0.1, -2.0, 5.0]),
    ("divide_by", {"value": 7.0}, [0.1, -2.0, 5.0]),
    ("one_minus", {}, [0.1, -2.0]),
    ("days_to_weeks", {}, [280.0, 200.0]),
    ("days_to_months", {}, [100.0]),
    ("anti_log_linear", {"adult_age": 20.0}, [-1.3, 0.4, 2.0]),
    ("log_linear", {"adult_age": 20.0}, [3.0, 45.0]),
    ("expit", {}, [-2.0, 0.0, 1.7]),
    ("exp", {}, [-1.0, 0.5]),
    ("anti_logp2", {}, [-1.0, 0.5]),
    ("anti_log_log", {}, [-0.5, 1.2]),
    ("cox_to_years", {"cox_mean": 1.0, "cox_std": 2.0,
                      "age_mean": 60.0, "age_std": 8.0}, [0.5, 3.0]),
    ("scale_and_shift", {"scale": 0.3, "offset": 2.0}, [1.0, -4.0]),
    ("petkovich_blood", {}, [0.2, 1.0]),
    ("stubbs_multitissue", {}, [-1.0, 0.5]),
    ("mortality_to_phenoage", {}, [-4.0, -3.0]),
    ("beta_to_m", {}, [0.2, 0.65]),
    ("m_to_beta", {}, [-1.0, 2.0]),
])
def test_every_derivative_matches_a_central_difference(name, kwargs, x):
    """A finite difference is not accurate enough to *use*, but it is a fine
    referee for an analytic derivative away from the kinks."""
    fn = ops.POSTPROCESS.get(name) or ops.PREPROCESS[name]
    d = ops.DERIVATIVE[name]
    x = np.asarray(x, dtype=float)
    h = 1e-6
    num = (fn(x + h, **kwargs) - fn(x - h, **kwargs)) / (2 * h)
    got = d(x, **kwargs)
    assert np.allclose(got, num, rtol=2e-4, atol=1e-8), f"{name}: {got} vs {num}"


def test_anti_log_linear_has_a_different_slope_below_zero():
    """The reason a single cohort-wide slope would be wrong.

    Horvath's transform is exponential below zero and linear above it, so a
    child's uncertainty and an adult's scale differently from the same raw SE.
    """
    d = ops.DERIVATIVE["anti_log_linear"]
    assert d(np.array([-2.0]))[0] == pytest.approx(21.0 * np.exp(-2.0))
    assert d(np.array([1.0]))[0] == pytest.approx(21.0)


def test_clip_kills_sensitivity_outside_its_bounds():
    d = ops.DERIVATIVE["clip"](np.array([-1.0, 0.5, 2.0]), low=0.0, high=1.0)
    assert list(d) == [0.0, 1.0, 0.0]


def test_a_chain_derivative_is_the_product_of_its_steps():
    chain = ({"op": "add", "value": 0.6955}, {"op": "anti_log_linear", "adult_age": 20})
    x = np.array([-0.4, 0.3])
    y, slope = ops.chain_derivative(x, chain, ops.POSTPROCESS)
    assert np.allclose(y, ops.apply_chain(x, chain, ops.POSTPROCESS))
    h = 1e-6
    num = (ops.apply_chain(x + h, chain, ops.POSTPROCESS)
           - ops.apply_chain(x - h, chain, ops.POSTPROCESS)) / (2 * h)
    assert np.allclose(slope, num, rtol=1e-5)


def test_a_chain_with_a_non_differentiable_op_refuses():
    """Skipping the step would turn an unquantifiable uncertainty into a small
    confident one, which is the failure mode worth being loud about."""
    from falconage.core.errors import ScoringError

    with pytest.raises(ScoringError, match="binarize"):
        ops.chain_derivative(np.array([0.4]), ({"op": "binarize"},), ops.PREPROCESS)


# ---------------------------------------------------------------------------
# the bundled reliability table
# ---------------------------------------------------------------------------

def test_the_bundled_icc_table_loads_and_is_plausible():
    icc = fa.uncertainty.load_probe_icc()
    assert len(icc) > 250_000
    assert icc.index.is_unique
    # ICC(2,1) can be negative when within-subject spread exceeds between.
    assert -1.0 <= icc.min() and icc.max() <= 1.0
    # Sugden 2020 reported 18% of 450K probes at ICC >= 0.5 in whole blood. This
    # is a different cohort and platform (ADNI, EPIC v1) and lands higher, but a
    # table where most probes looked excellent would mean the wrong column was
    # read.
    assert 0.2 < (icc >= 0.5).mean() < 0.6


def test_the_table_records_where_it_came_from():
    """An interval whose provenance is not recorded is worse than none."""
    src = fa.uncertainty.probe_icc_source()
    assert "10.1080/15592294.2024.2333660" in src["citation"]
    assert src["licence"] == "CC BY 4.0"
    assert len(src["sha256"]) == 64


def test_every_tier_a_clock_has_most_of_its_probes_in_the_table(registry):
    icc = fa.uncertainty.load_probe_icc()
    have = set(icc.index)
    for c in registry.filter(availability="A"):
        if not registry.has_coefficients(c.id) or c.formula:
            continue
        feats = registry.feature_ids(c.id)
        frac = sum(f in have for f in feats) / len(feats)
        assert frac > 0.5, f"{c.id}: only {frac:.0%} of probes have a published ICC"


# ---------------------------------------------------------------------------
# propagation
# ---------------------------------------------------------------------------

def _blood(data):
    obs = data.obs.copy()
    obs["tissue"] = "whole blood"
    return fa.FalconData(X=data.X, obs=obs, modality=data.modality,
                         platform=data.platform)


def test_the_propagation_is_reproducible_by_hand(synthetic_betas, registry):
    """Var = f'(raw)^2 * sum_j w_j^2 s_j^2 (1 - ICC_j), computed twice."""
    d = _blood(synthetic_betas)
    res = fa.score(d, clocks=["hannum"], min_coverage=0.0)
    got = fa.technical_se(res, d)

    feats, w = registry.coefficients("hannum")
    X = d.X.reindex(columns=list(feats)).to_numpy(dtype=np.float64)
    s2 = np.nanvar(X, axis=0, ddof=1)
    icc = fa.uncertainty.load_probe_icc().reindex(list(feats)).to_numpy()
    icc = np.where(np.isfinite(icc), icc, np.nanmedian(icc))
    var = ((w ** 2) * s2 * (1.0 - np.clip(icc, 0, 1))).sum()
    # hannum's postprocess is the identity, so the slope is 1.
    assert got.se["hannum"].iloc[0] == pytest.approx(np.sqrt(var), rel=1e-9)


def test_an_imputed_feature_widens_the_interval(synthetic_betas):
    """The part most likely to look wrong, so it gets its own test.

    An imputed probe is the cohort mean, not a measurement of this sample. It
    contributes its whole between-sample variance. Treating it as well measured
    would make worse data produce a narrower interval.
    """
    d = _blood(synthetic_betas)
    full = fa.technical_se(fa.score(d, clocks=["hannum"], min_coverage=0.0), d)

    X = d.X.copy()
    from falconage.registry import load

    feats = list(load().feature_ids("hannum"))
    X.loc[X.index[0], feats[:20]] = np.nan
    holed = fa.FalconData(X=X, obs=d.obs, modality=d.modality, platform=d.platform)
    part = fa.technical_se(fa.score(holed, clocks=["hannum"], min_coverage=0.0), holed)

    assert part.se["hannum"].iloc[0] > full.se["hannum"].iloc[0]


def test_horvath_technical_spread_lands_where_the_literature_says(synthetic_betas):
    """A sanity check against a published number, not against ourselves.

    Higgins-Chen 2022 measured deviations of up to 9 years between technical
    replicates of the same DNA. Two replicates differ with SD sqrt(2)*SE, so a
    95% range of about +-2*sqrt(2)*SE. An SE of a few years reproduces that;
    an SE of 0.1 or of 40 would mean the propagation is wrong by orders.
    """
    d = _blood(synthetic_betas)
    res = fa.score(d, clocks=["horvath2013"], min_coverage=0.0)
    se = float(fa.technical_se(res, d).se["horvath2013"].median())
    replicate_range = 2 * np.sqrt(2) * se
    assert 2.0 < replicate_range < 15.0, f"replicate range {replicate_range:.1f} years"


def test_diagnostics_say_how_much_was_guessed(synthetic_betas):
    d = _blood(synthetic_betas)
    res = fa.score(d, clocks=["horvath2013", "hannum"], min_coverage=0.0)
    dg = fa.technical_se(res, d).diagnostics
    assert set(dg.index) == {"horvath2013", "hannum"}
    assert (dg["method"] == "probe").all()
    assert (dg["n_icc_published"] + dg["n_icc_imputed"] == dg["n_features"]).all()
    assert dg["n_icc_published"].min() > 0
    # The implied cohort ICC is the number a reader can compare to a paper.
    assert dg["implied_cohort_icc"].between(-1, 1).all()


def test_a_user_supplied_icc_table_overrides_the_bundled_one(synthetic_betas):
    d = _blood(synthetic_betas)
    res = fa.score(d, clocks=["hannum"], min_coverage=0.0)
    perfect = pd.Series(1.0, index=d.X.columns)
    se = fa.technical_se(res, d, icc=perfect)
    assert np.allclose(se.se["hannum"], 0.0), "ICC 1 means no measurement error"
    assert se.source == {"source": "user-supplied"}


def test_the_clock_path_refuses_rather_than_inventing_an_icc(synthetic_betas):
    d = _blood(synthetic_betas)
    res = fa.score(d, clocks=["hannum"], min_coverage=0.0)
    out = fa.technical_se(res, d, source="clock")
    assert out.se.empty
    assert "no published technical ICC" in out.refused["hannum"]


def test_the_clock_path_works_where_an_icc_is_published(synthetic_betas, fresh_registry):
    import dataclasses

    from falconage.registry.registry import Reliability

    reg = fresh_registry
    reg._clocks["hannum"] = dataclasses.replace(
        reg.get("hannum"), reliability=Reliability(technical_icc=0.75, source="test"))
    d = _blood(synthetic_betas)
    res = fa.score(d, clocks=["hannum"], min_coverage=0.0, registry=reg)
    out = fa.technical_se(res, d, source="clock", registry=reg)
    expected = float(res.scores["hannum"].std(ddof=1)) * np.sqrt(0.25)
    assert out.se["hannum"].iloc[0] == pytest.approx(expected)


def test_interval_brackets_the_score(synthetic_betas):
    d = _blood(synthetic_betas)
    res = fa.score(d, clocks=["horvath2013"], min_coverage=0.0)
    iv = fa.interval(res, d, level=0.95)
    assert (iv["lo"] < iv["value"]).all() and (iv["value"] < iv["hi"]).all()
    # 95% is 1.959964 sigma either side.
    assert ((iv["hi"] - iv["lo"]) / iv["se"]).round(4).eq(3.9199).all()


def test_the_result_and_manifest_carry_the_interval(synthetic_betas):
    """So a table written to disk cannot lose the uncertainty it was reported
    with, and summary() shows it without being handed the matrix again."""
    d = _blood(synthetic_betas)
    res = fa.score(d, clocks=["hannum"], min_coverage=0.0)
    assert res.se is None
    fa.technical_se(res, d)
    assert res.se is not None and "technical_se" in res.summary().columns
    rec = res.manifest.config["technical_se"]
    assert rec["source"] == "auto"
    assert len(rec["reliability"]["sha256"]) == 64


# ---------------------------------------------------------------------------
# ICC from the user's own replicates
# ---------------------------------------------------------------------------

def test_icc_from_replicates_recovers_a_known_value(rng):
    """Built with a known between/within split, then measured."""
    n_subj, k, n_feat = 60, 2, 200
    between, within = 0.04, 0.01          # variances; true ICC(1,1) = 0.8
    truth = rng.normal(0.5, np.sqrt(between), size=(n_subj, n_feat))
    vals = np.repeat(truth, k, axis=0) + rng.normal(0, np.sqrt(within),
                                                    size=(n_subj * k, n_feat))
    ids = [f"R{i:04d}" for i in range(n_subj * k)]
    obs = pd.DataFrame({"subject": np.repeat([f"P{i}" for i in range(n_subj)], k)},
                       index=ids)
    d = fa.FalconData(X=pd.DataFrame(vals, index=ids,
                                     columns=[f"cg{i:08d}" for i in range(n_feat)]),
                      obs=obs, modality="dna_methylation")
    icc = fa.icc_from_replicates(d, "subject")
    assert len(icc) == n_feat
    assert icc.median() == pytest.approx(0.8, abs=0.06)


def test_icc_from_replicates_keeps_a_negative_value(rng):
    """A probe whose within-subject spread exceeds its between-subject spread
    has a negative ICC. Clipping it to zero would hide a probe that measures
    nothing behind one that measures a little."""
    n_subj, k, n_feat = 40, 2, 50
    truth = rng.normal(0.5, 0.001, size=(n_subj, n_feat))
    vals = np.repeat(truth, k, axis=0) + rng.normal(0, 0.05,
                                                    size=(n_subj * k, n_feat))
    ids = [f"R{i:04d}" for i in range(n_subj * k)]
    obs = pd.DataFrame({"subject": np.repeat([f"P{i}" for i in range(n_subj)], k)},
                       index=ids)
    d = fa.FalconData(X=pd.DataFrame(vals, index=ids,
                                     columns=[f"cg{i:08d}" for i in range(n_feat)]),
                      obs=obs, modality="dna_methylation")
    icc = fa.icc_from_replicates(d, "subject")
    assert icc.min() < 0


def test_icc_from_replicates_refuses_without_enough_replicated_subjects(rng):
    ids = [f"R{i}" for i in range(6)]
    obs = pd.DataFrame({"subject": ["A", "A", "B", "C", "D", "E"]}, index=ids)
    d = fa.FalconData(X=pd.DataFrame(rng.normal(0.5, 0.1, (6, 5)), index=ids,
                                     columns=[f"cg{i:08d}" for i in range(5)]),
                      obs=obs, modality="dna_methylation")
    with pytest.raises(UncertaintyError, match="more than one sample"):
        fa.icc_from_replicates(d, "subject")


# ---------------------------------------------------------------------------
# conformal prediction intervals
# ---------------------------------------------------------------------------

CAL = fa.uncertainty.load_conformal()
needs_cal = pytest.mark.skipif(
    CAL.empty, reason="conformal.csv absent; run python/tools/build_conformal.py")


@needs_cal
def test_only_age_scale_clocks_are_calibrated(registry):
    """A conformal band on a log-hazard is not an age interval, and quoting one
    in years would invent the units."""
    for cid in CAL["clock"].unique():
        assert registry.get(cid).scale_type == "age_years", cid


@needs_cal
def test_a_wider_level_gives_a_wider_interval():
    h = CAL[(CAL["clock"] == "horvath2013") & (CAL["age_band"] == "all")]
    w = h.set_index("level")["half_width"]
    assert w[0.80] <= w[0.90] <= w[0.95]


@needs_cal
def test_the_half_width_is_larger_than_the_mean_absolute_error():
    """A quantile of |residual| at 90% cannot be below its mean unless the
    calibration was computed on something other than the residuals."""
    a = CAL[(CAL["age_band"] == "all") & (np.isclose(CAL["level"], 0.90))]
    assert (a["half_width"] >= a["mae"]).all()


@needs_cal
def test_horvath_lands_where_the_literature_says():
    """Median absolute error of at least 3.6 years is the figure quoted against
    every clock as a limit on individual use (PMC12714307). A calibration that
    came out at half a year would mean the residuals were computed against the
    clock's own predictions rather than against age."""
    row = CAL[(CAL["clock"] == "horvath2013") & (CAL["age_band"] == "all")
              & (np.isclose(CAL["level"], 0.90))].iloc[0]
    assert 3.0 < row["mae"] < 12.0
    assert abs(row["median_bias"]) < 3.0, "Horvath should be near-unbiased on age"


@needs_cal
def test_prediction_error_is_much_larger_than_measurement_error(synthetic_betas):
    """The two uncertainties answer different questions, and conflating them is
    the mistake this pair of functions exists to prevent."""
    d = _blood(synthetic_betas)
    res = fa.score(d, clocks=["horvath2013"], min_coverage=0.0)
    tech = float(fa.technical_se(res, d).se["horvath2013"].median())
    conf = float(fa.conformal_interval(res, level=0.90)["half_width"].iloc[0])
    assert conf > 2 * tech


@needs_cal
def test_the_interval_declines_to_claim_exchangeability(synthetic_betas):
    """The guarantee is conditional on the new samples being drawn like the
    calibration ones, and nothing here can check that."""
    d = _blood(synthetic_betas)
    res = fa.score(d, clocks=["horvath2013"], min_coverage=0.0)
    iv = fa.conformal_interval(res)
    assert not iv["exchangeable"].any()
    assert (iv["lo"] < iv["value"]).all() and (iv["value"] < iv["hi"]).all()
    assert iv["n_calibration"].min() >= 40


@needs_cal
def test_an_uncalibrated_clock_refuses_rather_than_borrowing_a_width(synthetic_betas):
    d = _blood(synthetic_betas)
    res = fa.score(d, clocks=["zhangmortality"], min_coverage=0.0)
    with pytest.raises(UncertaintyError, match="scale is age in years"):
        fa.conformal_interval(res)


@needs_cal
def test_the_clocks_with_no_fixed_origin_are_not_calibrated():
    """This started as an observation and became a registry change.

    The first calibration put DamAge and AdaptAge 37-150 years from
    chronological age, which prompted the measurement in
    ``scratch/ying_scale.py``: their slope against age is fine (0.967 pooled for
    DamAge, better than DNAmPhenoAge) but their offset swings 162 years between
    cohorts, as near mirror images of each other. They are now
    ``age_years_relative``, and a prediction band against chronological age is
    not defined for a clock whose zero moves -- so the calibration excludes
    them rather than quoting a 90-year half-width at somebody.
    """
    import falconage as fa

    reg = fa.registry.load()
    relative = {c.id for c in reg if c.scale_type == "age_years_relative"}
    assert relative, "the scale should still have members"
    assert not (set(CAL["clock"]) & relative)


# ---------------------------------------------------------------------------
# the quantile rule itself
# ---------------------------------------------------------------------------
# Split conformal's guarantee rests on one specific choice: the
# ceil((n+1)*level)-th order statistic of the absolute residuals. numpy's
# quantile interpolates between order statistics and lands below it, so
# substituting np.quantile silently under-covers while still carrying the word
# "guarantee". That substitution shipped once, in the per-age-band rows only,
# and widened Horvath's 50-70 band from 11.5 to 14.9 years when corrected. These
# two tests exist so it cannot come back quietly.


def _conformal_half_width(absr, level):
    """The rule, restated here rather than imported.

    ``python/tools/`` is not on the path for a plain ``pytest python/tests`` run
    and is not part of the wheel. Restating the one line is the point: if the
    builder's version drifts from this one, the next test fails.
    """
    n = len(absr)
    k = int(np.ceil((n + 1) * level))
    return float(np.sort(absr)[min(k, n) - 1])


@pytest.mark.parametrize("n", [20, 36, 40, 55, 61, 164])
@pytest.mark.parametrize("level", [0.80, 0.90, 0.95])
def test_the_conformal_quantile_is_never_the_interpolated_one(rng, n, level):
    """np.quantile is <= the order statistic, so it would under-cover."""
    absr = np.abs(rng.normal(0, 7.0, size=n))
    conformal = _conformal_half_width(absr, level)
    interpolated = float(np.quantile(absr, level))
    assert conformal >= interpolated
    # And it is genuinely one of the observed residuals, not a value between two.
    assert np.isclose(np.sort(absr), conformal).any()


@needs_cal
def test_every_shipped_half_width_obeys_the_order_statistic_rule():
    """Empirical coverage on the calibration set must reach the stated level.

    A width taken from the ceil((n+1)*level)-th order statistic covers at least
    ceil((n+1)*level)/n of the calibration residuals by construction. An
    interpolated quantile does not, and on the band rows it did not.
    """
    for _, r in CAL.iterrows():
        n, level = int(r["n_calibration"]), float(r["level"])
        k = int(np.ceil((n + 1) * level))
        if not bool(r["exact"]):
            assert k > n, (
                f"{r['clock']} {r['age_band']} is flagged inexact but "
                f"ceil((n+1)*level)={k} <= n={n}, so the level was attainable")
            continue
        assert k <= n, f"{r['clock']} {r['age_band']}: level {level} needs n>={k}, has {n}"
        # The guaranteed coverage this width buys on its own calibration set.
        assert k / n >= level, (
            f"{r['clock']} {r['age_band']}: covers {k}/{n}={k/n:.3f} of the "
            f"calibration residuals, below the stated {level}")
