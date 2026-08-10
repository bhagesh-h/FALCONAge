"""The two figure conventions the package could not draw until now.

Everything here is deliberately small. The whole suite has to fit in an 8 GB
Docker allocation alongside the registry, so these use 120-sample cohorts and
40-row association tables -- enough to exercise the estimator arithmetic, and
nowhere near enough to matter for memory. The heavy figures (the clock atlas,
the scatter matrix) are exercised by test/run_all.py against the real corpus,
not here.

The Kaplan-Meier and log-rank implementations are asserted against hand-worked
values rather than against another library, because the point of writing them
in twenty lines was to avoid the dependency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import falconage as fa

pytest.importorskip("matplotlib")


# ---------------------------------------------------------------------------
# the estimator itself
# ---------------------------------------------------------------------------

def test_km_matches_a_hand_worked_example():
    """Five subjects, one censored, worked through by hand.

    times  1, 2, 3, 4, 5      events  1, 0, 1, 1, 0
      t=1: 5 at risk, 1 event -> S = 1 - 1/5           = 0.8
      t=2: censored, leaves the risk set, S unchanged
      t=3: 3 at risk, 1 event -> S = 0.8 * (1 - 1/3)   = 0.5333...
      t=4: 2 at risk, 1 event -> S = 0.5333 * (1 - 1/2)= 0.2666...
      t=5: censored
    """
    from falconage.plot.outcomes import _km_curve

    t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    e = np.array([1, 0, 1, 1, 0])
    times, surv, at_risk = _km_curve(t, e)

    np.testing.assert_allclose(times, [0, 1, 3, 4])
    np.testing.assert_allclose(surv, [1.0, 0.8, 0.8 * 2 / 3, 0.8 * 2 / 3 * 0.5])
    np.testing.assert_array_equal(at_risk, [5, 5, 3, 2])


def test_censoring_is_not_an_event():
    """The distinction the product-limit form exists for.

    Same times, but every observation censored: survival never drops. A naive
    one-minus-cumulative-proportion would fall to zero.
    """
    from falconage.plot.outcomes import _km_curve

    _, surv, _ = _km_curve(np.arange(1.0, 6.0), np.zeros(5))
    assert (surv == 1.0).all()


def test_logrank_finds_no_difference_between_identical_groups():
    from falconage.plot.outcomes import _logrank

    t = np.arange(1.0, 21.0)
    e = np.ones(20)
    p = _logrank(t, e, t.copy(), e.copy())
    assert p > 0.9, "identical groups should be nowhere near significant"


def test_logrank_separates_groups_that_plainly_differ():
    from falconage.plot.outcomes import _logrank

    early = np.arange(1.0, 21.0)              # all die 1-20
    late = np.arange(40.0, 60.0)              # all die 40-59
    p = _logrank(early, np.ones(20), late, np.ones(20))
    assert p < 1e-6


# ---------------------------------------------------------------------------
# kaplan_meier
# ---------------------------------------------------------------------------

def _survival_cohort(n=120, seed=20260809, effect=True):
    """A cohort where fast agers die sooner, if `effect`."""
    rng = np.random.default_rng(seed)
    age = rng.uniform(40, 80, n)
    # Predicted age = chronological + a per-subject acceleration.
    accel = rng.normal(0, 5, n)
    hazard = np.exp(0.12 * accel) if effect else np.ones(n)
    time = rng.exponential(40.0 / hazard)
    return age, accel, time, (time < 35).astype(int)


def _result_with_survival(n=120, effect=True):
    """A 120-subject cohort carrying one clock's features and nothing else.

    Not the shared `synthetic_betas` fixture, for two reasons. It has 24
    samples, and a 10% tail of 24 is two subjects -- no log-rank test can find
    anything there, which is a property of the fixture rather than of the
    figure. And it spans the union of every tier A clock's features, roughly
    320,000 columns; scaling that to 120 rows would be ~300 MB for a test that
    needs one clock. Horvath's 353 features at 120 samples is 340 kB.
    """
    reg = fa.registry.load()
    feats = list(reg.coefficients("horvath2013")[0])

    rng = np.random.default_rng(20260809)
    X = pd.DataFrame(rng.uniform(0.05, 0.95, (n, len(feats))),
                     columns=feats, index=[f"s{i:03d}" for i in range(n)])
    age, accel, time, event = _survival_cohort(n, effect=effect)
    obs = pd.DataFrame({"age": age, "time": time, "event": event}, index=X.index)

    d = fa.FalconData(X=X, obs=obs, modality="dna_methylation", platform="450K")
    res = fa.score(d, clocks=["horvath2013"])
    # Drive the score directly so acceleration is the quantity under test
    # rather than whatever random betas happened to produce.
    res.scores["horvath2013"] = pd.Series(age + accel, index=X.index)
    return res


def test_kaplan_meier_returns_the_two_tails():
    res = _result_with_survival()
    fig, table = fa.plot.kaplan_meier(res, "horvath2013",
                                      time_col="time", event_col="event")
    assert list(table["group"]) == ["slow", "fast"]
    assert (table["n"] > 0).all()
    # Two step curves plus the reference lines the theme draws.
    assert len(fig.axes[0].lines) >= 2


def test_kaplan_meier_detects_a_real_hazard_difference():
    """The figure has to be able to show the thing it is drawn for."""
    res = _result_with_survival(effect=True)
    _, table = fa.plot.kaplan_meier(res, "horvath2013",
                                    time_col="time", event_col="event")
    assert table["logrank_p"].iloc[0] < 0.05


def test_kaplan_meier_refuses_when_nothing_happened():
    from falconage.plot import NothingToPlot

    res = _result_with_survival()
    res.obs["event"] = 0
    with pytest.raises(NothingToPlot, match="every subject is censored"):
        fa.plot.kaplan_meier(res, "horvath2013",
                             time_col="time", event_col="event")


def test_kaplan_meier_refuses_a_cohort_too_small_to_split():
    from falconage.plot import NothingToPlot

    res = _result_with_survival()
    res.obs.loc[res.obs.index[3:], "time"] = np.nan
    with pytest.raises(NothingToPlot, match="at least 8"):
        fa.plot.kaplan_meier(res, "horvath2013",
                             time_col="time", event_col="event")


# ---------------------------------------------------------------------------
# volcano
# ---------------------------------------------------------------------------

def _assoc_table(n=200, seed=7):
    """200 tests: five real, the rest null.

    200 rather than 40 so the raw-versus-BH gap is present by construction. At
    n = 40 a uniform null yields about two rows under a raw 0.05, and the first
    version of this fixture happened to give exactly five -- the same count as
    the planted hits, so the assertion that the two thresholds differ compared
    5 with 5 and failed. At 200 the null contributes about ten, none of which
    survive the correction.
    """
    rng = np.random.default_rng(seed)
    beta = rng.normal(0, 1, n)
    p = rng.uniform(0, 1, n)
    p[:5] = rng.uniform(1e-9, 1e-6, 5)          # a handful of real hits
    df = pd.DataFrame({"n": 200, "beta": beta, "se": 0.1,
                       "t": beta / 0.1, "p": p},
                      index=[f"clock{i}" for i in range(n)])
    from falconage.analysis import _bh
    df["q"] = _bh(df["p"].to_numpy())
    return df.sort_values("p")


def test_volcano_marks_the_bh_hits_not_the_raw_p_hits():
    """The distinction that matters across many tests.

    Several nulls land under a raw 0.05 by chance; BH keeps only what survives
    the multiplicity correction. Drawing the raw cut is the common error, and
    it calls noise significant.
    """
    d = _assoc_table()
    _, out = fa.plot.volcano(d, fdr=0.05)

    raw_hits = int((d["p"] <= 0.05).sum())
    bh_hits = int((d["q"] <= 0.05).sum())
    assert bh_hits < raw_hits, "the fixture must exercise the difference"
    assert bh_hits >= 5, "the five planted hits should survive BH"
    assert len(out) == len(d)


def test_volcano_says_which_columns_it_wanted():
    from falconage.core.errors import AnalysisError

    d = _assoc_table().rename(columns={"beta": "estimate"})
    with pytest.raises(AnalysisError, match="no 'beta' column"):
        fa.plot.volcano(d)


def test_volcano_survives_a_p_value_of_exactly_zero():
    """Underflow to 0.0 is routine with large n; -log10(0) is an infinity that
    silently removes the most significant point from the figure."""
    d = _assoc_table()
    d.loc[d.index[0], "p"] = 0.0
    fig, out = fa.plot.volcano(d)
    ys = np.concatenate([c.get_offsets()[:, 1] for c in fig.axes[0].collections])
    assert np.isfinite(ys).all()


def test_volcano_refuses_an_empty_table():
    from falconage.plot import NothingToPlot

    d = _assoc_table().head(0)
    with pytest.raises(NothingToPlot):
        fa.plot.volcano(d)


def test_volcano_runs_on_a_real_associate_result(synthetic_betas):
    """End to end: the frame associate() produces is the frame volcano takes."""
    res = fa.score(synthetic_betas, clocks=["horvath2013", "hannum", "lin"])
    rng = np.random.default_rng(3)
    res.obs = res.obs.copy()
    res.obs["outcome"] = rng.normal(0, 1, res.scores.shape[0])
    assoc = fa.associate(res, "outcome", covariates=())
    fig, out = fa.plot.volcano(assoc)
    assert len(out) == 3


def test_the_two_colorscheme_copies_have_not_drifted():
    """`colorscheme.yaml` exists twice and nothing kept them in step.

    The repository root copy is the one a user edits and the one the README
    points at; `plot/colorscheme.yaml` is the one that ships in the wheel and
    the only one `spec.load()` reads. They were byte-identical and maintained
    by hand, so adding a plot to the root copy changed nothing at runtime --
    which is precisely how the volcano plot came to raise KeyError from its own
    text lookup while the text sat in the file next to it.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[3] / "colorscheme.yaml"
    packaged = Path(fa.plot.spec.PACKAGED)
    assert root.exists(), f"no colour scheme at {root}"
    assert root.read_bytes() == packaged.read_bytes(), (
        "colorscheme.yaml and python/src/falconage/plot/colorscheme.yaml have "
        "drifted. The packaged copy is the one that loads; copy the root one "
        "over it.")


def test_every_plot_function_has_text_defined():
    """A figure whose text is missing raises at draw time, not at import.

    So the failure lands on a user rendering a report rather than on the person
    who added the function, and only for the figures they happen to draw.
    """
    import inspect

    scheme = fa.plot.spec.load()["plots"]

    # Functions only. `__all__` also carries the NothingToPlot exception and
    # the spec submodule, neither of which draws anything -- the first version
    # of this test used `callable()` and duly demanded colour-scheme text for
    # an exception class.
    helpers = {"save_all", "palette", "semantic", "theme_value",
               "group_colours", "platform_colour"}
    undefined = [
        name for name in fa.plot.__all__
        if inspect.isfunction(getattr(fa.plot, name, None))
        and name not in helpers
        and name not in scheme
    ]
    assert not undefined, f"no colorscheme.yaml text for: {', '.join(undefined)}"
