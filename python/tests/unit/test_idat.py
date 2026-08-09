"""The raw-IDAT chain.

The pieces that need no network are tested here against synthetic signal with a
known answer. The chain end to end, against real IDATs and the published betas
for the same physical samples, is in the corpus tests -- it needs a manifest
download and 850,000 probes, and neither belongs in the unit suite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from falconage.core.errors import DataError
from falconage.preprocess import idat as I
from falconage.preprocess.manifest import MANIFESTS, detect_manifest_platform


# ---------------------------------------------------------------------------
# platform detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,expect", [
    (27_578, "27K"), (622_399, "450K"), (1_051_815, "EPICv1"), (1_105_209, "EPICv2"),
    (1_051_539, "EPICv1"),      # a real chip, slightly off the nominal count
])
def test_platform_is_detected_from_the_address_count(n, expect):
    assert detect_manifest_platform(n) == expect


def test_an_unrecognised_array_refuses_rather_than_guessing():
    """A misdetected platform maps addresses to the wrong probes and returns a
    full matrix of plausible wrong betas, which nothing downstream can catch."""
    with pytest.raises(DataError, match="matches no known array"):
        detect_manifest_platform(300_000)


def test_epic_v1_and_v2_are_close_enough_to_need_the_margin_rule():
    """The finding that put the second condition in the detector.

    27K, 450K and EPIC are separated by factors. EPIC v1 and v2 are separated by
    five percent, so a tolerance wide enough for chip-to-chip variation is also
    wide enough to match both -- and picking the closer one silently would map
    addresses to the wrong probes.
    """
    counts = sorted(v[1] for v in MANIFESTS.values())
    ratios = [hi / lo for lo, hi in zip(counts, counts[1:])]
    assert min(ratios) < 1.1, "if this ever holds, the margin rule is unnecessary"
    assert sorted(ratios)[1] > 1.5, "everything else really is a factor apart"


def test_a_count_between_epic_v1_and_v2_is_refused_as_ambiguous():
    midpoint = (MANIFESTS["EPICv1"][1] + MANIFESTS["EPICv2"][1]) // 2
    with pytest.raises(DataError, match="ambiguous between"):
        detect_manifest_platform(midpoint)


# ---------------------------------------------------------------------------
# Huber's estimator
# ---------------------------------------------------------------------------

def test_huber_recovers_a_clean_gaussian(rng):
    x = rng.normal(500.0, 100.0, size=20_000)
    mu, sd = I._huber(x)
    assert mu == pytest.approx(500.0, abs=5.0)
    assert sd == pytest.approx(100.0, rel=0.05)


def test_huber_ignores_the_bright_tail_a_mean_would_follow(rng):
    """The reason minfi uses it here. The upper tail of the out-of-band signal
    is cross-hybridisation, not background, and it drags a plain mean up."""
    clean = rng.normal(500.0, 100.0, size=20_000)
    contaminated = np.concatenate([clean, rng.uniform(5_000, 50_000, size=1_000)])
    mu, sd = I._huber(contaminated)
    assert mu == pytest.approx(500.0, abs=20.0)
    assert float(np.mean(contaminated)) > 1_400, "the mean really is dragged"


# ---------------------------------------------------------------------------
# normal-exponential deconvolution
# ---------------------------------------------------------------------------

def test_normexp_subtracts_the_background_it_is_given():
    """A bright probe is reduced by roughly the background mean; the exact
    amount also carries the sigma^2/alpha term, so this is a bound not an
    equality."""
    out = I._normexp_signal(np.array([10_000.0]), mu=500.0, sigma=100.0, alpha=3_000.0)
    assert 9_000 < out[0] < 9_600


def test_normexp_is_finite_far_below_background():
    """The case that breaks a naive implementation: the correction is a density
    over a tail probability, and both underflow. Computed in logs it stays
    finite, and this is every undetected probe on the array."""
    out = I._normexp_signal(np.array([1.0, 10.0, 50.0]), mu=5_000.0, sigma=800.0,
                            alpha=3_000.0)
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0.0)


def test_normexp_is_monotone():
    x = np.array([100.0, 500.0, 1_000.0, 5_000.0, 20_000.0])
    out = I._normexp_signal(x, mu=400.0, sigma=150.0, alpha=2_000.0)
    assert np.all(np.diff(out) > 0)


# ---------------------------------------------------------------------------
# the pipeline object, on synthetic signal
# ---------------------------------------------------------------------------

def _signal(rng, n_ii=400, n_ig=150, n_ir=150, bg_grn=400.0, bg_red=900.0):
    """A RawSignal with known background and a known beta for every probe."""
    n = n_ii + n_ig + n_ir
    beta = rng.uniform(0.05, 0.95, size=n)
    total = rng.uniform(4_000, 12_000, size=n)
    m = beta * total
    u = (1 - beta) * total
    types = np.array(["II"] * n_ii + ["I"] * (n_ig + n_ir))
    chan = np.array([""] * n_ii + ["Grn"] * n_ig + ["Red"] * n_ir)
    idx = pd.Index([f"cg{i:08d}" for i in range(n)], name="feature_id")
    return I.RawSignal(
        meth=pd.Series(m, index=idx, name="S1"),
        unmeth=pd.Series(u, index=idx, name="S1"),
        probe_type=pd.Series(types, index=idx),
        channel=pd.Series(chan, index=idx),
        oob_grn=rng.normal(bg_grn, 120.0, size=6_000),
        oob_red=rng.normal(bg_red, 260.0, size=6_000),
        platform="EPICv1", sample_id="S1", steps=["decoded"]), beta


def test_betas_are_the_published_formula(rng):
    sig, beta = _signal(rng)
    got = sig.betas(detection_p=None).to_numpy()
    m = sig.meth.to_numpy()
    u = sig.unmeth.to_numpy()
    assert np.allclose(got, m / (m + u + I.BETA_OFFSET))
    # The offset pulls every beta slightly toward zero, and by design.
    assert np.all(got <= beta + 1e-12)


def test_poobah_detects_bright_probes_and_fails_dim_ones(rng):
    sig, _ = _signal(rng)
    m = sig.meth.to_numpy().copy()
    u = sig.unmeth.to_numpy().copy()
    m[:20] = 30.0          # well under either background
    u[:20] = 30.0
    sig.meth = pd.Series(m, index=sig.meth.index)
    sig.unmeth = pd.Series(u, index=sig.unmeth.index)

    sig = I.poobah(sig)
    p = sig.detection_p.to_numpy()
    assert np.all(p[:20] > I.DETECTION_P), "background-level probes must fail"
    assert np.mean(p[20:] < I.DETECTION_P) > 0.95, "bright probes must pass"


def test_an_undetected_probe_becomes_nan_not_a_number(rng):
    """The point of computing detection. A beta of 0.43 from two noise readings
    is indistinguishable from a real 0.43 once it is in the matrix."""
    sig, _ = _signal(rng)
    m, u = sig.meth.to_numpy().copy(), sig.unmeth.to_numpy().copy()
    m[:20], u[:20] = 30.0, 30.0
    sig.meth = pd.Series(m, index=sig.meth.index)
    sig.unmeth = pd.Series(u, index=sig.unmeth.index)
    sig = I.poobah(sig)

    kept = sig.betas(detection_p=None)
    dropped = sig.betas(detection_p=I.DETECTION_P)
    assert kept.iloc[:20].notna().all()
    assert dropped.iloc[:20].isna().all()


def test_noob_refuses_to_run_before_detection(rng):
    """Order is load-bearing: pOOBAH's null is the *uncorrected* out-of-band
    distribution, and correcting first removes the background from its own
    null."""
    sig, _ = _signal(rng)
    with pytest.raises(DataError, match="before noob"):
        I.noob(sig)


def test_noob_reduces_signal_and_keeps_it_positive(rng):
    sig, _ = _signal(rng)
    before_m = sig.meth.to_numpy().copy()
    sig = I.noob(I.poobah(sig))
    after_m = sig.meth.to_numpy()
    assert np.all(after_m >= 0)
    assert np.median(after_m) < np.median(before_m), "background is subtracted"
    assert "noob" in sig.steps


def test_noob_increases_contrast_rather_than_shrinking_toward_the_middle(rng):
    """The direction background correction actually moves betas, which is the
    opposite of the intuition.

    Subtracting a constant `b` from both beads: a low-beta probe at M=1000,
    U=9000 reads 0.099, and after b=400 it reads 600/9300 = 0.065. A high-beta
    probe at M=9000, U=1000 reads 0.891 and goes to 8600/9300 = 0.925. Both move
    *away* from the middle -- background correction sharpens the bimodality,
    which is what it is for. The first version of this test asserted the
    reverse and the implementation was right.
    """
    sig, _ = _signal(rng)
    before = sig.betas(detection_p=None).to_numpy()
    after = I.noob(I.poobah(sig)).betas(detection_p=None).to_numpy()
    low, high = before < 0.3, before > 0.7
    assert np.mean(after[low] < before[low]) > 0.9
    assert np.mean(after[high] > before[high]) > 0.9
    # And by hand, on the two numbers in the docstring.
    one = I.RawSignal(
        meth=pd.Series([1000.0, 9000.0], index=["a", "b"]),
        unmeth=pd.Series([9000.0, 1000.0], index=["a", "b"]),
        probe_type=pd.Series(["II", "II"], index=["a", "b"]),
        channel=pd.Series(["", ""], index=["a", "b"]),
        oob_grn=np.full(500, 400.0), oob_red=np.full(500, 400.0),
        platform="EPICv1", sample_id="S", steps=[])
    b0 = one.betas(detection_p=None)
    assert b0["a"] == pytest.approx(1000 / 10100)
    assert b0["b"] == pytest.approx(9000 / 10100)


def test_dye_bias_is_off_by_default_in_the_entry_point():
    """Measured on real IDATs it moves the median beta by +0.10 to +0.12, which
    is far more than a dye correction should. It stays available and opt-in."""
    import inspect

    assert inspect.signature(I.idat_to_betas).parameters["dye"].default is None
    assert inspect.signature(I.read_idat_dir).parameters["dye"].default is None


def test_dye_bias_runs_and_records_itself_when_asked(rng):
    sig, _ = _signal(rng)
    sig = I.dye_bias(I.poobah(sig), method="linear")
    assert "dye_bias:linear" in sig.steps


def test_dye_bias_needs_enough_type_i_probes(rng):
    sig, _ = _signal(rng, n_ig=10, n_ir=10)
    with pytest.raises(DataError, match="too few type I"):
        I.dye_bias(I.poobah(sig))


def test_the_summary_says_what_ran_and_how_much_was_undetected(rng):
    sig, _ = _signal(rng)
    s = I.noob(I.poobah(sig)).summary()
    assert s["platform"] == "EPICv1"
    assert s["n_type_i"] == 300 and s["n_type_ii"] == 400
    assert "poobah" in s["steps"] and "noob" in s["steps"]
    assert 0.0 <= s["frac_undetected"] <= 1.0


def test_no_pairs_found_says_what_a_pair_is(tmp_path):
    (tmp_path / "lonely_Grn.idat").write_bytes(b"IDAT")
    with pytest.raises(DataError, match="A lone Grn file is not a sample"):
        I.read_idat_dir(tmp_path)
