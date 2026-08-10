"""Every model either uses the device it is handed or declares that it will not.

WHY THIS FILE EXISTS. Three shipping clocks and two architectures accepted a
``DeviceSpec`` and computed in numpy regardless. Nothing caught it, because
every other test in this suite passes ``resolve("cpu")`` -- against which a
model that ignores the argument entirely is indistinguishable from one that
honours it. The scoring loop then wrote the *requested* device into the run
manifest, so a CUDA run scoring PhenoAge recorded ``device="cuda"`` for
arithmetic that never left the host.

The tests below are about the contract rather than about speed. A model may
legitimately decline a device -- nine clinical markers are not worth a kernel
launch -- but it has to say so, and the manifest has to record what happened
instead of what was asked for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import falconage as fa
from falconage.core.backend import DeviceSpec, resolve
from falconage.core.manifest import RunManifest
from falconage.models import effective_spec


class RecordingSpec:
    """A DeviceSpec that remembers whether the model actually consulted it.

    A proxy rather than a subclass: ``DeviceSpec`` is a frozen dataclass, and
    the point is to observe the three methods a forward pass has to go through
    to reach an array. Anything else delegates untouched, so a model cannot
    pass by reading ``.device`` and hard-coding numpy.
    """

    def __init__(self, inner: DeviceSpec):
        self._inner = inner
        self.touches: list[str] = []

    def xp(self):
        self.touches.append("xp")
        return self._inner.xp()

    def asarray(self, a):
        self.touches.append("asarray")
        return self._inner.asarray(a)

    def tonumpy(self, a):
        self.touches.append("tonumpy")
        return self._inner.tonumpy(a)

    def __getattr__(self, name):
        return getattr(self._inner, name)


# ---------------------------------------------------------------------------
# one instance of every model class that can produce a score
# ---------------------------------------------------------------------------

def _linear(registry, data):
    from falconage.models.linear import LinearClock

    return LinearClock.from_registry(registry, "horvath2013"), data


def _pc(registry, data):
    from falconage.models.pc import PCLinearClock, PCRotation

    rng = np.random.default_rng(20260810)
    feats = list(data.features[:120])
    rot = PCRotation(features=feats,
                     centre=rng.uniform(0.2, 0.8, len(feats)),
                     rotation=rng.normal(0, 0.1, (len(feats), 4)),
                     coefficients=rng.normal(0, 2.0, 4))
    return PCLinearClock(clock=registry.get("pchorvath2013"), rotation=rot), data


def _aggregation(registry, data):
    from falconage.models.aggregation import AggregationClock

    feats = list(data.features[:60])
    return AggregationClock(clock=registry.get("epitoc1"), features=feats,
                            coefficients=np.ones(len(feats)),
                            statistic="mean"), data


def _neural(registry, data):
    from falconage.models.neural import NeuralClock, NeuralWeights

    rng = np.random.default_rng(20260811)
    feats = list(data.features[:30])
    layers = [(rng.normal(0, 0.2, (8, 30)), rng.normal(0, 0.1, 8)),
              (rng.normal(0, 0.2, (1, 8)), rng.normal(0, 0.1, 1))]
    w = NeuralWeights(layers=layers, features=feats)
    return NeuralClock(clock=registry.get("horvath2013"), weights=w), data


def _clinical(registry, clinical_data):
    from falconage.models.clinical import ClinicalClock

    return ClinicalClock(clock=registry.get("phenoage")), clinical_data


#: (name, factory, uses_clinical_fixture). Every model class in the package
#: that returns a score. ScaffoldClock is absent on purpose: it raises by
#: design and has no forward pass to place anywhere.
BUILDERS = [
    ("LinearClock", _linear, False),
    ("PCLinearClock", _pc, False),
    ("AggregationClock", _aggregation, False),
    ("NeuralClock", _neural, False),
    ("ClinicalClock", _clinical, True),
]


@pytest.mark.parametrize("name,factory,clinical", BUILDERS)
def test_a_model_either_uses_the_spec_or_declares_cpu_only(
        name, factory, clinical, registry, synthetic_betas, synthetic_clinical):
    """The contract, asserted in both directions.

    Ignoring the spec is allowed. Ignoring it *silently* is not, because
    :func:`falconage.models.effective_spec` is what tells the manifest which
    device a clock ran on, and it reads the declaration rather than watching
    the arithmetic.
    """
    model, data = factory(registry, synthetic_clinical if clinical else synthetic_betas)
    rec = RecordingSpec(resolve("cpu"))
    model.predict(data, rec, min_coverage=0.0)

    declared_cpu_only = getattr(model, "CPU_ONLY", False)
    if declared_cpu_only:
        assert rec.touches == [], (
            f"{name} declares CPU_ONLY but reached through the spec "
            f"{rec.touches}. Either it now has a device path and the "
            f"declaration is stale, or the declaration is right and something "
            f"is routing arrays where it should not.")
    else:
        assert rec.touches, (
            f"{name} accepts a DeviceSpec and never used it. Route the "
            f"forward pass through spec.xp()/spec.asarray(), or declare "
            f"CPU_ONLY = True with the reason. Accepting a device and "
            f"computing somewhere else makes the manifest's device field a "
            f"fiction.")


def test_clinical_is_the_only_class_that_declines_a_device():
    """Pinned rather than derived. A class opting out of the device is a
    decision with a reason, and adding one should mean editing this line."""
    from falconage import models

    declining = {cls.__name__ for cls in (
        models.LinearClock, models.PCLinearClock, models.AggregationClock,
        models.NeuralClock, models.ClinicalClock)
        if getattr(cls, "CPU_ONLY", False)}
    assert declining == {"ClinicalClock"}


# ---------------------------------------------------------------------------
# effective_spec, and what it does to a spec
# ---------------------------------------------------------------------------

def test_effective_spec_passes_a_device_aware_model_through_unchanged(registry,
                                                                     synthetic_betas):
    model, _ = _linear(registry, synthetic_betas)
    spec = resolve("cpu")
    assert effective_spec(model, spec) is spec


def test_effective_spec_narrows_a_cpu_only_model_to_the_host(registry,
                                                             synthetic_clinical):
    model, _ = _clinical(registry, synthetic_clinical)
    pretend_cuda = DeviceSpec(device="cuda", dtype="float64", backend="torch")
    got = effective_spec(model, pretend_cuda)
    assert (got.device, got.backend) == ("cpu", "numpy")
    # Precision survives. as_cpu() answers "where", not "how precisely", and a
    # clock flagged requires_fp64 must not be quietly downgraded by moving.
    assert got.dtype == "float64"


def test_as_cpu_keeps_float32_and_is_identity_on_a_host_spec():
    f32 = DeviceSpec(device="cuda", dtype="float32", backend="torch")
    assert f32.as_cpu().dtype == "float32"
    host = resolve("cpu")
    assert host.as_cpu() is host


# ---------------------------------------------------------------------------
# what the manifest ends up saying
# ---------------------------------------------------------------------------

def test_manifest_reports_the_device_that_ran_not_the_one_requested():
    """The defect this file exists for, at the level it was visible.

    A run asked for CUDA. Twenty clocks went there; PhenoAge did not. The
    scalar field used to say ``cuda`` for the whole run because it was
    overwritten once per clock and PhenoAge happened not to be last.
    """
    m = RunManifest()
    m.device_requested = "cuda"
    m.record_compute("horvath2013", DeviceSpec("cuda", "float64", "torch"))
    m.record_compute("hannum", DeviceSpec("cuda", "float64", "torch"))
    m.record_compute("phenoage", DeviceSpec("cpu", "float64", "numpy"))

    assert m.device == "mixed" and m.backend == "mixed"
    assert m.dtype == "float64"          # precision did not vary, so not mixed
    assert m.device_requested == "cuda"
    assert m.compute["phenoage"] == {"device": "cpu", "dtype": "float64",
                                     "backend": "numpy"}
    assert m.compute["horvath2013"]["device"] == "cuda"


def test_a_uniform_run_still_reports_a_plain_device():
    """The common case must not become harder to read for the sake of the rare
    one. Nothing about a CPU run should mention mixing."""
    m = RunManifest()
    for cid in ("horvath2013", "hannum", "phenoage"):
        m.record_compute(cid, DeviceSpec("cpu", "float64", "numpy"))
    assert (m.device, m.dtype, m.backend) == ("cpu", "float64", "numpy")
    assert m.compute_summary() == "numpy:cpu/float64"


def test_compute_summary_names_both_halves_of_a_mixed_run():
    m = RunManifest()
    for cid in ("horvath2013", "hannum"):
        m.record_compute(cid, DeviceSpec("cuda", "float64", "torch"))
    m.record_compute("phenoage", DeviceSpec("cpu", "float64", "numpy"))
    s = m.compute_summary()
    assert s == "torch:cuda/float64 (2 clocks), numpy:cpu/float64 (1 clock)"


def test_a_requires_fp64_clock_makes_the_dtype_mixed_not_the_last_one_seen():
    """Precision can differ inside one run for a reason that has nothing to do
    with devices: the registry overrides float32 for the ill-conditioned
    chains. The old scalar reported whichever clock came last."""
    m = RunManifest()
    m.record_compute("hannum", DeviceSpec("cuda", "float32", "torch"))
    m.record_compute("pchorvath2013", DeviceSpec("cuda", "float64", "torch"))
    assert m.dtype == "mixed" and m.device == "cuda"


def test_a_real_run_records_compute_for_every_clock_it_scored(synthetic_betas):
    res = fa.score(synthetic_betas, clocks="compatible", device="cpu")
    assert set(res.manifest.compute) == set(res.scores.columns)
    assert res.manifest.device_requested == "cpu"
    assert all(c["backend"] == "numpy" for c in res.manifest.compute.values())


def test_the_clinical_clocks_record_cpu(synthetic_clinical):
    res = fa.score(synthetic_clinical, clocks=["phenoage"], device="cpu")
    assert res.manifest.compute["phenoage"]["device"] == "cpu"


# ---------------------------------------------------------------------------
# the torch path, where a GPU-less runner can still check the arithmetic
# ---------------------------------------------------------------------------
#
# `FALCONAGE_DEVICE=torch` with an explicit `device="cpu"` resolves to the torch
# backend on the host. That exercises every line the CUDA path takes except the
# transfer, which is the part least likely to be wrong and the only part that
# needs hardware. CI has no torch, so these skip there and run in
# `falconage:1.1.0-cuda`.

def _torch_host_spec(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setenv("FALCONAGE_DEVICE", "torch")
    spec = resolve("cpu")
    if spec.backend != "torch":                       # pragma: no cover
        pytest.skip("torch backend did not resolve")
    return spec


def test_the_neural_forward_pass_agrees_between_numpy_and_torch(
        monkeypatch, registry, synthetic_betas):
    model, data = _neural(registry, synthetic_betas)
    on_numpy, _ = model.predict(data, resolve("cpu"), min_coverage=0.0)
    on_torch, _ = model.predict(data, _torch_host_spec(monkeypatch), min_coverage=0.0)
    # Both are float64 on the same host BLAS, so the tolerance is for summation
    # order, not for precision.
    np.testing.assert_allclose(on_torch.to_numpy(), on_numpy.to_numpy(), rtol=1e-12)


@pytest.mark.parametrize("statistic,q", [("mean", None), ("quantile", 0.95),
                                         ("weighted_mean", None)])
def test_every_aggregation_statistic_agrees_between_numpy_and_torch(
        monkeypatch, registry, statistic, q):
    from falconage.models.aggregation import AggregationClock

    rng = np.random.default_rng(20260812)
    feats = [f"cg{i:08d}" for i in range(80)]
    ids = [f"S{i:03d}" for i in range(10)]
    d = fa.FalconData(X=pd.DataFrame(rng.uniform(0.05, 0.95, (10, 80)),
                                     index=ids, columns=feats),
                      obs=pd.DataFrame(index=ids),
                      modality="dna_methylation", platform="450K")
    cid = {"mean": "epitoc1", "quantile": "stemtoc",
           "weighted_mean": "reedbmi"}[statistic]
    m = AggregationClock(clock=registry.get(cid), features=feats,
                         coefficients=rng.uniform(0.1, 3.0, len(feats)),
                         statistic=statistic, q=q)
    a, _ = m.predict(d, resolve("cpu"), min_coverage=0.0)
    b, _ = m.predict(d, _torch_host_spec(monkeypatch), min_coverage=0.0)
    np.testing.assert_allclose(b.to_numpy(), a.to_numpy(), rtol=1e-12)


def test_a_torch_run_leaves_the_clinical_clocks_in_numpy(monkeypatch,
                                                         synthetic_clinical):
    """The mixed manifest, end to end, without needing a card.

    The linear path reports torch and the clinical path reports numpy, from one
    call, which is exactly the shape a CUDA run has.
    """
    _torch_host_spec(monkeypatch)
    res = fa.score(synthetic_clinical, clocks=["phenoage"], device="cpu")
    assert res.manifest.compute["phenoage"]["backend"] == "numpy"
