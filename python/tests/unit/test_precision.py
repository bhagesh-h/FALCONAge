"""Precision is chosen once, recorded per clock, and never silently lowered."""

from __future__ import annotations

import warnings

import pytest

from falconage.core.backend import resolve
from falconage.core.errors import ConfigError
from falconage.core.manifest import RunManifest


def test_default_precision_is_float64():
    assert resolve().dtype == "float64"
    assert resolve().device == "cpu"


def test_float32_is_honoured_unless_the_clock_requires_float64():
    assert resolve("float32").dtype == "float32"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        spec = resolve("float32", requires_fp64=True)
    assert spec.dtype == "float64"
    assert any("requires_fp64" in str(x.message) for x in w)


def test_unknown_precision_is_refused():
    with pytest.raises(ConfigError, match="unknown dtype"):
        resolve("float16")


def test_manifest_records_each_clocks_precision_and_summarises_a_mixed_run():
    m = RunManifest()
    m.record_compute("a", resolve("float32"))
    assert m.dtype == "float32" and m.compute_summary() == "float32"
    m.record_compute("b", resolve())
    m.record_compute("c", resolve())
    assert m.dtype == "mixed"
    assert m.compute_summary() == "float64 (2 clocks), float32 (1 clock)"


def test_scored_result_records_precision_for_every_clock(synthetic_betas):
    import falconage as fa

    res = fa.score(synthetic_betas, clocks="compatible")
    assert set(res.manifest.compute) == set(res.scores.columns)
    assert all(c == {"dtype": "float64"} for c in res.manifest.compute.values())
