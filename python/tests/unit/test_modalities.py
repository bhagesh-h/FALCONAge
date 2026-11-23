"""Neural clocks and scAge.

Neither family ships weights here, so the tests exercise the machinery against
synthetic inputs with known answers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import falconage as fa
from falconage.core.errors import AnalysisError, DataError, ScoringError
from falconage.models import single_cell as SC


# ---------------------------------------------------------------------------
# neural clocks
# ---------------------------------------------------------------------------

def test_a_pickle_is_refused_by_extension(tmp_path):
    """Unpickling a file executes arbitrary code, and weights arrive
    by download from a third party."""
    p = tmp_path / "weights.pt"
    p.write_bytes(b"not really a pickle")
    with pytest.raises(ScoringError, match="threat model"):
        fa.models.read_neural_weights(p, features=["a"])


def test_a_network_without_its_feature_order_is_not_a_clock(tmp_path):
    pytest.importorskip("safetensors")
    from safetensors.numpy import save_file

    p = tmp_path / "w.safetensors"
    save_file({"layer0.weight": np.zeros((2, 3), dtype=np.float32),
               "layer0.bias": np.zeros(2, dtype=np.float32)}, str(p))
    with pytest.raises(ScoringError, match="probe order"):
        fa.models.read_neural_weights(p)


def test_a_two_layer_network_computes_the_forward_pass(tmp_path, registry, rng):
    pytest.importorskip("safetensors")
    from safetensors.numpy import save_file

    feats = [f"cg{i:08d}" for i in range(6)]
    w0 = rng.normal(size=(4, 6)).astype(np.float32)
    b0 = rng.normal(size=4).astype(np.float32)
    w1 = rng.normal(size=(1, 4)).astype(np.float32)
    b1 = rng.normal(size=1).astype(np.float32)
    p = tmp_path / "w.safetensors"
    save_file({"layer0.weight": w0, "layer0.bias": b0,
               "layer1.weight": w1, "layer1.bias": b1}, str(p))

    weights = fa.models.read_neural_weights(p, features=feats)
    assert weights.n_parameters == w0.size + b0.size + w1.size + b1.size

    X = pd.DataFrame(rng.uniform(0.1, 0.9, size=(5, 6)),
                     index=[f"S{i}" for i in range(5)], columns=feats)
    d = fa.FalconData(X=X, obs=pd.DataFrame(index=X.index),
                      modality="dna_methylation", platform="450K")
    model = fa.models.NeuralClock(clock=registry.get("horvath2013"), weights=weights)
    from falconage.core.backend import resolve

    got, _ = model.predict(d, resolve(None, requires_fp64=False),
                           min_coverage=0.0)

    h = np.maximum(X.to_numpy() @ w0.T.astype(np.float64) + b0, 0.0)
    raw = h @ w1.T.astype(np.float64) + b1
    want = fa.models.ops.apply_chain(raw.ravel(),
                                     registry.get("horvath2013").postprocess,
                                     fa.models.ops.POSTPROCESS)
    assert np.allclose(got.to_numpy(), np.asarray(want).ravel())


# ---------------------------------------------------------------------------
# scAge
# ---------------------------------------------------------------------------

def _bulk_reference(rng, n=120, g=400):
    """A bulk cohort where a known subset of CpGs tracks age linearly."""
    age = rng.uniform(20, 80, size=n)
    X = rng.uniform(0.35, 0.65, size=(n, g))
    informative = np.arange(0, 150)
    for j in informative:
        slope = rng.choice([-1, 1]) * rng.uniform(0.004, 0.008)
        X[:, j] = np.clip(0.5 + slope * (age - 50) + rng.normal(0, 0.03, n), 0.02, 0.98)
    idx = [f"S{i:03d}" for i in range(n)]
    cols = [f"cg{j:06d}" for j in range(g)]
    d = fa.FalconData(X=pd.DataFrame(X, index=idx, columns=cols),
                      obs=pd.DataFrame({"age": age}, index=idx),
                      modality="dna_methylation", platform="450K")
    return d, cols, informative, age


def test_the_reference_keeps_the_sites_that_track_age(rng):
    d, cols, informative, _ = _bulk_reference(rng)
    ref = SC.fit_scage_reference(d, min_abs_r=0.3)
    kept = {cols.index(c) for c in ref.slope.index}
    assert len(kept) > 50
    # Almost everything kept should come from the block that was built to move.
    assert len(kept & set(informative.tolist())) / len(kept) > 0.9


def test_the_reference_refuses_a_cohort_too_small_to_fit(rng):
    d, *_ = _bulk_reference(rng, n=10)
    with pytest.raises(AnalysisError, match="fitting noise"):
        SC.fit_scage_reference(d)


def test_a_cell_built_from_a_known_age_is_recovered(rng):
    """The end-to-end check: binarise a bulk profile at a known age and see if
    the likelihood puts it back."""
    d, cols, _, _ = _bulk_reference(rng)
    ref = SC.fit_scage_reference(d, min_abs_r=0.3)

    truth = 65.0
    p = pd.Series(ref.probability(truth), index=ref.slope.index)
    calls = (rng.uniform(size=len(p)) < p.to_numpy()).astype(float)
    # A single cell covers a small fraction of sites.
    covered = rng.choice(len(p), size=max(len(p) // 3, 40), replace=False)
    row = np.full(len(p), np.nan)
    row[covered] = calls[covered]
    cells = fa.FalconData(
        X=pd.DataFrame([row], index=["cell1"], columns=p.index),
        obs=pd.DataFrame(index=["cell1"]), modality="dna_methylation")

    out = SC.scage(cells, ref)
    assert out.loc["cell1", "n_sites"] == len(covered)
    assert abs(out.loc["cell1", "age"] - truth) < 15.0, out.loc["cell1"].to_dict()


def test_a_cell_with_too_few_sites_gets_nan_and_a_reason(rng):
    """Twenty binary observations against a straight line is not an age
    estimate, and returning one anyway fills a per-cell table with confident
    noise."""
    d, *_ = _bulk_reference(rng)
    ref = SC.fit_scage_reference(d, min_abs_r=0.3)
    row = np.full(len(ref.slope), np.nan)
    row[:5] = 1.0
    cells = fa.FalconData(
        X=pd.DataFrame([row], index=["c"], columns=ref.slope.index),
        obs=pd.DataFrame(index=["c"]), modality="dna_methylation")
    out = SC.scage(cells, ref)
    assert np.isnan(out.loc["c", "age"])
    assert "informative sites" in out.loc["c", "reason"]


def test_no_shared_sites_says_what_the_mismatch_is(rng):
    d, *_ = _bulk_reference(rng)
    ref = SC.fit_scage_reference(d, min_abs_r=0.3)
    cells = fa.FalconData(
        X=pd.DataFrame([[1.0, 0.0]], index=["c"], columns=["chr1:100", "chr1:200"]),
        obs=pd.DataFrame(index=["c"]), modality="dna_methylation")
    with pytest.raises(DataError, match="share no CpGs"):
        SC.scage(cells, ref)
