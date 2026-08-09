"""Proteomic and transcriptomic preparation, neural clocks, and scAge.

Four things that were architecture in the design notes and are code now. None
ships weights or coefficients -- every published model in these families is
licence-restricted -- so the tests exercise the machinery against synthetic
inputs with known answers, which is what a scaffold can honestly be checked
against.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import falconage as fa
from falconage.core.errors import AnalysisError, DataError, ScoringError
from falconage.models import single_cell as SC
from falconage.preprocess import proteomic as P
from falconage.preprocess import transcriptomic as T


# ---------------------------------------------------------------------------
# proteomics
# ---------------------------------------------------------------------------

def test_olink_reads_npx_and_leaves_the_scale_alone(tmp_path, rng):
    p = tmp_path / "npx.csv"
    df = pd.DataFrame(rng.normal(4.0, 1.5, size=(6, 5)),
                      index=[f"S{i}" for i in range(6)],
                      columns=[f"P{i}" for i in range(5)])
    df.to_csv(p)
    d = P.read_olink(p)
    assert d.modality == "proteomics" and d.platform == "olink"
    assert np.allclose(d.X.to_numpy(), df.to_numpy())
    assert set(d.units.values()) == {"NPX"}


def test_an_rfu_table_handed_to_the_olink_reader_is_caught(tmp_path, rng):
    """The two platforms report different quantities on different scales, and
    a coefficient fitted on one cannot be applied to the other."""
    p = tmp_path / "rfu.csv"
    pd.DataFrame(rng.uniform(500, 50_000, size=(4, 3)),
                 index=list("abcd"), columns=list("xyz")).to_csv(p)
    with pytest.raises(DataError, match="use read_somascan"):
        P.read_olink(p)


def test_somascan_is_logged_by_default_and_says_so(tmp_path, rng):
    p = tmp_path / "rfu.csv"
    df = pd.DataFrame(rng.uniform(500, 50_000, size=(5, 4)),
                      index=[f"S{i}" for i in range(5)], columns=list("wxyz"))
    df.to_csv(p)
    d = P.read_somascan(p)
    assert np.allclose(d.X.to_numpy(), np.log2(df.to_numpy()))
    assert d.uns["log2"] is True
    raw = P.read_somascan(p, log2=False)
    assert np.allclose(raw.X.to_numpy(), df.to_numpy())


def test_standardising_needs_the_training_cohorts_statistics(tmp_path, rng):
    """The trap every proteomic organ clock sets. The z must be against the
    training cohort, which travels with the model."""
    X = pd.DataFrame(rng.normal(4, 1, size=(8, 4)),
                     index=[f"S{i}" for i in range(8)], columns=list("abcd"))
    d = fa.FalconData(X=X, obs=pd.DataFrame(index=X.index), modality="proteomics")
    with pytest.raises(DataError, match="travel with the model"):
        P.prepare_proteomic(d)

    ref = pd.DataFrame({"mean": [4.0] * 4, "sd": [1.0] * 4}, index=list("abcd"))
    out = P.prepare_proteomic(d, reference=ref)
    assert np.allclose(out.X.to_numpy(), (X.to_numpy() - 4.0) / 1.0)
    assert out.uns["proteomic_standardisation"]["standardise"] == "reference"


def test_cohort_standardisation_is_available_and_flagged(rng):
    X = pd.DataFrame(rng.normal(4, 1, size=(8, 4)),
                     index=[f"S{i}" for i in range(8)], columns=list("abcd"))
    d = fa.FalconData(X=X, obs=pd.DataFrame(index=X.index), modality="proteomics")
    out = P.prepare_proteomic(d, standardise="cohort")
    assert "warning" in out.uns["proteomic_standardisation"]
    assert np.allclose(out.X.mean(axis=0).to_numpy(), 0.0, atol=1e-12)


def test_cohort_standardisation_refuses_one_sample():
    X = pd.DataFrame([[1.0, 2.0, 3.0]], index=["S0"], columns=list("abc"))
    d = fa.FalconData(X=X, obs=pd.DataFrame(index=X.index), modality="proteomics")
    with pytest.raises(DataError, match="intercept for anybody"):
        P.prepare_proteomic(d, standardise="cohort")


# ---------------------------------------------------------------------------
# transcriptomics
# ---------------------------------------------------------------------------

def _counts(rng, n=8, g=200):
    X = pd.DataFrame(rng.integers(1, 5_000, size=(n, g)).astype(float),
                     index=[f"S{i}" for i in range(n)],
                     columns=[f"G{i}" for i in range(g)])
    return fa.FalconData(X=X, obs=pd.DataFrame(index=X.index),
                         modality="transcriptomics")


def test_rle_removes_a_pure_depth_difference(rng):
    """A sample sequenced twice as deep should not look twice as expressed."""
    d = _counts(rng)
    X = d.X.copy()
    X.iloc[0] = X.iloc[0] * 3.0
    out = T.rle_normalise(X)
    ratio = out.iloc[0].median() / out.iloc[1:].median().median()
    assert 0.8 < ratio < 1.25, ratio


def test_rle_refuses_a_matrix_with_no_stable_reference(rng):
    X = pd.DataFrame(np.zeros((5, 20)), index=[f"S{i}" for i in range(5)],
                     columns=[f"G{i}" for i in range(20)])
    with pytest.raises(DataError, match="no stable reference"):
        T.rle_normalise(X)


def test_yugene_is_invariant_to_rescaling_a_sample(rng):
    """The property that makes it a platform correction: rank-based, so any
    monotone rescaling of one sample leaves it unchanged."""
    X = pd.DataFrame(rng.uniform(1, 1_000, size=(4, 60)),
                     index=list("abcd"), columns=[f"G{i}" for i in range(60)])
    a = T.yugene(X)
    scaled = X.copy()
    scaled.iloc[2] = scaled.iloc[2] * 17.0
    b = T.yugene(scaled)
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_median_centring_refuses_one_sample():
    """The whole reason transcriptomic clocks carry requires_cohort."""
    X = pd.DataFrame([[1.0, 2.0, 3.0]], index=["S0"], columns=list("abc"))
    with pytest.raises(DataError, match="centres it against itself"):
        T.median_centre(X)


def test_the_chain_runs_in_order_and_records_it(rng):
    d = _counts(rng)
    out = T.prepare_transcriptomic(d)
    steps = out.uns["transcriptomic_pipeline"]["steps"]
    assert steps == ["rle", "log10(x+1)", "z_per_sample", "yugene", "median_centre"]
    assert out.uns["transcriptomic_pipeline"]["centred_over"] == "this call's samples"


def test_a_many_to_one_orthologue_map_is_refused(rng):
    """Summing paralogues into one column loses which one a coefficient
    referred to, and nothing downstream can recover it."""
    d = _counts(rng, g=10)
    bad = {f"G{i}": "TARGET" for i in range(4)}
    with pytest.raises(DataError, match="not one-to-one"):
        T.prepare_transcriptomic(d, orthologues=bad)


def test_absent_genes_are_padded_not_dropped(rng):
    """So coverage reporting sees them missing instead of never knowing they
    were expected."""
    d = _counts(rng, g=50)
    want = [f"G{i}" for i in range(40)] + ["NOT_MEASURED_1", "NOT_MEASURED_2"]
    out = T.prepare_transcriptomic(d, genes=want, centre=False)
    assert list(out.X.columns) == want
    assert out.X["NOT_MEASURED_1"].isna().all()


def test_counts_the_wrong_way_round_are_transposed_on_the_way_in(tmp_path, rng):
    p = tmp_path / "counts.tsv"
    genes = [f"G{i}" for i in range(30)]
    samples = [f"S{i}" for i in range(6)]
    pd.DataFrame(rng.integers(0, 999, size=(30, 6)), index=genes,
                 columns=samples).to_csv(p, sep="\t")
    d = T.read_counts(p)
    assert list(d.X.index) == samples and list(d.X.columns) == genes


def test_negative_values_are_not_counts(tmp_path):
    p = tmp_path / "counts.tsv"
    pd.DataFrame([[-1, 2], [3, 4]], index=["G1", "G2"],
                 columns=["S1", "S2"]).to_csv(p, sep="\t")
    with pytest.raises(DataError, match="cannot be negative"):
        T.read_counts(p)


# ---------------------------------------------------------------------------
# neural clocks
# ---------------------------------------------------------------------------

def test_a_pickle_is_refused_by_extension(tmp_path):
    """torch.load executes arbitrary code while unpickling, and weights arrive
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

    got, _ = model.predict(d, resolve("cpu", None, requires_fp64=False),
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
