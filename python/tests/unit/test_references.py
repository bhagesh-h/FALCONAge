"""Reference values for absent features: where they come from and when they apply.

A CpG the data does not carry used to be filled with the mean of the clock's
other CpGs, about 0.5 for most clocks. These tests pin the replacement: the
authors' own per-CpG values where they publish them, a healthy adult blood mean
for the other blood clocks, and a named, weighted warning for what is left.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

import falconage as fa
from falconage.core.tissue import BLOOD, family, normalise
from falconage.models.linear import align
from falconage.registry.registry import DATA_DIR

PUBLISHED = ("horvath2013", "dunedinpoam38", "corticalclock", "altumage")


def _with_reference(registry):
    return [c for c in registry if c.reference_values is not None]


def test_every_reference_file_matches_its_digest(registry):
    clocks = _with_reference(registry)
    assert len(clocks) == 27
    for c in clocks:
        p = DATA_DIR / c.reference_values.file
        assert hashlib.sha256(p.read_bytes()).hexdigest() == c.reference_values.sha256, c.id


def test_a_published_reference_covers_every_feature(registry):
    for cid in PUBLISHED:
        ref = registry.reference_values(cid)
        assert set(ref) == set(registry.feature_ids(cid)), cid


@pytest.mark.parametrize("cid, cpg, value", [
    # Horvath 2013, Additional file 22, goldstandard2, as printed there.
    ("horvath2013", "cg00075967", 0.790221397),
    # DunedinPoAm38 sysdata.rda, mPOA_Models$model_means, 15 significant digits.
    ("dunedinpoam38", "cg02582848", 0.880072541775306),
    # CorticalClock Ref_DNAm_brain_values.rdat.
    ("corticalclock", "cg00059225", 0.216116176865395),
    # AltumAge scaler.pkl, RobustScaler.center_.
    ("altumage", "cg00000292", 0.7598633952352156),
])
def test_published_values_are_the_authors(registry, cid, cpg, value):
    assert registry.reference_values(cid)[cpg] == value


def test_the_blood_reference_serves_only_blood_clocks(registry):
    """Adult blood means in a placenta or buccal clock would replace one wrong
    constant with another; those clocks have no reference instead."""
    for c in _with_reference(registry):
        if not c.reference_values.file.endswith("blood_adult.csv.gz"):
            continue
        assert BLOOD in {family(normalise(t)) for t in c.tissue}, c.id
        ref = registry.reference_values(c.id)
        feats = registry.feature_ids(c.id)
        assert sum(f in ref for f in feats) / len(feats) >= 0.95, c.id
        assert all(0.0 < v < 1.0 for v in ref.values())


def _frame(values: dict[str, list[float]]):
    X = pd.DataFrame(values, index=[f"s{i}" for i in range(len(next(iter(values.values()))))])
    return fa.FalconData(X=X, obs=pd.DataFrame(index=X.index), modality="dna_methylation")


def test_the_reference_fills_only_absent_features():
    """A present feature's missing value takes its own cohort mean, as
    PoAmProjector and CorticalClock.r do; only an absent one takes the
    reference."""
    d = _frame({"b": [0.2, np.nan, 0.4], "c": [0.6, 0.7, 0.8]})
    al = align(d, ["a", "b", "c"], reference={"a": 0.9, "b": 0.1},
               coefficients=np.array([1.0, 1.0, 1.0]))
    assert np.allclose(al.matrix[:, 0], 0.9)
    assert al.matrix[1, 1] == pytest.approx(0.3)
    assert (al.n_from_reference, al.n_pooled) == (1, 0)
    assert al.pooled_mass == 0.0


def test_an_absent_feature_the_reference_lacks_is_pooled_and_weighed():
    d = _frame({"b": [0.2, 0.3, 0.4], "c": [0.6, 0.7, 0.8]})
    al = align(d, ["a", "b", "c"], reference={"b": 0.1},
               coefficients=np.array([2.0, 1.0, 1.0]))
    assert np.allclose(al.matrix[:, 0], 0.5)          # mean of b and c
    assert (al.n_from_reference, al.n_pooled) == (0, 1)
    assert al.pooled_mass == pytest.approx(0.5)


def test_mean_imputation_skips_the_reference():
    d = _frame({"b": [0.2, 0.3, 0.4], "c": [0.6, 0.7, 0.8]})
    al = align(d, ["a", "b", "c"], imputation="mean", reference={"a": 0.9})
    assert np.allclose(al.matrix[:, 0], 0.5)
    assert al.n_pooled == 1


def _drop(synthetic_betas, cols, tissue="whole blood"):
    obs = synthetic_betas.obs.copy()
    obs["tissue"] = tissue
    return fa.FalconData(X=synthetic_betas.X.drop(columns=list(cols)), obs=obs,
                         modality="dna_methylation")


def test_dunedinpoam38_fills_as_the_authors_projector_does(registry, synthetic_betas):
    """PoAmProjector() substitutes the Dunedin training mean for a probe the data
    lacks; the score is the intercept plus the weighted sum over that matrix."""
    feats, w = registry.coefficients("dunedinpoam38")
    gone = feats[:5]
    d = _drop(synthetic_betas, gone)
    got = fa.score(d, clocks=["dunedinpoam38"], min_coverage=0.0).scores["dunedinpoam38"]
    ref = registry.reference_values("dunedinpoam38")
    X = d.X.reindex(columns=feats).copy()
    for f in gone:
        X[f] = ref[f]
    want = -0.06929805 + X.to_numpy() @ w
    assert np.allclose(got.to_numpy(), want, atol=1e-12)


def test_the_run_names_the_reference_it_used(synthetic_betas):
    d = _drop(synthetic_betas, ["cg09809672"])     # a Hannum CpG
    res = fa.score(d, clocks=["hannum"], min_coverage=0.0)
    cov = res.manifest.coverage["hannum"]
    assert cov["n_from_reference"] == 1 and cov["n_pooled"] == 0
    assert cov["reference"] == "healthy adult whole blood (FALCONAge corpus)"
    assert not [w for w in res.manifest.warnings if w.get("category") == "imputation"]


def test_the_fallback_is_named_and_weighed(registry, synthetic_betas):
    """A clock with no published values and no blood counterpart keeps the
    pooled fill, and the run says so with the weight at stake."""
    assert registry.get("pedbe").reference_values is None
    feats, _ = registry.coefficients("pedbe")
    d = _drop(synthetic_betas, feats[:2], tissue="buccal epithelium")
    res = fa.score(d, clocks=["pedbe"], min_coverage=0.0)
    msg = [w["message"] for w in res.manifest.warnings
           if w.get("category") == "imputation" and w.get("clock") == "pedbe"]
    assert msg and "2 absent feature(s)" in msg[0] and "|coefficient|" in msg[0]
    assert "none is published" in msg[0]
