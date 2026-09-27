"""Horvath's gold-standard normalisation, against his own code.

``python/tests/data/horvath_norm_reference.R`` sourced Horvath 2013 Additional
file 24 unmodified (checked by SHA-256) and ran ``BMIQcalibration`` on three
synthetic samples against goldstandard2, and recorded the ``set.seed(1);
sample(...)`` draw the code fits on.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import falconage as fa
from falconage.preprocess.horvath import bmiq_calibration, goldstandard2, r_sample

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def reference():
    X = pd.read_csv(DATA / "horvath_norm_input.csv.gz", index_col=0)
    R = pd.read_csv(DATA / "horvath_norm_output.csv.gz", index_col=0)
    return X, R


def test_the_gold_standard_is_additional_file_22():
    g = goldstandard2()
    assert len(g) == 21368 and g.index.is_unique
    assert g["cg00000292"] == pytest.approx(0.807065832)


def test_the_probe_draw_is_rs():
    """The fits use 20,000 of the 21,368 probes, drawn by R's sample() after
    set.seed(1); the same draw is needed to fit the same mixtures."""
    with gzip.open(DATA / "horvath_norm_sample.txt.gz", "rt") as fh:
        want = np.array([int(v) for v in fh])
    assert np.array_equal(r_sample(21368, 20000, seed=1) + 1, want)


def test_calibration_matches_horvaths_code(reference):
    X, R = reference
    g = goldstandard2()
    assert list(X.columns) == list(g.index)
    got = bmiq_calibration(X.to_numpy(), g.to_numpy())
    # R's output is written to 12 significant figures; the rest is R summing in
    # extended precision and its own dbeta, pbeta and qbeta.
    assert np.max(np.abs(got - R.to_numpy())) < 1e-9


def test_normalise_drops_absent_probes_fills_missing_and_leaves_the_rest(reference):
    X, R = reference
    X = X.iloc[:2].copy()
    X["cg99999999"] = 0.5                              # not a gold-standard probe
    X.iloc[0, 5] = np.nan                              # a missing value
    X = X.drop(columns=X.columns[10:20])               # ten absent probes
    d = fa.FalconData(X=X, obs=pd.DataFrame(index=X.index), modality="dna_methylation")
    for absent in ("drop", "fill"):
        out = fa.preprocess.horvath_normalise(d, absent=absent)
        rec = out.uns["horvath_normalisation"]
        assert rec["n_absent_from_data"] == 10 and rec["absent"] == absent
        assert rec["filled_with_gold_standard"] == {"S1": 1, "S2": 0}
        assert (out.X["cg99999999"] == 0.5).all()
        assert list(out.X.columns) == list(X.columns)
        assert out.X.iloc[:, :5].sub(X.iloc[:, :5]).abs().to_numpy().max() > 1e-3


def test_knights_published_example_output():
    """Knight et al. 2016, Additional file 7: their code on their TestDataset
    (Additional file 6) gives 37.366, 38.346 and 39.324 weeks. They ran R
    3.1.2, whose sample() drew by rounding, and imputed the 51 missing values
    by k-nearest neighbours, which this does not; that is the last 0.002."""
    t = pd.read_csv(DATA / "knight_testdataset.csv.gz", index_col="CpGName").T
    d = fa.FalconData(X=t, obs=pd.DataFrame({"tissue": "cord blood"}, index=t.index),
                      modality="dna_methylation", platform="27K")
    raw = fa.score(d, clocks=["knight"]).scores["knight"]
    got = fa.score(fa.preprocess.horvath_normalise(d, sample_kind="Rounding"),
                   clocks=["knight"]).scores["knight"]
    want = np.array([37.366, 38.346, 39.324])
    assert np.max(np.abs(got.to_numpy() - want)) < 0.0025
    assert np.max(np.abs(raw.to_numpy() - want)) > 0.4, "without it, off by up to 0.7 weeks"
