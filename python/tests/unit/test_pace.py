"""DunedinPACE's projector, against the authors' code and preprocessCore.

``python/tests/data/pace_reference.R`` wrote every file used here: it ran the
authors' ``PACEProjector`` (danbelsky/DunedinPACE at 4b56998) on a synthetic
model with the layout of ``mPACE_Models`` and data that exercise each branch
(absent background and model probes, partly and mostly missing probes, a
sample with too much missing, ties), and ran
``preprocessCore::normalize.quantiles.use.target`` on its own.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import falconage as fa
from falconage.core.errors import FeatureCoverageError, RegistryError
from falconage.models.pace import quantile_normalize_to_target
from falconage.registry.registry import ClockRegistry

DATA = Path(__file__).resolve().parents[1] / "data"
needs_r = pytest.mark.skipif(shutil.which("Rscript") is None, reason="needs Rscript")


@pytest.mark.parametrize("target, expected, cols", [
    ("qnorm_target50.csv", "qnorm_out50.csv", 6),      # same length, ties, one column with NA
    ("qnorm_target64.csv", "qnorm_out64.csv", 5),      # a target of another length
])
def test_normalisation_is_preprocesscores(target, expected, cols):
    M = pd.read_csv(DATA / "qnorm_input.csv").to_numpy(float)[:, :cols]
    t = pd.read_csv(DATA / target).iloc[:, 0].to_numpy(float)
    want = pd.read_csv(DATA / expected, skipinitialspace=True).to_numpy(float)
    got = quantile_normalize_to_target(M, t)
    assert np.array_equal(np.isnan(got), np.isnan(want))
    assert np.allclose(got[~np.isnan(got)], want[~np.isnan(want)], rtol=0, atol=1e-14)


def _betas(platform="EPICv1"):
    X = pd.read_csv(DATA / "pace_betas.csv", index_col=0).T
    obs = pd.DataFrame({"tissue": "whole blood"}, index=X.index)
    return fa.FalconData(X=X, obs=obs, modality="dna_methylation", platform=platform)


@needs_r
def test_the_import_scores_as_the_authors_projector_does(tmp_path):
    reg = ClockRegistry.from_yaml()           # registration mutates the registry
    reg.import_dunedinpace(DATA / "pace_model.rda", out_dir=tmp_path)
    res = fa.score(_betas(), clocks=["dunedinpace"], registry=reg)
    want = pd.read_csv(DATA / "pace_reference.csv", index_col=0)["score"]
    got = res.scores["dunedinpace"].reindex(want.index)
    assert got.isna().equals(want.isna()), "S09 misses too much and is NA in both"
    assert np.allclose(got.dropna(), want.dropna(), rtol=0, atol=1e-12)
    assert res.manifest.weights["dunedinpace"]["source"] == "user_supplied"


@needs_r
def test_too_little_background_is_refused_by_name(tmp_path):
    reg = ClockRegistry.from_yaml()
    reg.import_dunedinpace(DATA / "pace_model.rda", out_dir=tmp_path)
    d = _betas()
    thin = fa.FalconData(X=d.X.iloc[:, :250], obs=d.obs, modality=d.modality,
                         platform=d.platform)
    with pytest.raises(FeatureCoverageError, match="background probes"):
        fa.score(thin, clocks=["dunedinpace"], registry=reg)


def test_without_the_package_it_is_a_scaffold():
    reg = ClockRegistry.from_yaml()
    assert reg.get("dunedinpace").availability == "licensed"
    assert "import_dunedinpace" in reg.unavailable_message("dunedinpace")
    with pytest.raises(RegistryError, match="no such file"):
        reg.import_dunedinpace("/nonexistent/DunedinPACE")
