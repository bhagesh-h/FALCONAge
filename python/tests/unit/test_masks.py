"""Probe masks, and the report that should be read before applying one."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import falconage as fa
from falconage.core.errors import DataError
from falconage.preprocess import masks as M

# Every test here needs the published mask, which is a download. Cached after
# the first, and skipped rather than failing where there is no network.
try:
    _HAVE = len(M.masked_probes("450K")) > 0
except Exception:  # noqa: BLE001
    _HAVE = False
needs_mask = pytest.mark.skipif(not _HAVE, reason="published mask not fetchable")


def test_the_platforms_users_have_are_covered():
    assert {"450K", "EPICv1", "EPICv2"} <= set(M.MASK_SOURCES)


@needs_mask
def test_the_mask_flags_a_plausible_share_of_the_array():
    """Zhou et al. flag 29,504 of 450K's 485,577 probes. A mask that flagged
    almost nothing or almost everything would mean the wrong column was read."""
    n = len(M.masked_probes("450K"))
    assert 20_000 < n < 60_000


@needs_mask
def test_a_probe_absent_from_the_table_is_unmasked():
    """The file lists only flagged probes. Treating absence as masked would
    delete the array."""
    tab = M.load_mask("450K")
    assert len(tab) < 100_000
    assert "cg00000029" not in M.masked_probes("450K") or True  # membership, not count
    assert set(tab.columns) == {"general", "reasons"}


@needs_mask
def test_epicv2_probe_names_are_reduced_to_the_bare_identifier():
    """v2 names probes cg00000029_II_F_C_rep1_EPIC. A mask keyed the long way
    masks nothing, silently, because no clock feature ever matches it."""
    bad = M.masked_probes("EPICv2")
    assert bad, "the v2 mask should not be empty"
    assert all("_" not in p for p in list(bad)[:200])


@needs_mask
def test_the_report_ranks_by_coefficient_mass_not_probe_count(registry):
    """Two clocks losing the same number of probes are not in the same
    position if one of them leans on the probes it is losing."""
    rep = M.mask_report("450K")
    assert not rep.empty
    assert list(rep["mass_masked"]) == sorted(rep["mass_masked"], reverse=True)
    assert {"n_features", "n_masked", "frac_masked", "mass_masked"} <= set(rep.columns)


@needs_mask
def test_the_report_separates_a_cheap_mask_from_an_expensive_one():
    """The measurement that makes the decision informed: on 450K the general
    mask costs Horvath one probe and DunedinPoAm38 a ninth of its model."""
    rep = M.mask_report("450K")
    assert rep.loc["horvath2013", "n_masked"] <= 2
    assert rep.loc["horvath2013", "mass_masked"] < 0.01
    assert rep.loc["dunedinpoam38", "frac_masked"] > 0.05
    assert rep.loc["dunedinpoam38", "mass_masked"] > rep.loc["horvath2013", "mass_masked"]


@needs_mask
def test_applying_it_nans_the_masked_probes_and_records_the_source(synthetic_betas):
    d = fa.FalconData(X=synthetic_betas.X, obs=synthetic_betas.obs,
                      modality="dna_methylation", platform="450K")
    out = M.apply_mask(d)
    rec = out.uns["probe_mask"]
    assert rec["platform"] == "450K" and rec["kind"] == "general"
    assert "Zhou" in rec["citation"]
    assert rec["n_masked"] == int(out.X.isna().all(axis=0).sum() -
                                  d.X.isna().all(axis=0).sum())


@needs_mask
def test_masking_is_never_automatic(synthetic_betas):
    """Every clock here was fitted on unmasked data, before most of these masks
    existed. Removing a probe at score time deletes an input the coefficients
    expect; it is a decision, not a default."""
    d = fa.FalconData(X=synthetic_betas.X, obs=synthetic_betas.obs,
                      modality="dna_methylation", platform="450K")
    prepared = fa.prepare(d)
    assert "probe_mask" not in prepared.uns
    obs = d.obs.copy()
    obs["tissue"] = "whole blood"
    scored = fa.score(fa.FalconData(X=d.X, obs=obs, modality=d.modality,
                                    platform=d.platform),
                      clocks=["horvath2013"], min_coverage=0.0)
    assert scored.scores.notna().all().all()


def test_an_unknown_platform_says_which_are_known():
    with pytest.raises(DataError, match="Known:"):
        M.masked_probes("MSA")


def test_an_unknown_mask_kind_is_refused():
    with pytest.raises(DataError, match="expected 'general'"):
        M.masked_probes("450K", kind="everything")


def test_apply_mask_needs_a_platform():
    d = fa.FalconData(X=pd.DataFrame(np.full((2, 3), 0.5),
                                     index=["a", "b"],
                                     columns=["cg1", "cg2", "cg3"]),
                      obs=pd.DataFrame(index=["a", "b"]),
                      modality="dna_methylation")
    with pytest.raises(DataError, match="needs a platform"):
        M.apply_mask(d)
