"""Targeted panels and the one specimen that is always a refusal."""

from __future__ import annotations

import pytest

import falconage as fa
from falconage.core.errors import DataError, ScoringError


# ---------------------------------------------------------------------------
# targeted panels
# ---------------------------------------------------------------------------

def test_a_panel_in_percent_is_read_as_a_fraction(tmp_path):
    p = tmp_path / "panel.csv"
    p.write_text("sample,cg00000001,cg00000002\nS1,75.0,12.5\nS2,60.0,20.0\n",
                 encoding="utf-8")
    d = fa.read_panel(p)
    assert d.X.loc["S1", "cg00000001"] == pytest.approx(0.75)
    assert d.uns["panel_size"] == 2


def test_a_declared_panel_is_checked_not_intersected(tmp_path):
    """A typo in a probe name would otherwise become a missing probe, and a
    missing probe becomes an imputed one."""
    p = tmp_path / "panel.csv"
    p.write_text("sample,cg00000001\nS1,0.75\n", encoding="utf-8")
    with pytest.raises(DataError, match="are absent"):
        fa.read_panel(p, cpgs=["cg00000001", "cg99999999"])


def test_a_panel_with_no_declared_cpg_present_says_what_it_found(tmp_path):
    p = tmp_path / "panel.csv"
    p.write_text("sample,foo,bar\nS1,0.75,0.2\n", encoding="utf-8")
    with pytest.raises(DataError, match="none of the"):
        fa.read_panel(p, cpgs=["cg00000001"])


def test_a_panel_still_fails_the_coverage_floor(registry, tmp_path):
    """The reader exists to reach the refusal with an accurate feature list, not
    to get around it: six of Horvath's 353 probes is not a Horvath score."""
    feats = list(registry.feature_ids("horvath2013"))[:6]
    p = tmp_path / "panel.csv"
    p.write_text("sample," + ",".join(feats) + "\nS1," + ",".join(["0.5"] * 6) + "\n",
                 encoding="utf-8")
    d = fa.read_panel(p, cpgs=feats)
    with pytest.raises(Exception, match="coverage|floor"):
        fa.score(d, clocks=["horvath2013"])


# ---------------------------------------------------------------------------
# cell-free DNA
# ---------------------------------------------------------------------------

def test_cfdna_is_refused_by_every_clock_whatever_its_policy(synthetic_betas):
    """Not a tissue -- a fragment population shed from many of them, on which
    array clocks are published to perform poorly (bioRxiv 2025.11.27.690895).
    horvath2013 is tissue_policy 'allow' and is still refused."""
    obs = synthetic_betas.obs.copy()
    obs["tissue"] = "plasma cell-free DNA"
    d = fa.FalconData(X=synthetic_betas.X, obs=obs, modality="dna_methylation",
                      platform="450K")
    with pytest.raises(ScoringError, match="category error"):
        fa.score(d, clocks=["horvath2013"], min_coverage=0.0)


def test_cfdna_is_skipped_not_raised_under_compatible(synthetic_betas):
    obs = synthetic_betas.obs.copy()
    obs["tissue"] = "cfDNA"
    d = fa.FalconData(X=synthetic_betas.X, obs=obs, modality="dna_methylation",
                      platform="450K")
    with pytest.raises(ScoringError, match="every requested clock was skipped"):
        fa.score(d, clocks="compatible")
