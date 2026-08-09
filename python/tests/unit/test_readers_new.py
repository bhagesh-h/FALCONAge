"""Nanopore, targeted panels, and the one specimen that is always a refusal."""

from __future__ import annotations

import pandas as pd
import pytest

import falconage as fa
from falconage.core.errors import DataError, ScoringError


# ---------------------------------------------------------------------------
# bedMethyl
# ---------------------------------------------------------------------------

def _bedmethyl(tmp_path, rows, name="s1.bed"):
    """Write a minimal modkit-style bedMethyl file.

    BED9 then modkit's columns; only 4 (code), 10 (valid coverage) and 11
    (percent modified) are read, and they are taken by position because the
    format has no header and callers differ past column 11.
    """
    lines = []
    for chrom, pos, code, cov, pct in rows:
        lines.append("\t".join(map(str, [
            chrom, pos, pos + 1, code, 100, "+", pos, pos + 1, "255,0,0",
            cov, f"{pct:.2f}", 0, 0, 0, 0, 0, 0, 0])))
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def test_bedmethyl_reads_percentages_as_fractions(tmp_path):
    p = _bedmethyl(tmp_path, [("chr1", 100, "m", 30, 75.0),
                              ("chr1", 200, "m", 40, 12.5)])
    s = fa.read_bedmethyl(p)
    assert list(s.index) == ["chr1:100", "chr1:200"]
    assert s.iloc[0] == pytest.approx(0.75)
    assert s.iloc[1] == pytest.approx(0.125)


def test_low_coverage_sites_are_dropped_not_kept_with_a_wide_error(tmp_path):
    """A fraction from three reads takes four values. Keeping it and widening
    the interval later pretends it is a measurement on the same scale."""
    p = _bedmethyl(tmp_path, [("chr1", 100, "m", 3, 66.7),
                              ("chr1", 200, "m", 40, 12.5)])
    s = fa.read_bedmethyl(p, min_coverage=10)
    assert list(s.index) == ["chr1:200"]


def test_5hmc_is_not_added_to_5mc(tmp_path):
    """Different measurements. Summing them gives a quantity no array reported."""
    p = _bedmethyl(tmp_path, [("chr1", 100, "m", 30, 60.0),
                              ("chr1", 100, "h", 30, 10.0)])
    m = fa.read_bedmethyl(p, mod_code="m")
    h = fa.read_bedmethyl(p, mod_code="h")
    assert m.iloc[0] == pytest.approx(0.60)
    assert h.iloc[0] == pytest.approx(0.10)


def test_a_missing_modification_code_says_what_the_file_has(tmp_path):
    p = _bedmethyl(tmp_path, [("chr1", 100, "h", 30, 10.0)])
    with pytest.raises(DataError, match="carries"):
        fa.read_bedmethyl(p, mod_code="m")


def test_a_plain_bed_is_refused(tmp_path):
    p = tmp_path / "plain.bed"
    p.write_text("chr1\t100\t101\tm\t100\t+\t100\t101\t255,0,0\n", encoding="utf-8")
    with pytest.raises(DataError, match="at least 11"):
        fa.read_bedmethyl(p)


def test_a_manifest_maps_coordinates_onto_probe_ids(tmp_path):
    """Without one the values are coordinate-keyed: fine for a clock trained on
    sequencing, useless for one trained on an array, and said out loud."""
    p = _bedmethyl(tmp_path, [("chr1", 100, "m", 30, 75.0),
                              ("chr1", 200, "m", 40, 12.5)])
    man = pd.DataFrame({"chrom": ["chr1", "chr1"], "pos": [100, 200],
                        "feature_id": ["cg00000001", "cg00000002"]})
    s = fa.read_bedmethyl(p, manifest=man)
    assert list(s.index) == ["cg00000001", "cg00000002"]


def test_several_files_become_one_matrix_with_its_coverage(tmp_path):
    a = _bedmethyl(tmp_path, [("chr1", 100, "m", 30, 75.0),
                              ("chr1", 200, "m", 40, 12.5)], "a.bed")
    b = _bedmethyl(tmp_path, [("chr1", 100, "m", 55, 71.0),
                              ("chr1", 200, "m", 60, 15.0)], "b.bed")
    d = fa.read_bedmethyl_dir([a, b])
    assert d.X.shape == (2, 2)
    assert d.platform == "nanopore"
    cov = d.uns["site_coverage"]
    assert cov.loc["a", "chr1:100"] == 30 and cov.loc["b", "chr1:100"] == 55


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
