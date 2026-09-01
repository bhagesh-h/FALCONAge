"""The overlap table's classifiers, and the claims overlap.csv makes.

The three ``*_class`` columns are the ones people filter on, so a
misclassification here does not produce an error, it produces a shortlist with
the wrong clocks in it. That is the failure worth testing: a wrong row looks
exactly like a right one.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
TOOL = ROOT / "python" / "tools" / "build_overlap.py"

spec = importlib.util.spec_from_file_location("build_overlap", TOOL)
bo = importlib.util.module_from_spec(spec)
sys.modules["build_overlap"] = bo
spec.loader.exec_module(bo)


# ---------------------------------------------------------------------------
# tissue
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tissues,expect", [
    (["whole blood"], "whole_blood"),
    (["whole blood", "saliva"], "whole_blood"),
    (["purified blood leukocytes"], "blood_derived"),
    (["sorted monocytes"], "blood_derived"),
    (["B cells"], "blood_derived"),
    (["multi-tissue"], "multi_tissue"),
    # multi-tissue wins over whole blood: a clock fitted across tissues is not a
    # blood clock, and putting it in the blood bucket is how it ends up on a
    # blood shortlist it does not belong on.
    (["multi-tissue", "whole blood"], "multi_tissue"),
    (["brain cortex"], "brain"),
    (["placenta"], "perinatal_tissue"),
    (["cultured fibroblasts"], "cell_culture"),
    (["prostate"], "other_tissue"),
    ([], "unstated"),
])
def test_tissue_class(tissues, expect):
    assert bo.tissue_class(tissues) == expect


# ---------------------------------------------------------------------------
# population
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pop,expect", [
    ("adults", "adults"),
    ("adult men", "adults"),
    ("human, age unspecified", "adults"),
    ("older adults", "older_adults"),
    ("centenarians", "older_adults"),
    ("all ages", "all_ages"),
    ("children and adolescents", "children"),
    ("pregnancies", "perinatal"),
    ("newborns", "perinatal"),
    ("mice", "non_human"),
    ("multiple mammalian species", "non_human"),
    ("human cell cultures", "cell_culture"),
    ("", "unstated"),
])
def test_population_class(pop, expect):
    assert bo.population_class(pop) == expect


# ---------------------------------------------------------------------------
# training target
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target,expect", [
    (["chronological age"], "chronological_age"),
    (["gestational age"], "chronological_age"),
    (["mortality"], "mortality"),
    (["pace of aging"], "pace_of_aging"),
    (["cell-type proportions"], "cell_composition"),
    (["population doublings"], "mitotic"),
    (["phenotypic age"], "phenotype"),
    (["frailty index"], "phenotype"),
    (["smoking exposure"], "exposure"),
    (["leptin"], "protein_or_analyte"),
    # Written both ways in the registry, and neither spelling is more correct.
    (["GDF-15"], "protein_or_analyte"),
    (["gdf15"], "protein_or_analyte"),
    (["growth differentiation factor 15"], "protein_or_analyte"),
    # disease has to beat everything: it is the only class that says the
    # training cohort was not healthy.
    (["hepatocellular carcinoma"], "disease"),
    (["late-onset Alzheimer's disease"], "disease"),
    (["prostate cancer"], "disease"),
    # A PC clock is refitted to reproduce another clock, not to predict age.
    (["Horvath clock output"], "clock_output"),
    (["DNAm GrimAge output"], "clock_output"),
    (["Y-chromosome presence"], "sex_or_chromosome"),
    (["not applicable"], "unstated"),
    ([], "unstated"),
])
def test_target_class(target, expect):
    assert bo.target_class(target) == expect


def test_a_clock_output_target_is_not_reported_as_an_age_predictor():
    """PC clocks recalibrate an existing clock. Filing them under
    chronological_age would put a dozen derived models on a shortlist of primary
    age predictors, which is the opposite of what the column is for."""
    assert bo.target_class(["Hannum clock output"]) != "chronological_age"


# ---------------------------------------------------------------------------
# name matching against third-party catalogues
# ---------------------------------------------------------------------------

def test_permuted_keys_matches_reordered_compound_names():
    """biolearn files mccartneyalcohol as AlcoholMcCartney."""
    assert bo._norm("mccartneyalcohol") in bo._permuted_keys("AlcoholMcCartney")


def test_the_field_s_best_known_clocks_have_an_alias():
    """Horvath 2013 is Horvath1 everywhere else. Without the alias the five
    most-used clocks in the field reported as being in no external catalogue,
    which was wrong in the direction that flatters our own registry."""
    for cid in ("horvath2013", "skinandblood", "dnamphenoage"):
        assert cid in bo.CATALOGUE_ALIASES


# ---------------------------------------------------------------------------
# the emitted file
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def table():
    path = ROOT / "overlap.csv"
    if not path.exists():
        pytest.skip("overlap.csv absent; run python/tools/build_overlap.py")
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_every_registry_clock_has_a_row(table):
    import falconage as fa

    assert {r["clock_id"] for r in table} == set(fa.registry.load().list())


def test_every_row_carries_a_verification_url(table):
    missing = [r["clock_id"] for r in table if not r["verify_url"].startswith("http")]
    assert not missing, f"rows with no verify_url: {missing[:8]}"


def test_no_verification_url_is_unresolved(table):
    """Checked with --check-urls. A 403 from a publisher that blocks bots is
    recorded as such against Crossref rather than as a dead link, so anything
    still marked unresolved is a genuinely broken reference."""
    bad = [r["clock_id"] for r in table if "unresolved" in r["verify_url_status"]]
    assert not bad, f"unresolved: {bad[:8]}"


def test_feature_overlap_is_blank_where_features_are_not_distributed(table):
    """Blank, not zero. "shares no CpGs" and "we cannot see this clock's CpGs"
    are different facts and a reader sorting on the column must not have them
    collapsed."""
    for r in table:
        if r["ships_coefficients"] == "no":
            assert r["feature_overlap_jaccard"] == ""
            assert r["feature_overlap_note"]


def test_profile_peers_are_symmetric(table):
    """If A lists B as sharing its training profile, B must list A."""
    peers = {r["clock_id"]: {p for p in r["peers_same_profile"].split("; ") if p}
             for r in table}
    for cid, mine in peers.items():
        for other in mine:
            assert cid in peers[other], f"{cid} lists {other} but not the reverse"
