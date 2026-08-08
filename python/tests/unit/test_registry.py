"""The registry is a data artefact, so most of these are assertions about data."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from falconage.core.errors import ClockNotFoundError, RegistryError, WeightsUnavailableError
from falconage.registry.registry import DATA_DIR, LEGAL_OPS


def test_registry_size_and_tiers(registry):
    assert len(registry) == 161
    tiers = {t: len(registry.filter(availability=t)) for t in "ABC"}
    assert sum(tiers.values()) == 161
    assert tiers["C"] == 28, "the 28 scaffold-only clocks in README section 6"


def test_every_entry_is_well_formed(registry):
    for c in registry:
        assert c.id and c.name
        assert c.scale_type in LEGAL_OPS, f"{c.id}: unknown scale {c.scale_type}"
        assert c.availability in "ABC"
        assert c.data_type in ("dna_methylation", "clinical_chemistry")
        assert c.generation in ("first", "second", "pace", "causal", "mitotic",
                                "system", "other")


def test_tier_a_coefficients_load_and_match_their_digest(registry):
    """Every bundled file is present, parseable, and byte-for-byte what the
    registry says it is. A coefficient set that drifted from its recorded digest
    is the one failure that would change results silently."""
    checked = 0
    for c in registry.filter(availability="A"):
        if c.formula:
            continue
        feats, coefs = registry.coefficients(c.id)
        assert len(feats) == len(coefs) == c.n_features
        assert len(set(feats)) == len(feats), f"{c.id}: duplicate feature ids"
        assert np.isfinite(coefs).all(), f"{c.id}: non-finite coefficient"

        path = DATA_DIR / c.coefficient_source.file
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == c.coefficient_source.sha256, f"{c.id}: digest drift"
        checked += 1
    assert checked == 20, "20 clocks ship coefficient files in v1.0"


def test_known_feature_counts(registry):
    """Spot checks against the published papers, not against another package."""
    assert registry.get("horvath2013").n_features == 353     # Horvath 2013
    assert registry.get("dnamphenoage").n_features == 513    # Levine 2018 Table S6
    assert registry.get("hannum").n_features == 71           # Hannum 2013
    assert registry.get("zhangmortality").n_features == 10   # Zhang 2017


def test_horvath_came_from_the_paper_not_a_package(registry):
    """The two primary-source extractions are flagged as such, and the rest are
    honest about not being."""
    assert registry.get("horvath2013").coefficient_source.primary_source_traced
    assert registry.get("dnamphenoage").coefficient_source.primary_source_traced
    assert not registry.get("hannum").coefficient_source.primary_source_traced
    assert len(registry.untraced()) > 100


def test_scaffold_clocks_refuse_and_say_why(registry):
    for cid in ("grimage2", "dunedinpace", "systemsage", "pcgrimage"):
        c = registry.get(cid)
        assert c.availability == "C"
        assert c.coefficient_source.redistributable is False
        with pytest.raises(WeightsUnavailableError) as exc:
            registry.coefficients(cid)
        msg = str(exc.value)
        assert "scaffold-only" in msg
        assert "Obtain them from" in msg


def test_scaffold_error_names_an_open_alternative(registry):
    msg = registry.unavailable_message("grimage2")
    assert "Open alternatives" in msg
    named = [a for a in ("dnamphenoage", "zhangmortality", "hrsinchphenoage")
             if a in msg]
    assert named, "a user who wanted a mortality clock should leave with one"
    for a in named:
        assert registry.has_coefficients(a), f"{a} is offered but does not work"


def test_unknown_clock_suggests_near_matches(registry):
    with pytest.raises(ClockNotFoundError) as exc:
        registry.get("horvath")
    assert "horvath2013" in str(exc.value)


def test_legal_operations_by_scale(registry):
    assert "acceleration" in registry.get("horvath2013").legal_operations
    assert "acceleration" not in registry.get("dunedinpoam38").legal_operations
    assert "acceleration" not in registry.get("zhangmortality").legal_operations


def test_search_and_filter(registry):
    assert any(c.id == "dnamphenoage" for c in registry.search("mortality"))
    first = registry.filter(generation="first", availability="A")
    assert {c.id for c in first} >= {"horvath2013", "hannum"}


def test_register_local_weights_validates(registry, tmp_path):
    """Registering a file is also how you check one somebody handed you."""
    good = tmp_path / "g.csv"
    feats = [f"cg{i:08d}" for i in range(registry.get("grimage2").n_features or 1032)]
    good.write_text("feature_id,coefficient\n"
                    + "\n".join(f"{f},0.001" for f in feats) + "\n")
    digest = registry.register_local_weights("grimage2", good)
    assert len(digest) == 64
    assert registry.has_coefficients("grimage2")
    assert registry.weight_record("grimage2")["source"] == "user_supplied"

    bad = tmp_path / "b.csv"
    bad.write_text("feature_id,coefficient\ncg1,0.5\ncg1,0.7\n")
    with pytest.raises(RegistryError, match="duplicate"):
        registry.register_local_weights("dunedinpace", bad)


def test_weight_record_distinguishes_bundled_from_supplied(registry):
    rec = registry.weight_record("horvath2013")
    assert rec["source"] == "bundled"
    assert len(rec["sha256"]) == 64


# ---------------------------------------------------------------------------
# paper-vs-implementation disagreements
# ---------------------------------------------------------------------------

# The eleven cases docs/science.qmd records. Pinned by name rather than by
# count alone: a count assertion passes just as happily when one entry is lost
# and another gained, and the whole value of this field is that a specific
# clock carries a specific caveat to the person scoring with it.
DOCUMENTED_DISCREPANCIES = {
    "bocklandt", "bohlin", "cvdwesterman", "zhangmortality",
    "yingcausage", "yingdamage", "yingadaptage",
    "senchronoage", "sencultureage", "senmortalityage",
    "phenoage",
}


def test_every_documented_discrepancy_reaches_the_registry(registry):
    """Prose in the docs warns nobody. This field is what a user actually sees."""
    carried = {c.id for c in registry if c.known_discrepancies}
    assert carried == DOCUMENTED_DISCREPANCIES


def test_discrepancy_text_is_useful(registry):
    """A bare integer difference is not actionable; say what it means."""
    for cid in DOCUMENTED_DISCREPANCIES:
        for d in registry.get(cid).known_discrepancies:
            assert len(d) > 60, f"{cid}: discrepancy note is too terse"
            assert d.rstrip().endswith("."), f"{cid}: not a sentence"


def test_a_discrepancy_becomes_a_warning_at_score_time(synthetic_betas):
    """The path that matters: registry field -> run manifest -> user."""
    import falconage as fa

    res = fa.score(synthetic_betas, clocks=["yingcausage"], min_coverage=0.0)
    notes = [w for w in res.manifest.warnings if w.get("category") == "discrepancy"]
    assert notes and notes[0]["clock"] == "yingcausage"
    assert "586" in notes[0]["message"]


def test_every_op_named_in_the_registry_is_dispatchable(registry):
    """A typo in a chain is invisible until the clock is scored.

    Most of these clocks cannot be scored at all yet -- no coefficients -- so
    nothing would exercise the chain until the day someone registers a weight
    file, at which point the failure looks like a coefficient problem. Check
    the names now.
    """
    from falconage.models import ops

    for c in registry:
        for step in c.preprocess:
            assert step.get("op") in ops.PREPROCESS, f"{c.id}: preprocess {step}"
        for step in c.postprocess:
            assert step.get("op") in ops.POSTPROCESS, f"{c.id}: postprocess {step}"


def test_a_clock_reported_in_days_does_not_claim_to_be_in_weeks(registry):
    """The reason the chains above were worth wiring before the coefficients.

    An empty chain is not neutral -- it means identity. Bohlin and EPICGA are
    trained on gestational age in days and declared on a weeks scale, so with
    no transform they would return roughly seven times the right number and
    nothing would say so.
    """
    from falconage.models import ops

    for cid in ("bohlin", "epicga"):
        assert ops.describe_chain(registry.get(cid).postprocess) == "days_to_weeks()"
