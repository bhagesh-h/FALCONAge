"""The registry is a data artefact, so most of these are assertions about data."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

import falconage as fa
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
    assert checked == 22, ("22 clocks ship coefficient files: the 20 linear and PC "
                           "sets, plus the epiTOC1 and HypoClock probe lists")


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


# ---------------------------------------------------------------------------
# cohort-centred clocks
# ---------------------------------------------------------------------------

def test_requires_cohort_defaults_off(registry):
    """Every clock shipping today is per-sample; the flag must not change them."""
    assert not any(c.requires_cohort for c in registry)
    assert all(c.min_samples == 1 for c in registry)


def _registry_with_cohort_clock(cid: str, min_samples: int):
    """A private registry instance with one clock marked cohort-centred.

    `from_yaml()` and NOT `load()`. `load()` is `lru_cache`d and returns the
    same object to every caller for the lifetime of the process, so mutating
    what it hands back edits the registry every other test is using. Doing that
    here broke 23 tests in files that never mention this flag, and each of them
    failed with a message about horvath2013 needing 999 samples -- which is a
    long way from pointing at the test that caused it.
    """
    import dataclasses

    from falconage.registry.registry import ClockRegistry

    reg = ClockRegistry.from_yaml()
    reg._clocks[cid] = dataclasses.replace(
        reg.get(cid), requires_cohort=True, min_samples=min_samples)
    return reg


def test_a_cohort_clock_refuses_a_single_sample(synthetic_betas):
    """The failure this flag exists to prevent.

    Centring one row against itself makes every feature zero, so the model
    returns its intercept -- the same confident number for any input. Nothing
    in the arithmetic can notice, which is why it is a declared property.
    """
    import falconage as fa
    from falconage.core.errors import ScoringError

    reg = _registry_with_cohort_clock("horvath2013", 8)

    one = synthetic_betas.subset(samples=[synthetic_betas.sample_ids[0]])
    with pytest.raises(ScoringError, match="centres each feature"):
        fa.score(one, clocks=["horvath2013"], registry=reg)

    # And the whole cohort is fine.
    res = fa.score(synthetic_betas, clocks=["horvath2013"], registry=reg)
    assert res.scores.shape[0] == synthetic_betas.n_samples


def test_a_cohort_clock_is_skipped_not_raised_when_not_asked_for(synthetic_betas):
    """`clocks="compatible"` must not blow up a whole run over one clock."""
    import falconage as fa

    reg = _registry_with_cohort_clock("horvath2013", 999)

    res = fa.score(synthetic_betas, clocks="compatible", registry=reg)
    assert "horvath2013" in res.skipped
    assert "undefined for" in res.skipped["horvath2013"]
    assert res.scores.shape[1] > 0, "the other clocks should still have run"


def test_the_shared_registry_was_not_mutated(registry):
    """Guard for the mistake above: `load()` is cached and shared."""
    assert not registry.get("horvath2013").requires_cohort
    assert registry.get("horvath2013").min_samples == 1


# ---------------------------------------------------------------------------
# the relative-origin age scale
# ---------------------------------------------------------------------------
# A clock can be in years, track age with a slope of one, and still have no
# fixed zero. Ying's DamAge and AdaptAge are the case: measured across three
# healthy cohorts their median offset against chronological age swings 162
# years each while Horvath's swings 15, they move as near mirror images
# (r = -0.975), and the swing in their sum is only 33 -- one dataset-level
# shift amplified by intercepts of +543.43 and -511.97, not two broken clocks.

RELATIVE_ORIGIN = {"yingdamage", "yingadaptage"}


def test_the_relative_origin_scale_is_exactly_the_two_measured_clocks(registry):
    got = {c.id for c in registry if c.scale_type == "age_years_relative"}
    assert got == RELATIVE_ORIGIN


def test_it_forbids_the_absolute_convention_and_allows_the_residual(registry):
    """The whole distinction. `predicted - chronological` needs the zero to
    mean something; a residual fitted inside the dataset does not."""
    for cid in RELATIVE_ORIGIN:
        ops = registry.get(cid).legal_operations
        assert "acceleration" not in ops
        assert {"residual", "difference", "mean", "correlate"} <= ops


def test_causage_keeps_the_ordinary_age_scale(registry):
    """Only the two that were measured to drift. CausAge's offset swings 15
    years, the same as Horvath's, so nothing about it changed."""
    assert registry.get("yingcausage").scale_type == "age_years"
    assert "acceleration" in registry.get("yingcausage").legal_operations


def test_it_is_not_relative_score(registry):
    """The other error, and the tempting one. `relative_score` admits only
    correlate and rank, which would forbid the residual and the group
    difference -- the operations the paper itself reports -- and would
    contradict a unit of years and a training target of chronological age,
    both of which are accurate."""
    for cid in RELATIVE_ORIGIN:
        c = registry.get(cid)
        assert c.unit == ("years",)
        assert c.training_target == ("chronological age",)
        assert "residual" in c.legal_operations


def test_the_reason_travels_with_the_clock(registry):
    """A scale change nobody can trace is the same problem as a silent offset."""
    for cid in RELATIVE_ORIGIN:
        notes = " ".join(registry.get(cid).known_discrepancies)
        assert "162 years" in notes and "-0.975" in notes


def test_absolute_acceleration_is_refused_with_the_alternative_named(synthetic_betas):
    from falconage.core.errors import IllegalOperationError

    obs = synthetic_betas.obs.copy()
    obs["tissue"] = "whole blood"
    d = fa.FalconData(X=synthetic_betas.X, obs=obs, modality="dna_methylation",
                      platform="450K")
    res = fa.score(d, clocks=["yingdamage"], min_coverage=0.0)

    with pytest.raises(IllegalOperationError, match="origin is not fixed"):
        fa.acceleration(res, method="absolute", clocks=["yingdamage"])
    # "both" returns an absolute column, so it is refused for the same reason.
    with pytest.raises(IllegalOperationError):
        fa.acceleration(res, method="both", clocks=["yingdamage"])

    # And the residual, which is what the published analyses use, works.
    acc = fa.acceleration(res, method="residual", clocks=["yingdamage"])
    assert acc["yingdamage"].notna().all()
    assert abs(float(acc["yingdamage"].mean())) < 1e-8, "a residual is centred"


def test_it_is_dropped_from_an_unnamed_absolute_run_rather_than_raising(synthetic_betas):
    """`clocks=None` means 'the ones this makes sense for'. Under the absolute
    convention that no longer includes these two, and the rest still run."""
    obs = synthetic_betas.obs.copy()
    obs["tissue"] = "whole blood"
    d = fa.FalconData(X=synthetic_betas.X, obs=obs, modality="dna_methylation",
                      platform="450K")
    res = fa.score(d, clocks=["yingdamage", "horvath2013"], min_coverage=0.0)

    absolute = fa.acceleration(res, method="absolute")
    assert list(absolute.columns) == ["horvath2013"]

    residual = fa.acceleration(res, method="residual")
    assert set(residual.columns) == {"yingdamage", "horvath2013"}


def test_a_conformal_interval_is_not_offered_for_them(synthetic_betas):
    """A prediction band against chronological age has no meaning for a clock
    whose origin moves between cohorts, so the calibration excludes them."""
    cal = fa.uncertainty.load_conformal()
    if cal.empty:
        pytest.skip("conformal.csv absent")
    assert not (set(cal["clock"]) & RELATIVE_ORIGIN)
