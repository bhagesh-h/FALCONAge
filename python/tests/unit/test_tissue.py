"""Specimen-type checking.

The failure this exists to prevent is quiet. Scoring a whole-blood clock on
saliva returns a plausible number that correlates with age and is wrong by
3.83-16.46 years (bioRxiv 2025.09.16.673560). Nothing in the arithmetic can
notice, so the check has to come from the declared tissue.
"""

from __future__ import annotations

import dataclasses

import pytest

import falconage as fa
from falconage.core import tissue as T
from falconage.core.errors import ScoringError


# ---------------------------------------------------------------------------
# the lookup table
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("given,expect", [
    ("whole blood", "whole blood"),
    ("Whole_Blood", "whole blood"),
    ("peripheral blood", "whole blood"),
    ("buffy coat", "buffy coat"),
    ("PBMC", "pbmc"),
    ("Peripheral Blood Mononuclear Cells", "pbmc"),
    ("Saliva", "saliva"),
    ("buccal swab", "buccal epithelium"),
    ("cord blood", "cord blood"),
    ("plasma cell-free DNA", "cell-free dna"),
    ("whole blood (fasting)", "whole blood"),
    ("multi-tissue", "multi-tissue"),
])
def test_normalise_maps_what_people_actually_write(given, expect):
    assert T.normalise(given) == expect


@pytest.mark.parametrize("given", ["", None, "NA", "unknown", "-", "widget"])
def test_normalise_refuses_to_guess(given):
    """``None`` means "not recognised", which is a true statement.

    The alternative -- fuzzy matching -- turns an unknown label into a confident
    wrong family, and this check only earns its place if a pass means something.
    """
    assert T.normalise(given) is None


def test_every_tissue_string_in_the_registry_is_recognised(registry):
    """The guard that keeps the table honest.

    A tissue in clocks.yaml the table has never heard of makes the check
    silently not run for that clock -- the worst outcome, because the run looks
    checked. Adding a clock with a new specimen must fail here first.
    """
    unmapped = sorted({t for c in registry for t in c.tissue if T.normalise(t) is None})
    assert not unmapped, f"unmapped tissue strings: {unmapped}"


def test_cell_free_dna_is_not_a_kind_of_blood():
    """Grouping it under blood would make the check pass on the one case it
    exists to stop: array clocks on cfDNA perform poorly
    (bioRxiv 2025.11.27.690895)."""
    assert T.family(T.normalise("plasma cell-free DNA")) == T.CFDNA
    assert T.family("whole blood") == T.BLOOD


# ---------------------------------------------------------------------------
# compare()
# ---------------------------------------------------------------------------

def test_exact_match_says_nothing():
    r = T.compare("whole blood", ("whole blood",))
    assert r["verdict"] == "exact" and r["message"] == ""


def test_a_pan_tissue_clock_has_nothing_to_check():
    r = T.compare("saliva", ("multi-tissue",))
    assert r["verdict"] == "unrestricted"


def test_buffy_coat_on_a_whole_blood_clock_is_a_family_note_with_the_number():
    r = T.compare("buffy coat", ("whole blood",))
    assert r["verdict"] == "family"
    assert "0.97" in r["message"], "the note should carry the measured agreement"


def test_saliva_on_a_blood_clock_is_a_mismatch_carrying_the_measured_error():
    r = T.compare("saliva", ("whole blood",))
    assert r["verdict"] == "mismatch"
    assert "3.83-16.46" in r["message"]


def test_an_unrecognised_specimen_is_reported_as_unchecked_not_as_fine():
    r = T.compare("widget", ("whole blood",))
    assert r["verdict"] == "unrecognised"
    assert "did not run" in r["message"]


# ---------------------------------------------------------------------------
# the registry field
# ---------------------------------------------------------------------------

def test_policy_defaults_are_derived_from_the_tissue_list(registry):
    """No entry should have to state the obvious case."""
    assert registry.get("horvath2013").tissue_policy == "allow", "multi-tissue"
    assert registry.get("hannum").tissue_policy == "warn", "whole blood"
    assert registry.get("pedbe").tissue_policy == "refuse", "buccal, stated in the YAML"


def test_the_refusing_clocks_are_the_ones_with_no_peripheral_counterpart(registry):
    refuse = {c.id for c in registry if c.tissue_policy == "refuse"}
    assert refuse == {
        "bohlin", "corticalclock", "downsyndrome", "epicga", "gliasin", "knight",
        "leecontrol", "leerefinedrobust", "leerobust", "mayne", "neusin", "pedbe",
    }


def test_an_illegal_policy_is_a_load_error(tmp_path):
    from falconage.registry.registry import DATA_DIR, ClockRegistry, RegistryError

    src = (DATA_DIR / "clocks.yaml").read_text(encoding="utf-8")
    bad = src.replace('    tissue_policy: "refuse"', '    tissue_policy: "maybe"', 1)
    p = tmp_path / "clocks.yaml"
    p.write_text(bad, encoding="utf-8")
    with pytest.raises(RegistryError, match="tissue_policy"):
        ClockRegistry.from_yaml(p)


# ---------------------------------------------------------------------------
# enforcement at score time
# ---------------------------------------------------------------------------

def _with_tissue(data, value):
    obs = data.obs.copy()
    obs["tissue"] = value
    return fa.FalconData(X=data.X, obs=obs, modality=data.modality,
                         platform=data.platform)


def _tissue_warnings(res):
    return [w for w in res.manifest.warnings if w.get("category") == "tissue"]


def test_a_run_with_no_tissue_column_says_the_check_did_not_run(synthetic_betas):
    """Silence is the current-state bug. An absent column is not a pass."""
    res = fa.score(synthetic_betas, clocks=["hannum"], min_coverage=0.0)
    w = _tissue_warnings(res)
    assert len(w) == 1, "once for the run, not once per clock"
    assert "no 'tissue' column" in w[0]["message"]


def test_an_exact_match_is_silent(synthetic_betas):
    res = fa.score(_with_tissue(synthetic_betas, "whole blood"),
                   clocks=["hannum"], min_coverage=0.0)
    assert _tissue_warnings(res) == []


def test_saliva_on_a_blood_clock_warns_with_the_number(synthetic_betas):
    res = fa.score(_with_tissue(synthetic_betas, "saliva"),
                   clocks=["hannum"], min_coverage=0.0)
    w = _tissue_warnings(res)
    assert len(w) == 1 and w[0]["clock"] == "hannum"
    assert "3.83-16.46" in w[0]["message"]
    # and the score is still produced -- a warning, not a refusal
    assert res.scores.shape == (synthetic_betas.n_samples, 1)


def test_a_pan_tissue_clock_is_not_warned_about(synthetic_betas):
    res = fa.score(_with_tissue(synthetic_betas, "saliva"),
                   clocks=["horvath2013"], min_coverage=0.0)
    assert _tissue_warnings(res) == [], "horvath2013 is multi-tissue by design"


def test_the_worst_specimen_in_a_mixed_run_is_the_one_reported(synthetic_betas):
    """Reporting the first specimen would hide the saliva behind the blood."""
    obs = synthetic_betas.obs.copy()
    obs["tissue"] = ["whole blood"] * (len(obs) - 3) + ["saliva"] * 3
    mixed = fa.FalconData(X=synthetic_betas.X, obs=obs,
                          modality=synthetic_betas.modality,
                          platform=synthetic_betas.platform)
    w = _tissue_warnings(fa.score(mixed, clocks=["hannum"], min_coverage=0.0))
    assert len(w) == 1 and "3.83-16.46" in w[0]["message"]


def test_a_refusing_clock_raises_when_asked_for_by_name(synthetic_betas):
    with pytest.raises(ScoringError, match="category error"):
        fa.score(_with_tissue(synthetic_betas, "whole blood"),
                 clocks=["pedbe"], min_coverage=0.0)


def test_a_refusing_clock_is_skipped_not_raised_under_compatible(synthetic_betas):
    res = fa.score(_with_tissue(synthetic_betas, "whole blood"), clocks="compatible")
    assert "pedbe" in res.skipped
    assert "tissue_policy=refuse" in res.skipped["pedbe"]
    assert res.scores.shape[1] > 0, "the blood clocks should still have run"


def test_a_refusing_clock_runs_on_its_own_tissue(synthetic_betas):
    res = fa.score(_with_tissue(synthetic_betas, "buccal swab"),
                   clocks=["pedbe"], min_coverage=0.0)
    assert res.scores.shape == (synthetic_betas.n_samples, 1)
    assert _tissue_warnings(res) == []


def test_policy_allow_disables_the_check_entirely(synthetic_betas, fresh_registry):
    reg = fresh_registry
    reg._clocks["hannum"] = dataclasses.replace(reg.get("hannum"), tissue_policy="allow")
    res = fa.score(_with_tissue(synthetic_betas, "saliva"), clocks=["hannum"],
                   min_coverage=0.0, registry=reg)
    assert _tissue_warnings(res) == []


def test_interpretation_says_what_each_clock_was_trained_on(synthetic_betas):
    res = fa.score(_with_tissue(synthetic_betas, "whole blood"),
                   clocks=["hannum", "horvath2013"], min_coverage=0.0)
    interp = res.interpretation()
    assert interp.loc["hannum", "trained_on"] == "whole blood"
    assert interp.loc["horvath2013", "trained_on"] == "multi-tissue"
