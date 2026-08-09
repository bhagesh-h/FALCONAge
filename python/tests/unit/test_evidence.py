"""The evidence pack: what a score has been shown to predict.

The tests here are mostly about the table's discipline rather than its
arithmetic. Its whole value is that a reader can check every row, so the rules
that keep it checkable are what needs guarding.
"""

from __future__ import annotations


import falconage as fa


def test_every_row_carries_a_doi():
    """An uncheckable effect size looks identical to a checkable one in a table,
    and is worth less than nothing. ``evidence()`` raises rather than returning
    a row without a source."""
    df = fa.registry.evidence()
    assert not df.empty
    assert (df["doi"].str.len() > 0).all()
    assert (df["citation"].str.len() > 0).all()


def test_every_row_carries_the_design_it_came_from():
    """A hazard ratio without its cohort and follow-up is a number, not a
    finding."""
    df = fa.registry.evidence()
    assert (df["design"].str.len() > 20).all()


def test_the_clocks_named_are_real_registry_entries(registry):
    df = fa.registry.evidence()
    named = set(df["clock"]) - {"*"}
    unknown = {c for c in named if c not in {x.id for x in registry}}
    assert not unknown, f"evidence.yaml names clocks that do not exist: {unknown}"


def test_the_cross_clock_finding_behind_legal_ops_is_recorded():
    """R = 0.12 between age accuracy and mortality prediction is the empirical
    argument for refusing acceleration on a mortality scale, and it should be
    quotable from the package rather than only from the docs."""
    df = fa.registry.evidence()
    g = df[df["clock"] == "*"]
    row = g[g["outcome"].str.contains("mortality prediction")]
    assert len(row) == 1
    assert row.iloc[0]["value"] == 0.12


def test_grimage2_carries_its_hazard_ratios():
    df = fa.registry.evidence("grimage2")
    hr = df[df["measure"] == "hazard_ratio"]
    assert set(hr[hr["outcome"] == "cirrhosis"]["value"]) == {1.86}
    # Two mortality rows, from two studies with different designs and different
    # numbers -- 1.54 per SD in Generation Scotland, 2.57 pooled across the
    # Biolearn cohorts. Both are kept. Collapsing them to one would be picking a
    # favourite, and the design column is there precisely so a reader can see
    # why they differ.
    mort = hr[hr["outcome"] == "all-cause mortality"]
    assert sorted(mort["value"]) == [1.54, 2.57]
    assert mort["doi"].nunique() == 2


def test_an_unknown_clock_returns_an_empty_table_not_an_error():
    assert fa.registry.evidence("no_such_clock").empty


def test_interpretation_carries_a_one_line_digest(synthetic_betas):
    obs = synthetic_betas.obs.copy()
    obs["tissue"] = "whole blood"
    d = fa.FalconData(X=synthetic_betas.X, obs=obs, modality="dna_methylation",
                      platform="450K")
    res = fa.score(d, clocks=["horvath2013", "hannum"], min_coverage=0.0)
    col = res.interpretation()["published_associations"]
    assert col.str.len().max() > 20
    assert all(len(v) < 200 for v in col), "a digest, not the whole table"


def test_the_result_can_hand_back_the_full_rows(synthetic_betas):
    obs = synthetic_betas.obs.copy()
    obs["tissue"] = "whole blood"
    d = fa.FalconData(X=synthetic_betas.X, obs=obs, modality="dna_methylation",
                      platform="450K")
    res = fa.score(d, clocks=["horvath2013", "hannum"], min_coverage=0.0)
    ev = res.evidence()
    assert set(ev["clock"]) <= {"horvath2013", "hannum"}
    assert (ev["doi"].str.len() > 0).all()
