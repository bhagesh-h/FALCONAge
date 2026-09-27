"""consensus() on repeated measures: paired visits and a random-intercept model.

The fixture turns the 24 synthetic samples into 12 people seen twice: each
person's second score is their first plus a little noise, so people differ from
one another far more than they change, which is the situation a paired or mixed
design exists for.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

import falconage as fa
from falconage.core.errors import AnalysisError


def _blood(synthetic_betas):
    obs = synthetic_betas.obs.copy()
    obs["tissue"] = "whole blood"
    return fa.FalconData(X=synthetic_betas.X, obs=obs, modality="dna_methylation",
                         platform=synthetic_betas.platform)


@pytest.fixture(scope="module")
def visits(synthetic_betas):
    res = fa.score(_blood(synthetic_betas), clocks="compatible")
    n = res.scores.shape[0] // 2
    rng = np.random.default_rng(7)
    out = copy.copy(res)
    scores = res.scores.copy()
    first = scores.iloc[:n].to_numpy()
    sd = np.nanstd(first, axis=0, ddof=1)
    scores.iloc[n:] = first + rng.normal(0, 0.05, size=first.shape) * sd
    out.scores = scores
    obs = res.obs.copy()
    obs["person"] = [f"P{i % n:02d}" for i in range(2 * n)]
    obs["visit"] = ["v1"] * n + ["v2"] * n
    age = obs["age"].to_numpy(float)
    obs["age"] = np.r_[age[:n], age[:n] + 0.25]
    out.obs = obs
    return out


def _gens(res):
    g: dict[str, list[str]] = {}
    for cid in res.scores.columns:
        g.setdefault(res.registry.get(cid).generation, []).append(cid)
    return g


def _move(res, clocks, amount_sd):
    out = copy.copy(res)
    out.scores = res.scores.copy()
    later = (res.obs["visit"] == "v2").to_numpy()
    for cid in clocks:
        sd = float(res.scores.loc[~later, cid].std(ddof=1))
        out.scores.loc[later, cid] += amount_sd * sd
    return out


def test_a_within_person_change_needs_the_paired_design(visits):
    gens = _gens(visits)
    move = gens["first"][:2] + gens["second"][:2]
    res = _move(visits, move, 0.3)
    paired = fa.consensus(res, "visit", reference="v1", design="paired", subject_col="person")
    assert paired.verdict == "supported", paired.why
    assert all(paired.table.loc[c, "sig_bonferroni"] for c in move)
    independent = fa.consensus(res, "visit", reference="v1")
    assert not any(independent.table.loc[c, "sig_bonferroni"] for c in move)


def test_complete_pairs_give_the_same_test_either_way(visits):
    res = _move(visits, _gens(visits)["first"][:2], 0.3)
    p = fa.consensus(res, "visit", reference="v1", design="paired", subject_col="person").table
    m = fa.consensus(res, "visit", reference="v1", design="mixed", subject_col="person").table
    common = p.index.intersection(m.index)
    assert len(common) > 5
    assert np.allclose(p.loc[common, "t"], m.loc[common, "t"], rtol=1e-5)
    assert np.allclose(p.loc[common, "p"], m.loc[common, "p"], rtol=1e-4)
    assert (p.loc[common, "df"] == m.loc[common, "df"]).all()


def test_a_missed_visit_still_counts_in_the_mixed_model(visits):
    res = copy.copy(visits)
    keep = ~((visits.obs["person"] == "P03") & (visits.obs["visit"] == "v2")).to_numpy()
    res.scores, res.obs = visits.scores[keep], visits.obs[keep]
    m = fa.consensus(res, "visit", reference="v1", design="mixed", subject_col="person").table
    p = fa.consensus(res, "visit", reference="v1", design="paired", subject_col="person").table
    assert (m["n_subjects"] == 12).all() and (p["n_subjects"] == 11).all()


def test_three_visits_are_tested_jointly(visits):
    res = copy.copy(visits)
    obs = visits.obs.copy()
    obs.loc[obs.index[::3], "visit"] = "v3"
    res.obs = obs
    rep = fa.consensus(res, "visit", reference="v1", design="mixed", subject_col="person")
    assert {"F", "delta_v2", "delta_v3"} <= set(rep.table.columns)
    with pytest.raises(AnalysisError, match="exactly two"):
        fa.consensus(res, "visit", reference="v1", design="paired", subject_col="person")


def test_a_significant_change_smaller_than_one_person_can_show_is_flagged(visits):
    """Twelve people make a mean change of 1.6 SD of the change significant
    (t = 1.6 sqrt(12) = 5.5), while one person's change needs 1.96 SD to be told
    from noise."""
    gens = _gens(visits)
    move = gens["first"][:2] + gens["second"][:2]
    base = fa.consensus(visits, "visit", reference="v1", design="paired",
                        subject_col="person").table
    res = copy.copy(visits)
    res.scores = visits.scores.copy()
    later = (visits.obs["visit"] == "v2").to_numpy()
    for cid in move:
        sd_diff = base.loc[cid, "mdc95"] / 1.96
        res.scores.loc[later, cid] += 1.6 * sd_diff - base.loc[cid, "delta"]
    rep = fa.consensus(res, "visit", reference="v1", design="paired", subject_col="person")
    for cid in move:
        assert rep.table.loc[cid, "sig_bonferroni"] and rep.table.loc[cid, "below_mdc"]
    assert "smaller than the minimum detectable change" in rep.why


def test_the_repeated_designs_need_the_person(visits):
    with pytest.raises(AnalysisError, match="subject_col"):
        fa.consensus(visits, "visit", design="paired")
    dup = copy.copy(visits)
    obs = visits.obs.copy()
    obs.loc[obs.index[0], "person"] = obs.loc[obs.index[1], "person"]
    dup.obs = obs
    with pytest.raises(AnalysisError, match="more than one sample"):
        fa.consensus(dup, "visit", design="paired", subject_col="person")
    with pytest.raises(AnalysisError, match="design must be"):
        fa.consensus(visits, "visit", design="crossover", subject_col="person")
