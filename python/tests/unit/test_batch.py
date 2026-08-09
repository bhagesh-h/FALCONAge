"""Frozen-reference batch correction.

The test that matters is the third one: correcting batch 1 alone and correcting
batches 1 and 2 together must give batch 1 *bit-identical* values. Everything
else is scaffolding around demonstrating that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import falconage as fa
from falconage.preprocess.batch import BatchError, BatchReference


def _cohort(rng, n_per=20, n_feat=400, batches=("A", "B", "C"), shift=0.10,
            scale=1.4, seed_offset=0.0):
    """Batches with a real additive and multiplicative offset on top of signal."""
    rows, obs = [], []
    for i, b in enumerate(batches):
        age = rng.uniform(25, 80, n_per)
        base = 0.5 + seed_offset + 0.002 * (age[:, None] - 50)
        x = base + rng.normal(0, 0.03, (n_per, n_feat))
        x = x + shift * i                                    # additive batch effect
        x = (x - x.mean()) * (scale ** i) + x.mean()         # multiplicative
        rows.append(np.clip(x, 0.001, 0.999))
        obs.append(pd.DataFrame({
            "age": age, "plate": b,
            "sex": ["M" if j % 2 else "F" for j in range(n_per)],
        }, index=[f"{b}{j:03d}" for j in range(n_per)]))
    obs = pd.concat(obs)
    X = pd.DataFrame(np.vstack(rows), index=obs.index,
                     columns=[f"cg{i:08d}" for i in range(n_feat)])
    return fa.FalconData(X=X, obs=obs, modality="dna_methylation", platform="450K")


def _subset(data, batches):
    keep = data.obs["plate"].isin(batches)
    return fa.FalconData(X=data.X[keep], obs=data.obs[keep],
                         modality=data.modality, platform=data.platform)


# ---------------------------------------------------------------------------
# it corrects
# ---------------------------------------------------------------------------

def test_it_removes_the_batch_offset(rng):
    d = _cohort(rng)
    ref = fa.fit_batch_reference(d, batch_col="plate", covariates=["age", "sex"])
    out = fa.apply_batch_reference(d, ref, batch_col="plate")

    def spread(x):
        m = x.X.groupby(x.obs["plate"].to_numpy()).mean()
        return float(m.max().max() - m.min().min())

    assert spread(out) < spread(d) / 3, "batch means should collapse together"


def test_it_preserves_the_covariate_it_was_told_to_keep(rng):
    """Correcting out the biology along with the artefact is the failure mode
    that produces a clean-looking null."""
    d = _cohort(rng)
    ref = fa.fit_batch_reference(d, batch_col="plate", covariates=["age", "sex"])
    out = fa.apply_batch_reference(d, ref, batch_col="plate")
    age = d.obs["age"].to_numpy()
    before = np.corrcoef(d.X.mean(axis=1), age)[0, 1]
    after = np.corrcoef(out.X.mean(axis=1), age)[0, 1]
    assert abs(after) > abs(before) * 0.8


# ---------------------------------------------------------------------------
# the point of the whole exercise
# ---------------------------------------------------------------------------

def test_adding_a_batch_does_not_move_an_earlier_one(rng):
    """iComBat's claim, reproduced: zero change, not small change.

    The published measurement of what standard ComBat does instead is a mean
    shift of 0.077-0.39 years in already-reported epigenetic ages, with a
    maximum of 2.20 (PMC12495439).
    """
    d = _cohort(rng, batches=("A", "B", "C"))
    ref = fa.fit_batch_reference(_subset(d, ["A"]), batch_col="plate",
                                 covariates=["age", "sex"])

    first = fa.apply_batch_reference(_subset(d, ["A"]), ref, batch_col="plate")
    later = fa.apply_batch_reference(d, ref, batch_col="plate")

    a = first.X
    b = later.X.loc[a.index, a.columns]
    assert np.array_equal(a.to_numpy(), b.to_numpy()), (
        "batch A must be byte-identical whether or not B and C were run")


def test_a_refit_does_move_it(rng):
    """The failure that motivates the feature has to be demonstrable, or nobody
    believes the fix is needed."""
    d = _cohort(rng, batches=("A", "B", "C"))
    only_a = _subset(d, ["A"])

    ref_a = fa.fit_batch_reference(only_a, batch_col="plate", covariates=["age", "sex"])
    ref_all = fa.fit_batch_reference(d, batch_col="plate", covariates=["age", "sex"])

    from_a = fa.apply_batch_reference(only_a, ref_a, batch_col="plate").X
    from_all = fa.apply_batch_reference(only_a, ref_all, batch_col="plate").X
    delta = np.abs(from_a.to_numpy() - from_all.to_numpy()).max()
    assert delta > 1e-6, "re-fitting on more batches shifts the old values"


def test_the_reference_round_trips_through_a_file(rng, tmp_path):
    d = _cohort(rng)
    ref = fa.fit_batch_reference(d, batch_col="plate", covariates=["age", "sex"])
    p = ref.write(tmp_path / "ref.npz")
    back = BatchReference.read(p)
    assert back.digest == ref.digest
    a = fa.apply_batch_reference(d, ref, batch_col="plate").X.to_numpy()
    b = fa.apply_batch_reference(d, back, batch_col="plate").X.to_numpy()
    assert np.array_equal(a, b)


def test_the_corrected_data_records_which_reference_touched_it(rng):
    d = _cohort(rng)
    ref = fa.fit_batch_reference(d, batch_col="plate")
    out = fa.apply_batch_reference(d, ref, batch_col="plate")
    rec = out.uns["batch_reference"]
    assert rec["digest"] == ref.digest
    assert rec["features_corrected"] == d.n_features


def test_the_digest_changes_when_the_parameters_do(rng):
    d = _cohort(rng)
    a = fa.fit_batch_reference(d, batch_col="plate")
    b = fa.fit_batch_reference(_cohort(rng, shift=0.3), batch_col="plate")
    assert a.digest != b.digest


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------

def test_a_confounded_design_is_refused(rng):
    """Batch and condition perfectly nested: correcting removes the effect and
    returns a clean null, which is the one failure that looks like success."""
    d = _cohort(rng)
    obs = d.obs.copy()
    obs["condition"] = obs["plate"].map({"A": "HC", "B": "CASE", "C": "CASE"})
    nested = fa.FalconData(X=d.X, obs=obs, modality=d.modality, platform=d.platform)
    with pytest.raises(BatchError, match="nested inside"):
        fa.fit_batch_reference(nested, batch_col="plate")
    # and it can be overridden deliberately
    fa.fit_batch_reference(nested, batch_col="plate", protect=())


def test_a_tiny_batch_is_refused(rng):
    d = _cohort(rng, n_per=4)
    with pytest.raises(BatchError, match="fewer than"):
        fa.fit_batch_reference(d, batch_col="plate")


def test_a_reference_from_another_feature_space_is_refused(rng):
    d = _cohort(rng)
    ref = fa.fit_batch_reference(d, batch_col="plate")
    other = _cohort(rng)
    other.X.columns = [f"ch{i:08d}" for i in range(other.n_features)]
    with pytest.raises(BatchError, match="share no features"):
        fa.apply_batch_reference(other, ref, batch_col="plate")


def test_a_changed_covariate_design_is_refused(rng):
    """A factor level in one and not the other silently redefines the columns
    the frozen coefficients belong to."""
    d = _cohort(rng)
    ref = fa.fit_batch_reference(d, batch_col="plate", covariates=["sex"])
    obs = d.obs.copy()
    obs.loc[obs.index[:5], "sex"] = "U"
    changed = fa.FalconData(X=d.X, obs=obs, modality=d.modality, platform=d.platform)
    with pytest.raises(BatchError, match="covariate columns differ"):
        fa.apply_batch_reference(changed, ref, batch_col="plate")


def test_a_missing_batch_label_is_refused(rng):
    d = _cohort(rng)
    obs = d.obs.copy()
    obs.loc[obs.index[0], "plate"] = np.nan
    bad = fa.FalconData(X=d.X, obs=obs, modality=d.modality, platform=d.platform)
    with pytest.raises(BatchError, match="missing values"):
        fa.fit_batch_reference(bad, batch_col="plate")


# ---------------------------------------------------------------------------
# and the reason anyone cares: the scores
# ---------------------------------------------------------------------------

def test_scores_are_stable_across_a_new_plate(registry, rng):
    """End to end, in the units a reader sees: years."""
    feats = list(registry.feature_ids("hannum"))
    n_per = 20
    rows, obs = [], []
    for i, b in enumerate("AB"):
        age = rng.uniform(25, 80, n_per)
        x = 0.5 + 0.002 * (age[:, None] - 50) + rng.normal(0, 0.03, (n_per, len(feats)))
        rows.append(np.clip(x + 0.08 * i, 0.001, 0.999))
        obs.append(pd.DataFrame({"age": age, "plate": b, "tissue": "whole blood"},
                                index=[f"{b}{j:03d}" for j in range(n_per)]))
    obs = pd.concat(obs)
    d = fa.FalconData(X=pd.DataFrame(np.vstack(rows), index=obs.index, columns=feats),
                      obs=obs, modality="dna_methylation", platform="450K")

    ref = fa.fit_batch_reference(_subset(d, ["A"]), batch_col="plate",
                                 covariates=["age"])
    a_only = fa.apply_batch_reference(_subset(d, ["A"]), ref, batch_col="plate")
    both = fa.apply_batch_reference(d, ref, batch_col="plate")

    s1 = fa.score(a_only, clocks=["hannum"], min_coverage=0.0).scores["hannum"]
    s2 = fa.score(both, clocks=["hannum"], min_coverage=0.0).scores["hannum"]
    assert np.array_equal(s1.to_numpy(), s2.loc[s1.index].to_numpy()), (
        "a reported age must not change because someone else's plate was run")
