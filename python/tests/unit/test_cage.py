"""cAge (Bernabeu et al. 2023): the model, the switch, and the authors' output.

``python/tests/data/cage_reference.R`` ran the authors' ``cage_predictor.R``
(elenabernabeu/cage_bage at 301f414, fetched, not kept) on synthetic betas
whose predictions fall on both sides of the 20-year switch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import falconage as fa
from falconage.core.errors import RegistryError

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def reg():
    return fa.registry.load()


def test_the_two_models_are_tables_s8_and_s9(reg):
    feats, terms, b0, b0_log = reg.quadratic_model("cage")
    assert (b0, b0_log) == (30.7873139, 3.87249881475691)
    # Bernabeu 2023: 2,274 linear and 56 quadratic terms on age, 1,931 and 55 on log(age)
    assert [(terms[c] != 0).sum() for c in ("lin", "lin_sq", "log", "log_sq")] == [2274, 56, 1931, 55]
    assert len(feats) == 3225 == reg.get("cage").n_features
    assert reg.feature_ids("cage") == tuple(feats)
    assert not reg.has_coefficient_vector("cage")
    with pytest.raises(RegistryError, match="squared-CpG terms"):
        reg.coefficients("cage")


def test_scores_as_the_authors_predictor_does(reg):
    X = pd.read_csv(DATA / "cage_betas.csv.gz", index_col=0).T
    want = pd.read_csv(DATA / "cage_reference.csv", index_col=0)["cage"]
    d = fa.FalconData(X=X, obs=pd.DataFrame({"tissue": "whole blood"}, index=X.index),
                      modality="dna_methylation", platform="EPICv1")
    got = fa.score(d, clocks=["cage"]).scores["cage"].reindex(want.index)
    assert (want <= 20).sum() >= 3 and (want > 20).sum() >= 3, "both models exercised"
    # the betas are stored to six decimals and read identically by both sides;
    # the difference left is S8's rounding in the paper (4e-9 per weight)
    assert np.allclose(got, want, rtol=0, atol=1e-5)


def test_the_log_model_takes_over_at_twenty_or_younger(reg):
    from falconage.models.quadratic import QuadraticClock

    m = QuadraticClock.from_registry(reg, "cage")
    X = pd.read_csv(DATA / "cage_betas.csv.gz", index_col=0).T
    t = m.terms
    lin = m.intercept + X[m.features].to_numpy() @ t["lin"].to_numpy() \
        + (X[m.features].to_numpy() ** 2) @ t["lin_sq"].to_numpy()
    d = fa.FalconData(X=X, obs=pd.DataFrame(index=X.index), modality="dna_methylation")
    got = fa.score(d, clocks=["cage"]).scores["cage"].to_numpy()
    assert np.allclose(got[lin > 20], lin[lin > 20])
    assert not np.allclose(got[lin <= 20], lin[lin <= 20])
