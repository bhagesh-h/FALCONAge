"""LinAge2 (Fong et al. 2025) against the authors' own script.

``python/tools/build_linage2.py`` ran the authors' linAge2.R (Supplementary
Information of the paper, CC BY 4.0) and wrote the model it fitted and its
LinAge2 for the twelve example subjects in the archive (two user examples and
ten NHANES sanity samples), which are the fixture here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import falconage as fa
from falconage.models.linage2 import linage2

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def example():
    inp = pd.read_csv(DATA / "linage2_input.csv").set_index("SEQN")
    inp.index = inp.index.astype(str)
    want = pd.read_csv(DATA / "linage2_reference.csv").set_index("SEQN")["linage2"]
    want.index = want.index.astype(str)
    return inp, want


def test_matches_the_authors_script(example):
    inp, want = example
    got = linage2(inp, age="none", sex="RIAGENDR")          # age from RIDAGEEX, in months
    assert np.allclose(got[want.index], want, rtol=0, atol=1e-9)
    assert got.attrs["ldl_set_to_zero"] == [] and got.attrs["absent_features"] == []
    # the paper's Fig. 5: subject 8881 is more than 16 years older than his age
    assert got["8881"] - inp.loc["8881", "RIDAGEEX"] / 12 > 16


def test_scores_through_score_with_age_and_sex_in_obs(example):
    inp, want = example
    obs = pd.DataFrame({"age": inp["RIDAGEEX"] / 12,
                        "sex": inp["RIAGENDR"].map({1: "male", 2: "female"})}, index=inp.index)
    X = inp.drop(columns=["RIDAGEEX", "RIAGENDR"])
    d = fa.FalconData(X=X, obs=obs, modality="clinical_chemistry")
    got = fa.score(d, clocks=["linage2"]).scores["linage2"]
    assert np.allclose(got[want.index], want, rtol=0, atol=1e-9)


def test_cotinine_is_binned_unless_already_a_category(example):
    inp, want = example
    binned = inp.copy()
    c = binned["LBXCOT"]
    binned["LBXCOT"] = np.select([c < 10, c < 100, c < 200], [0, 1, 2], 3)
    got = linage2(binned, age="none", sex="RIAGENDR", cotinine="category")
    assert np.allclose(got[want.index], want, rtol=0, atol=1e-9)


def test_missing_lipids_give_ldl_zero_as_the_authors_code_does(example):
    inp, _ = example
    one = inp.iloc[[0]].copy()
    one["LBDHDLSI"] = np.nan
    got = linage2(one, age="none", sex="RIAGENDR")
    assert got.attrs["ldl_set_to_zero"] == [one.index[0]]
    assert np.isfinite(got.iloc[0])
    one["LBXCRP"] = np.nan                                    # any other gap: no value
    assert np.isnan(linage2(one, age="none", sex="RIAGENDR").iloc[0])
