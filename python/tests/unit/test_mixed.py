"""The random-intercept REML fit, against nlme::lme on the same data.

``python/tests/data/lmm_nlme.R`` wrote both files: 30 people, three visits, nine
visits missing, sex constant within a person. nlme 3.1.168, REML.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from falconage.analysis.mixed import fit_random_intercept

DATA = Path(__file__).resolve().parents[1] / "data"


def _design(d: pd.DataFrame, with_sex: bool) -> np.ndarray:
    cols = [np.ones(len(d)), (d["visit"] == "v2").to_numpy(float),
            (d["visit"] == "v3").to_numpy(float)]
    if with_sex:
        cols.append((d["sex"] == "M").to_numpy(float))
    return np.column_stack(cols)


@pytest.fixture(scope="module")
def nlme():
    d = pd.read_csv(DATA / "lmm_nlme_data.csv")
    ref = pd.read_csv(DATA / "lmm_nlme_reference.csv")
    return d, ref


@pytest.mark.parametrize("model, with_sex", [("y ~ visit", False), ("y ~ visit + sex", True)])
def test_the_fit_reproduces_nlme(nlme, model, with_sex):
    d, ref = nlme
    r = ref[ref["model"] == model].set_index("term")
    fit = fit_random_intercept(d["y"], _design(d, with_sex), d["subject"])
    terms = ["(Intercept)", "visitv2", "visitv3"] + (["sexM"] if with_sex else [])
    se = np.sqrt(np.diag(fit.cov))
    for k, term in enumerate(terms):
        assert fit.beta[k] == pytest.approx(r.loc[term, "estimate"], rel=1e-7, abs=1e-7)
        assert se[k] == pytest.approx(r.loc[term, "se"], rel=1e-6)
        assert fit.df[k] == r.loc[term, "df"]
        t = fit.beta[k] / se[k]
        p = 2 * stats.t.sf(abs(t), fit.df[k])
        assert p == pytest.approx(r.loc[term, "p"], rel=1e-5)
    assert np.sqrt(fit.sigma2) == pytest.approx(r.loc["sigma", "estimate"], rel=1e-6)
    # VarCorr prints the intercept SD to seven significant digits.
    assert np.sqrt(fit.sigma2_b) == pytest.approx(r.loc["tau", "estimate"], rel=1e-6)
    assert fit.loglik_reml == pytest.approx(r.loc["logLik", "estimate"], rel=1e-8)


def test_the_visit_term_is_nlmes_f_test(nlme):
    d, ref = nlme
    r = ref[ref["model"] == "y ~ visit"].set_index("term")
    fit = fit_random_intercept(d["y"], _design(d, False), d["subject"])
    F, p, q, d2 = fit.wald([1, 2])
    assert (q, d2) == (2, int(r.loc["df_visit", "estimate"]))
    assert F == pytest.approx(r.loc["F_visit", "estimate"], rel=1e-6)
    assert p == pytest.approx(r.loc["p_visit", "estimate"], rel=1e-5)


def test_complete_pairs_give_the_paired_t_test(rng):
    """With two visits and no gaps the model's visit t is the paired t, on the
    same n - 1 degrees of freedom."""
    n = 25
    person = rng.normal(0, 4, n)
    v1 = 60 + person + rng.normal(0, 1.5, n)
    v2 = 60 + person + 0.8 + rng.normal(0, 1.5, n)
    y = np.concatenate([v1, v2])
    X = np.column_stack([np.ones(2 * n), np.r_[np.zeros(n), np.ones(n)]])
    groups = np.r_[np.arange(n), np.arange(n)]
    fit = fit_random_intercept(y, X, groups)
    t, p = stats.ttest_rel(v2, v1)
    assert fit.df[1] == n - 1
    assert fit.beta[1] / np.sqrt(fit.cov[1, 1]) == pytest.approx(t, rel=1e-6)
