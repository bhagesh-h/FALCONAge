"""Thirteen clocks traced from the BSD catalogues back to their primary sources.

``python/tests/data/catalogue_clocks_reference.R`` ran the authors' own code
(predictGA for Bohlin, RunStochClocks for the three stochastic clocks, the
IntrinClock model object with the authors' returnAge) on a deterministic
synthetic input this file rebuilds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import falconage as fa

DATA = Path(__file__).resolve().parents[1] / "data"


def _synth(cpgs) -> pd.DataFrame:
    cpgs = sorted(cpgs)
    i = np.arange(len(cpgs))
    cols = {f"S{s}": 0.05 + 0.9 * (((i + 1) * 37 + (s + 1) * 101) % 997) / 996 for s in range(3)}
    return pd.DataFrame(cols, index=cpgs).T


@pytest.fixture(scope="module")
def reg():
    return fa.registry.load()


@pytest.mark.parametrize("clock", ["bohlin", "stoch", "stocz", "stocp", "intrinclock"])
def test_scores_as_the_authors_code_does(reg, clock):
    ref = pd.read_csv(DATA / "catalogue_clocks_reference.csv")
    ref = ref[ref.clock == clock].set_index("sample")
    X = _synth(reg.feature_ids(clock))
    d = fa.FalconData(X=X, obs=pd.DataFrame(index=X.index), modality="dna_methylation")
    got = fa.score(d, clocks=[clock]).scores[clock]
    want = ref["value"].astype(float)
    if ref["unit"].iloc[0] == "days":                  # predictGA returns days; the clock weeks
        want = want / 7
    assert np.allclose(got[want.index], want, rtol=0, atol=1e-9)


def test_the_catalogue_discrepancies_are_recorded(reg):
    """The primary source differed from the catalogues in three places."""
    prov = {c: reg.get(c).coefficient_source.provenance for c in ("bohlin", "intrinclock", "depressionbarbu")}
    assert "lambda.min" in prov["bohlin"] and len(reg.feature_ids("bohlin")) == 96
    assert "misses them by up to 1.5 years" in prov["intrinclock"]
    assert "12.2169841" in prov["depressionbarbu"]
    assert reg.get("depressionbarbu").postprocess == ()


@pytest.mark.parametrize("clock, n, intercept", [
    ("mayne", 62, 24.99026439), ("cellpopage", 42, 11.5267892697821), ("dnamfili", 20, 0.204),
    ("dnamic", 91, 0.7852837523517947), ("dnamstress", 211, -4.494069),
    ("prostatecancerkirby", 3, 6.524), ("downsyndrome", 652, None),
])
def test_counts_and_intercepts_are_the_papers(reg, clock, n, intercept):
    c = reg.get(clock)
    assert c.availability == "bundled" and c.coefficient_source.primary_source_traced
    assert len(reg.feature_ids(clock)) == n == c.n_features
    adds = [p["value"] for p in c.postprocess if p.get("op") == "add"]
    assert adds == ([] if intercept is None else [intercept])
