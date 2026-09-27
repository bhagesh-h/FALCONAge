"""The probe-level SE of the summary-statistic clocks uses their effective weights.

epiTOC1 and its kin average over the CpGs a sample carries, so each present
CpG weighs 1/n in the score and 1/n^2 in its variance. Propagating the stored
probe-list weight of 1.0 made the SE n times too large (20 to 190 times for the
mitotic clocks) and the implied cohort reliability absurdly negative.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import falconage as fa


def _data(reg, cid, n=30, seed=3):
    feats = list(reg.feature_ids(cid))
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.1, 0.9, len(feats))
    X = pd.DataFrame(np.clip(base + rng.normal(0, 0.05, (n, len(feats))), 0.001, 0.999),
                     columns=feats, index=[f"s{i}" for i in range(n)])
    return fa.FalconData(X=X, obs=pd.DataFrame({"age": np.linspace(20, 80, n)}, index=X.index),
                         modality="dna_methylation")


@pytest.fixture(scope="module")
def reg():
    return fa.registry.load()


@pytest.mark.parametrize("cid", ["epitoc1", "hypoclock"])
def test_a_mean_clock_propagates_one_over_n(reg, cid):
    d = _data(reg, cid)
    res = fa.score(d, clocks=[cid])
    icc = pd.Series(0.5, index=d.X.columns)
    se = fa.uncertainty.technical_se(res, d, source="probe", icc=icc)
    s2 = d.X.var(ddof=1).to_numpy()
    n = d.X.shape[1]
    want = np.sqrt((s2 * 0.5).sum()) / n       # the postprocess slope is 1 (or -1)
    got = se.se[cid].to_numpy()
    assert np.allclose(got, want, rtol=1e-9)
    dg = se.diagnostics
    implied = (dg.set_index("clock") if "clock" in dg else dg).loc[cid, "implied_cohort_icc"]
    assert 0.0 <= implied <= 1.0


def test_a_percentile_clock_weights_two_order_statistics(reg):
    cid = "stemtoc"
    d = _data(reg, cid)
    res = fa.score(d, clocks=[cid])
    icc = pd.Series(0.0, index=d.X.columns)
    se = fa.uncertainty.technical_se(res, d, source="probe", icc=icc).se[cid]
    s2 = d.X.var(ddof=1)
    # the largest possible SE puts all weight on the noisiest probe
    assert (se <= np.sqrt(s2.max()) + 1e-12).all()
    assert (se > 0).all()
