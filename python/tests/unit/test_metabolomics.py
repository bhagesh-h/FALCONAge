"""MetaboAge and MetaboHealth against MiMIR's own code.

``python/tests/data/mimir_reference.R`` ran MiMIR's ``QCprep``, ``apply.fit``
and ``comp.mort_score`` (DanieleBizzarri/MiMIR at 1746f2d) on the first 200
samples of MiMIR's synthetic metabolic dataset, which carry zeros and missing
values; six fail QC.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import falconage as fa
from falconage.core.errors import FeatureCoverageError
from falconage.models.metabolomics import metaboage, metabohealth

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def nmr():
    X = pd.read_csv(DATA / "mimir_input.csv", index_col=0)
    return fa.FalconData(X=X, obs=pd.DataFrame(index=X.index), modality="metabolomics_nmr")


def test_metaboage_is_mimirs(nmr):
    want = pd.read_csv(DATA / "mimir_metaboage.csv", index_col=0)["metaboage"]
    got, removed = metaboage(nmr.X)
    assert set(got.dropna().index) == set(want.index)
    assert set(removed) == set(nmr.X.index) - set(want.index)
    assert np.allclose(got[want.index], want, rtol=0, atol=1e-9)


def test_metabohealth_is_mimirs(nmr):
    want = pd.read_csv(DATA / "mimir_metabohealth.csv", index_col=0)["metabohealth"]
    got = metabohealth(nmr.X).reindex(want.index)
    assert got.isna().equals(want.isna())
    assert np.allclose(got.dropna(), want.dropna(), rtol=0, atol=1e-12)


def test_both_score_through_the_ordinary_interface(nmr):
    res = fa.score(nmr, clocks="compatible")
    assert {"metaboage", "metabohealth"} <= set(res.scores.columns)
    assert res.registry.get("metabohealth").scale_type == "mortality_log_hazard"
    with pytest.raises(Exception, match="acceleration|scale"):
        fa.acceleration(res, clocks=["metabohealth"])


def test_a_missing_measure_is_refused_by_name(nmr):
    with pytest.raises(FeatureCoverageError, match="measures are not in the data"):
        metaboage(nmr.X.drop(columns=["ala"]))


def test_nightingale_names_are_translated(tmp_path):
    p = tmp_path / "nmr.csv"
    pd.DataFrame({"sample": ["a", "b"], "Total_C": [4.1, 5.0], "UnSat": [1.2, 1.3],
                  "XXL_VLDL_P": [1e-10, 2e-10], "Mystery": [1.0, 2.0], "Age": [40, 50]}
                 ).to_csv(p, index=False)
    d = fa.read_nightingale(p)
    assert d.modality == "metabolomics_nmr"
    assert {"serum_c", "unsatdeg", "xxl_vldl_p"} <= set(d.X.columns)
    assert d.obs.attrs["untranslated"] == ["Mystery"]
    assert list(d.obs["age"]) == [40, 50]


def test_the_parameter_files_match_their_digests(registry):
    import hashlib

    from falconage.registry.registry import DATA_DIR

    for cid in ("metaboage", "metabohealth"):
        cs = registry.get(cid).coefficient_source
        assert hashlib.sha256((DATA_DIR / cs.file).read_bytes()).hexdigest() == cs.sha256, cid
