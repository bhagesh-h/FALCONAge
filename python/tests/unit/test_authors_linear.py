"""Four bundled linear clocks against their authors' own code.

``python/tests/data/authors_linear_reference.R`` ran planet's ``predictAge``
(the Robinson lab's package for the Lee et al. 2019 placental clocks) and
methylCIPHER's ``calcHRSInChPhenoAge`` (the Higgins-Chen lab's package, commit
bfe5d02) on synthetic betas covering every CpG the four clocks use.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import falconage as fa

DATA = Path(__file__).resolve().parents[1] / "data"
CLOCKS = ["leecontrol", "leerobust", "leerefinedrobust", "hrsinchphenoage"]


@pytest.fixture(scope="module")
def reference():
    X = pd.read_csv(DATA / "authors_linear_betas.csv.gz", index_col=0).T
    want = pd.read_csv(DATA / "authors_linear_reference.csv", index_col=0)
    return X, want


@pytest.mark.parametrize("clock", CLOCKS)
def test_scores_as_the_authors_code_does(reference, clock):
    X, want = reference
    tissue = "whole blood" if clock == "hrsinchphenoage" else "placenta"
    data = fa.FalconData(X=X, obs=pd.DataFrame({"tissue": tissue}, index=X.index),
                         modality="dna_methylation", platform="450K")
    got = fa.score(data, clocks=[clock]).scores[clock].reindex(want.index)
    assert np.allclose(got, want[clock], rtol=0, atol=1e-10)
