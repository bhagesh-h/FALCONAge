"""NLR, PLR, LMR and SII: the definitions, the units, and what is refused."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import falconage as fa
from falconage.core.errors import UnitsNotDeclaredError


def _cbc(**cols):
    return pd.DataFrame(cols, index=[f"s{i}" for i in range(len(next(iter(cols.values()))))])


def test_the_four_ratios_by_definition():
    df = _cbc(neutrophils=[4.0, 6.0], lymphocytes=[2.0, 1.5], monocytes=[0.5, 0.3],
              platelets=[250.0, 300.0])
    u = {c: "10^9/L" for c in df.columns}
    r = fa.blood_count_ratios(df, units=u)
    assert np.allclose(r["nlr"], [2.0, 4.0])
    assert np.allclose(r["lmr"], [4.0, 5.0])
    assert np.allclose(r["plr"], [125.0, 200.0])
    assert np.allclose(r["sii"], [500.0, 1200.0])   # 10^9/L


def test_units_are_converted_not_assumed():
    df = _cbc(neutrophils=[4000.0], lymphocytes=[2.0], platelets=[250.0])
    r = fa.blood_count_ratios(df, units={"neutrophils": "cells/uL", "lymphocytes": "10^9/L",
                                         "platelets": "10^3/uL"})
    assert r["nlr"].iloc[0] == pytest.approx(2.0)
    assert r["sii"].iloc[0] == pytest.approx(500.0)
    with pytest.raises(UnitsNotDeclaredError, match="neutrophils"):
        fa.blood_count_ratios(df, units={"lymphocytes": "10^9/L", "platelets": "10^9/L"})


def test_percentages_give_nlr_and_lmr_but_not_the_platelet_ratios():
    df = _cbc(neutrophils=[60.0], lymphocytes=[30.0], monocytes=[6.0], platelets=[250.0])
    r = fa.blood_count_ratios(df, units={"neutrophils": "%", "lymphocytes": "%",
                                         "monocytes": "%", "platelets": "10^9/L"})
    assert r["nlr"].iloc[0] == pytest.approx(2.0) and r["lmr"].iloc[0] == pytest.approx(5.0)
    assert "plr" not in r and "sii" not in r
    assert "absolute" in r.attrs["skipped"]["plr"]


def test_a_zero_lymphocyte_count_gives_no_ratio_rather_than_infinity():
    df = _cbc(neutrophils=[4.0], lymphocytes=[0.0])
    r = fa.blood_count_ratios(df, units={"neutrophils": "10^9/L", "lymphocytes": "10^9/L"})
    assert np.isnan(r["nlr"].iloc[0])
