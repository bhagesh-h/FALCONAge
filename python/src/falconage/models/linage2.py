"""LinAge2: a clinical clock trained on survival, applied as its authors' code applies it.

THE CLOCK (Fong et al. 2025, npj Aging 11:29). Fifty-nine inputs from a
routine examination and blood panel, in NHANES variable names and units, plus
three scores from health questionnaire answers, LDL and the urine
albumin-to-creatinine ratio derived from them. Each is transformed (log where
the authors log it), turned into a z-score against the NHANES 1999-2000 wave
aged 40 to 50 of the same sex (median and MAD), and folded in at +/-6. The
z-scores are projected onto principal components of the 1999-2000 wave, one
basis per sex, and a Cox model on age and 17 of those components, again one
per sex, gives the mortality risk. LinAge2 is the age at which a null model
on age alone gives the same risk:

.. math::

    \\text{LinAge2} = a + \\frac{\\log(r_\\text{model} / r_\\text{null})}{\\log 2}
    \\cdot \\text{MRDT},\\qquad \\text{MRDT} = \\text{round}(\\log 2 / \\beta_\\text{age}, 2)

with age in months inside the model, each risk centred on its training means
(``predict.coxph``'s default), and the result divided by 12.

WHERE THE NUMBERS COME FROM. The authors publish code, not a table: their
``linAge2.R`` fits all of this on the NHANES data it ships with every time it
runs. ``python/tools/build_linage2.py`` runs it once and records what it
fitted in ``registry/data/linage2/linage2.json``; this module applies those
numbers. ``python/tests/data/linage2_reference.csv`` is the script's own
output for the twelve example subjects in the archive.

AS THE AUTHORS DO IT, INCLUDING:

* cotinine (``LBXCOT``, ng/mL) is binned 0 / 1 / 2 / 3 at 10, 100 and 200
  unless it is already given as that smoking category;
* a missing questionnaire answer counts as the healthy one;
* LDL is total cholesterol - triglycerides / 5 - HDL (Friedewald, mmol/L),
  and 0 when any of the three is missing. That is the authors' code, and a 0
  folds to a z-score of -6; the result records which samples it happened to.

Any other missing input makes that sample's LinAge2 missing.
"""

from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
from importlib import resources

import numpy as np
import pandas as pd

from ..core.errors import AnalysisError, DataError

PARAMS_FILE = "linage2/linage2.json"

#: The questionnaire items the three derived scores read, with the value a
#: missing answer takes in the authors' code.
FS1_DEFAULTS = {
    "BPQ020": 2, "DIQ010": 2, "HUQ010": 3, "HUQ020": 3, "HUQ050": 0, "HUQ070": 2,
    "KIQ020": 2, "MCQ010": 2, "MCQ053": 2, "MCQ160A": 2, "MCQ160B": 2, "MCQ160C": 2,
    "MCQ160D": 2, "MCQ160E": 2, "MCQ160F": 2, "MCQ160G": 2, "MCQ160I": 2, "MCQ160J": 2,
    "MCQ160K": 2, "MCQ160L": 2, "MCQ220": 2, "OSQ010A": 2, "OSQ010B": 2, "OSQ010C": 2,
    "OSQ060": 2, "PFQ056": 2,
}


@lru_cache(maxsize=1)
def parameters(sha256: str | None = None) -> dict:
    """The fitted model, from the registry's data directory."""
    raw = (resources.files("falconage.registry") / "data" / PARAMS_FILE).read_bytes()
    if sha256 is not None and hashlib.sha256(raw).hexdigest() != sha256:
        raise DataError(f"{PARAMS_FILE} does not match the SHA-256 the registry records")
    return json.loads(raw)


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(df[name], errors="coerce") if name in df else pd.Series(np.nan, index=df.index)


def _derived(df: pd.DataFrame, cotinine: str) -> tuple[pd.DataFrame, pd.Index]:
    out = df.copy()
    if cotinine == "ng/mL" and "LBXCOT" in out:
        c = _col(out, "LBXCOT")
        out["LBXCOT"] = np.select([c < 10, c < 100, c < 200, c >= 200], [0, 1, 2, 3], np.nan)
    q = {k: _col(df, k).fillna(v) for k, v in FS1_DEFAULTS.items()}
    flags = [q["BPQ020"] == 1, (q["DIQ010"] == 1) | (q["DIQ010"] == 3)] + [
        q[k] == 1 for k in ("KIQ020", "MCQ010", "MCQ053", "MCQ160A", "MCQ160C", "MCQ160D",
                            "MCQ160E", "MCQ160F", "MCQ160G", "MCQ160I", "MCQ160J", "MCQ160K",
                            "MCQ160L", "MCQ220", "OSQ010A", "OSQ010B", "OSQ010C", "OSQ060",
                            "PFQ056", "HUQ070")]
    out["fs1Score"] = sum(f.astype(float) for f in flags) / 22
    h10, h20 = q["HUQ010"], q["HUQ020"]
    out["fs2Score"] = ((h10 == 4) * 2 + (h10 == 5) * 4) * (1 - (h20 == 1) * 0.5 + (h20 == 2))
    h50 = q["HUQ050"].where(~q["HUQ050"].isin([77, 99]), 0)
    out["fs3Score"] = h50
    tc, hdl, tg = _col(df, "LBDTCSI"), _col(df, "LBDHDLSI"), _col(df, "LBDSTRSI")
    ok = tc.notna() & hdl.notna() & tg.notna()
    out["LDLV"] = np.where(ok, tc - tg / 5 - hdl, 0.0)
    out["crAlbRat"] = _col(df, "URXUMASI") / (_col(df, "URXUCRSI") * 1.1312e-4)
    return out, df.index[~ok]


def _sex(v) -> float:
    s = str(v).strip().lower()
    if s in ("1", "1.0", "m", "male", "man"):
        return 1.0
    if s in ("2", "2.0", "f", "female", "woman"):
        return 2.0
    return np.nan


def linage2(df: pd.DataFrame, *, age: str = "age", sex: str = "sex",
            cotinine: str = "ng/mL") -> pd.Series:
    """LinAge2 in years, one value per row of ``df``.

    Parameters
    ----------
    df
        The inputs in NHANES variable names and SI units, as the authors'
        codebook gives them (``registry/data/linage2/linage2.json`` lists the
        59 features; the questionnaire items are in :data:`FS1_DEFAULTS`).
    age
        Column with age in years; ``RIDAGEEX`` (months) is used if present and
        this is not.
    sex
        Column with sex (1/2 as NHANES codes it, or male/female); ``RIAGENDR``
        is used if this is absent.
    cotinine
        ``"ng/mL"`` bins ``LBXCOT`` as the authors do; ``"category"`` takes it
        as already binned 0 to 3.

    The result's ``attrs`` record the samples whose LDL was set to 0 and the
    features that were missing.
    """
    if cotinine not in ("ng/mL", "category"):
        raise AnalysisError("cotinine is 'ng/mL' or 'category'")
    p = parameters()
    if age in df:
        months = pd.to_numeric(df[age], errors="coerce") * 12
    elif "RIDAGEEX" in df:
        months = pd.to_numeric(df["RIDAGEEX"], errors="coerce")
    else:
        raise AnalysisError(f"LinAge2 needs age: no {age!r} or RIDAGEEX column")
    sexcol = df[sex] if sex in df else df.get("RIAGENDR")
    if sexcol is None:
        raise AnalysisError(f"LinAge2 needs sex: no {sex!r} or RIAGENDR column")
    sx = sexcol.map(_sex)

    d, ldl_zero = _derived(df, cotinine)
    feats = p["features"]
    absent = [f for f in feats if f not in d]
    X = pd.DataFrame({f: _col(d, f) for f in feats}, index=df.index)
    for f in feats:
        lam = p["lambda"].get(f)
        if lam is None:
            continue
        X[f] = np.log(X[f]) if lam == 0 else (X[f] ** lam - 1) / lam

    out = pd.Series(np.nan, index=df.index, name="linage2")
    for code, key in ((1.0, "male"), (2.0, "female")):
        rows = sx == code
        if not rows.any():
            continue
        Z = X.loc[rows].copy()
        for f in feats:
            n = p["normalisation"][f]
            if n["normalise"]:
                Z[f] = (Z[f] - n[f"median_{key}"]) / n[f"mad_{key}"]
        Z = Z.clip(-p["z_max"], p["z_max"])
        m = p[key]
        cov = {"chronAge": months[rows].to_numpy(dtype=float)}
        for pc, load in m["loadings"].items():
            cov[pc] = Z.to_numpy(dtype=float) @ np.asarray(load, dtype=float)
        lp = sum(b * (cov[t] - mu) for t, b, mu in zip(m["terms"], m["coefficients"], m["means"]))
        lp_null = m["null_coefficient"] * (cov["chronAge"] - m["null_mean"])
        mrdt = round(math.log(2) / m["null_coefficient"], 2)
        bio = cov["chronAge"] + (lp - lp_null) / math.log(2) * mrdt
        out.loc[rows] = bio / 12
    out.attrs["ldl_set_to_zero"] = [str(i) for i in ldl_zero]
    out.attrs["absent_features"] = absent
    out.attrs["reference"] = "Fong et al. 2025, npj Aging 11:29 (linAge2.R, CC BY 4.0)"
    return out
