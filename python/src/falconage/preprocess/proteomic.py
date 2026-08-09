"""Plasma proteomics: Olink NPX and SomaScan RFU, onto one scale.

WHY THIS IS NOT JUST ANOTHER MATRIX READER. The two platforms that dominate
proteomic aging work do not measure the same thing in the same units, and
neither reports an absolute concentration:

**Olink** reports NPX -- Normalized Protein eXpression -- which is already
``log2`` and already *relative*. An NPX of 4.0 means "two doublings above this
platform's reference", not "4 units of protein". Differences are interpretable;
ratios and absolute values are not.

**SomaScan** reports RFU, relative fluorescence units, on a linear scale, with
the vendor's ANML normalisation applied or not depending on how the run was
ordered. RFU is also relative, to a different reference, and is not log.

So a coefficient fitted on Olink NPX cannot be applied to SomaScan RFU, or the
reverse, and no unit conversion exists between them: they are two different
relative scales anchored to two different references. That is a refusal, not a
conversion, and this module makes it one.

THE STANDARDISATION TRAP. Every published proteomic organ clock z-scores its
inputs. The z must be taken against **the training cohort's** mean and SD, which
travel with the model, and not against the cohort in front of you. Standardising
against your own samples makes every score depend on who else was in the batch:
score one person alone and every protein is exactly zero, so the model returns
its intercept -- confidently, for anybody. That is the same failure
``requires_cohort`` exists to refuse for transcriptomic clocks, and it is why
:func:`prepare_proteomic` will not compute a z-score from your data unless you
say so in as many words.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.container import FalconData
from ..core.errors import DataError

__all__ = ["PLATFORMS", "prepare_proteomic", "read_olink", "read_somascan"]

#: What each platform reports, and whether it is already on a log scale.
PLATFORMS: dict[str, dict] = {
    "olink": {"unit": "NPX", "log2": True,
              "note": "log2 relative to Olink's own reference; differences are "
                      "interpretable, absolute values are not"},
    "somascan": {"unit": "RFU", "log2": False,
                 "note": "linear relative fluorescence; ANML normalisation is a "
                         "run-order decision and is not recorded in the values"},
}


def _read_table(path, index_col, sep=None) -> pd.DataFrame:
    from pathlib import Path

    p = Path(path)
    if sep is None:
        sep = "\t" if p.suffix.lower() in (".tsv", ".txt") else ","
    df = pd.read_csv(p, sep=sep, index_col=index_col)
    return df.apply(pd.to_numeric, errors="coerce")


def read_olink(path, *, index_col: int | str = 0,
               obs: pd.DataFrame | None = None) -> FalconData:
    """An Olink NPX matrix: samples x proteins, values already log2.

    The values are left exactly as they are. NPX is the unit the coefficients
    were fitted in, and helpfully exponentiating it would produce a linear
    quantity no model here expects.
    """
    X = _read_table(path, index_col)
    if X.empty:
        raise DataError(f"{path}: no numeric columns; is this an NPX table?")
    lo, hi = float(np.nanmin(X.to_numpy())), float(np.nanmax(X.to_numpy()))
    if hi > 100:
        raise DataError(
            f"{path}: values run up to {hi:,.0f}, which is not NPX.\n"
            "  NPX is log2 and typically sits between about -2 and 20. A table "
            "in the thousands is probably SomaScan RFU -- use read_somascan.")
    return FalconData(X=X, obs=obs if obs is not None else pd.DataFrame(index=X.index),
                      modality="proteomics", platform="olink",
                      units={c: "NPX" for c in X.columns},
                      uns={"assay": "olink", "scale": "log2 relative",
                           "range": (round(lo, 3), round(hi, 3))})


def read_somascan(path, *, index_col: int | str = 0, log2: bool = True,
                  obs: pd.DataFrame | None = None) -> FalconData:
    """A SomaScan RFU matrix.

    ``log2=True`` by default, because every published SomaScan aging model this
    package knows of was fitted on log-transformed RFU, and handing a linear
    matrix to a model fitted on logs is a silent order-of-magnitude error rather
    than a visible one. The transform is recorded in ``uns`` so the run says
    which scale it scored on.
    """
    X = _read_table(path, index_col)
    if X.empty:
        raise DataError(f"{path}: no numeric columns; is this an RFU table?")
    if float(np.nanmin(X.to_numpy())) <= 0 and log2:
        raise DataError(
            f"{path}: contains non-positive values, which cannot be logged.\n"
            "  RFU is a fluorescence reading and is positive; zeros usually mean "
            "failed wells. Filter them, or pass log2=False and say why.")
    vals = np.log2(X.to_numpy()) if log2 else X.to_numpy()
    out = pd.DataFrame(vals, index=X.index, columns=X.columns)
    return FalconData(X=out, obs=obs if obs is not None else pd.DataFrame(index=X.index),
                      modality="proteomics", platform="somascan",
                      units={c: ("log2 RFU" if log2 else "RFU") for c in X.columns},
                      uns={"assay": "somascan", "log2": log2,
                           "scale": "log2 relative" if log2 else "linear relative"})


def prepare_proteomic(data: FalconData, *, reference: pd.DataFrame | None = None,
                      standardise: str = "reference") -> FalconData:
    """Put a proteomic matrix on the scale a clock's coefficients expect.

    Parameters
    ----------
    reference
        Two columns, ``mean`` and ``sd``, indexed by protein: the *training
        cohort's* statistics, which travel with the model. This is the correct
        input and the default path.
    standardise
        ``"reference"`` uses the frame above. ``"cohort"`` computes the mean and
        SD from the data in front of you -- available, refused for a single
        sample, and recorded in ``uns`` as a deviation, because it makes every
        score depend on who else was in the batch. ``"none"`` leaves the matrix
        alone.

    Raises
    ------
    DataError
        For ``standardise="reference"`` with no reference given. There is no
        sensible fallback: silently switching to the cohort would produce the
        exact failure the parameter exists to prevent.
    """
    if data.modality != "proteomics":
        raise DataError(f"prepare_proteomic on a {data.modality} dataset")

    if standardise == "none":
        return data

    X = data.X
    if standardise == "reference":
        if reference is None:
            raise DataError(
                "standardise='reference' needs the training cohort's mean and "
                "SD per protein.\n"
                "  They travel with the model, not with your samples. Falling "
                "back to your own cohort would make every score depend on who "
                "else was in the batch -- and for one sample it makes every "
                "protein exactly zero, so the model returns its intercept for "
                "anybody.\n"
                "  Pass standardise='cohort' to do it anyway, deliberately.")
        need = {"mean", "sd"}
        if not need <= set(reference.columns):
            raise DataError(f"reference needs columns {sorted(need)}; "
                            f"it has {list(reference.columns)}")
        mu = reference["mean"].reindex(X.columns)
        sd = reference["sd"].reindex(X.columns).replace(0, np.nan)
        missing = int(mu.isna().sum())
        z = (X - mu) / sd
        note = {"standardise": "reference",
                "proteins_without_reference": missing,
                "n_proteins": int(X.shape[1])}
    elif standardise == "cohort":
        if data.n_samples < 2:
            raise DataError(
                "standardise='cohort' on one sample makes every protein exactly "
                "zero, and the model then returns its intercept for anybody.")
        mu = X.mean(axis=0)
        sd = X.std(axis=0, ddof=1).replace(0, np.nan)
        z = (X - mu) / sd
        note = {"standardise": "cohort",
                "warning": "z-scored against these samples, not the training "
                           "cohort; scores are not comparable with any other run",
                "n_proteins": int(X.shape[1])}
    else:
        raise DataError(f"standardise={standardise!r}; expected reference, "
                        "cohort or none")

    out = FalconData(X=z, obs=data.obs, modality="proteomics",
                     units={c: "z" for c in X.columns},
                     platform=data.platform, uns=dict(data.uns))
    out.uns["proteomic_standardisation"] = note
    return out
