"""MetaboAge and MetaboHealth from Nightingale NMR metabolomics, as MiMIR computes them.

Both scores were fitted in BBMRI-NL and are distributed with their reference
implementation, MiMIR (DanieleBizzarri/MiMIR, GPL-3), whose parameters ship
here (``python/tools/build_mimir.py``). Each function below follows MiMIR's
code step for step, including what it does to a sample that fails QC.

MetaboAge (van den Akker et al. 2020, Circulation: Genomic and Precision
Medicine) predicts chronological age from 56 measures. ``QCprep``:

1. keep the 56 measures;
2. drop a sample with more than one missing value among them, then one with
   more than one zero;
3. drop a sample any of whose log values lies more than 5 SD from the BBMRI-NL
   log mean (a log of zero counts as missing here);
4. z-score each measure against the BBMRI-NL mean and SD on the raw scale;
5. set a missing z to 0, the BBMRI-NL mean;

then ``apply.fit``: the intercept plus the weighted sum.

MetaboHealth (Deelen et al. 2019, Nature Communications 10:3346) is a
mortality score from 14 measures. ``comp.mort_score``: add 1 to every value of
a measure that has a zero anywhere in the cohort, take logs, z-score each
measure **within the cohort**, and take the weighted sum. The within-cohort
scaling is the published model's, and it makes one sample's score depend on
the others, so the registry marks the clock ``requires_cohort``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from ..core.errors import FeatureCoverageError
from ..registry.registry import DATA_DIR

DIR = DATA_DIR / "metabolomics"


@lru_cache(maxsize=1)
def _metaboage_params() -> pd.DataFrame:
    return pd.read_csv(DIR / "metaboage.csv", index_col="feature_id")


@lru_cache(maxsize=1)
def _metabohealth_betas() -> pd.Series:
    return pd.read_csv(DIR / "metabohealth.csv", index_col="feature_id")["coefficient"]


def _require(X: pd.DataFrame, need, clock: str) -> None:
    gone = [m for m in need if m not in X.columns]
    if gone:
        raise FeatureCoverageError(
            f"{clock}: {len(gone)} of its {len(need)} measures are not in the data "
            f"({', '.join(gone[:6])}{', ...' if len(gone) > 6 else ''}). MiMIR does "
            "not score a sample without them either. Nightingale export names are "
            "translated by fa.read_nightingale().")


def metaboage(X: pd.DataFrame) -> tuple[pd.Series, dict[str, str]]:
    """MetaboAge per sample, NaN for a sample QC removed, and why."""
    p = _metaboage_params()
    b0 = float(p.loc["(Intercept)", "coefficient"])
    p = p.drop(index="(Intercept)")
    met = list(p.index)
    _require(X, met, "metaboage")
    M = X[met].astype(float)

    removed: dict[str, str] = {}
    miss = M.isna().sum(axis=1)
    for s in M.index[miss > 1]:
        removed[s] = f"{int(miss[s])} of the 56 measures missing (MiMIR allows 1)"
    M = M[miss <= 1]
    zero = (M == 0).sum(axis=1)
    for s in M.index[zero > 1]:
        removed[s] = f"{int(zero[s])} of the 56 measures zero (MiMIR allows 1)"
    M = M[zero <= 1]
    with np.errstate(divide="ignore"):
        L = np.log(M.where(M > 0))
    out = ((L - p["log_mean"]).abs() > 5 * p["log_sd"]).any(axis=1)
    for s in M.index[out]:
        removed[s] = "a measure more than 5 SD from the BBMRI-NL log mean"
    M = M[~out]

    Z = ((M - p["mean"]) / p["sd"]).fillna(0.0)
    age = pd.Series(np.nan, index=X.index, name="metaboage")
    age[M.index] = b0 + Z.to_numpy() @ p["coefficient"].to_numpy()
    return age, removed


def metabohealth(X: pd.DataFrame) -> pd.Series:
    """MetaboHealth per sample, scaled within this cohort as MiMIR scales it."""
    beta = _metabohealth_betas()
    _require(X, list(beta.index), "metabohealth")
    M = X[list(beta.index)].astype(float).copy()
    zeros = (M == 0).any(axis=0)
    M.loc[:, zeros] = M.loc[:, zeros] + 1.0
    L = np.log(M)
    # R's scale(): column mean and SD with n - 1, over the samples present.
    Z = (L - L.mean(axis=0)) / L.std(axis=0, ddof=1)
    return (Z @ beta).rename("metabohealth")


@dataclass
class MetabolomicsClock:
    """The two NMR scores behind the formula interface the clinical clocks use."""

    CPU_ONLY = True

    clock: object

    def predict(self, data, spec=None, *, reference=None, **kw):
        name = self.clock.formula
        if name == "metaboage":
            values, removed = metaboage(data.X)
            self.removed = removed
            return values.rename(self.clock.id), None
        if name == "metabohealth":
            return metabohealth(data.X).rename(self.clock.id), None
        raise ValueError(f"unknown metabolomics formula {name!r}")


def is_metabolomics(clock) -> bool:
    return clock.formula in ("metaboage", "metabohealth")
