"""Batch correction that does not change a result you already reported.

THE PROBLEM. ComBat estimates its location and scale parameters from every
sample it is given at once. Add a plate and re-run, and every previously
corrected value moves -- and with it every epigenetic age already reported.
Measured directly on three cohorts (Comput Struct Biotechnol J 2025, iComBat,
PMC12495439), adding one batch shifted the existing samples' epigenetic ages by
a mean of 0.077, 0.36 and -0.39 years, with a maximum of **2.20 years**.

A 2.2-year retrospective change in a person's reported biological age, caused by
somebody else's sample being run three months later, is not a rounding
difference. It is the reason the individual-level critique lists "longitudinal
comparability without re-running all samples" as a prerequisite for any clinical
use (PMC12714307).

THE FIX, AND WHY IT IS A DESIGN DECISION RATHER THAN AN ALGORITHM. Fit once on a
reference cohort. Freeze the global parameters -- per-feature grand mean,
covariate effects, pooled variance -- and the empirical-Bayes hyperparameters.
Every batch afterwards is standardised against *those* and gets only its own
additive and multiplicative effects estimated. The reference is an artefact the
user keeps and version-controls, exactly like a coefficient file.

That freezing is the whole point. Without it this is `ComBat` with extra steps.

    ref = fit_batch_reference(pilot, batch_col="plate")
    ref.write("study_batch_reference.npz")
    ...six months later, on a new plate...
    corrected = apply_batch_reference(new_plate, BatchReference.read(...),
                                      batch_col="plate")

WHAT IS NOT DONE HERE. The correction is never applied silently inside
:func:`falconage.score`. A score adjusted by an untraceable factor is exactly
what the coefficient digests exist to prevent; the user calls this, and the
reference's digest goes in the run manifest.

The ComBat model itself is the parametric empirical-Bayes formulation of
Johnson, Li and Rabinovic (Biostatistics 2007), implemented here from the paper
rather than ported.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ..core.container import FalconData
from ..core.errors import FalconError

__all__ = ["BatchReference", "BatchError", "apply_batch_reference",
           "fit_batch_reference"]

#: Features per chunk. ComBat is independent across features, so the whole thing
#: chunks trivially, and a 500-sample EPICv2 matrix in one block is 3.7 GB of
#: float64 on a machine with 8 GB for everything.
CHUNK = 20_000

#: Below this, a batch's mean and variance are being estimated from too little
#: to shrink sensibly, and the correction does more harm than the batch effect.
MIN_BATCH = 8


class BatchError(FalconError):
    """Raised when a batch correction would not be defensible."""


def _design(obs: pd.DataFrame, covariates: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    """Covariate design, without an intercept -- the grand mean supplies it."""
    if not covariates:
        return np.zeros((len(obs), 0)), []
    parts, names = [], []
    for c in covariates:
        col = obs[c]
        if pd.api.types.is_numeric_dtype(col):
            parts.append(pd.to_numeric(col, errors="coerce").to_numpy(float)[:, None])
            names.append(c)
        else:
            d = pd.get_dummies(col.astype(str), prefix=c, drop_first=True, dtype=float)
            parts.append(d.to_numpy(float))
            names.extend(d.columns)
    return np.hstack(parts) if parts else np.zeros((len(obs), 0)), names


def _aprior(delta_hat: np.ndarray) -> float:
    m, s2 = float(np.mean(delta_hat)), float(np.var(delta_hat, ddof=1))
    return (2.0 * s2 + m * m) / s2


def _bprior(delta_hat: np.ndarray) -> float:
    m, s2 = float(np.mean(delta_hat)), float(np.var(delta_hat, ddof=1))
    return (m * s2 + m ** 3) / s2


def _postmean(g_hat, g_bar, n, d_star, t2):
    return (t2 * n * g_hat + d_star * g_bar) / (t2 * n + d_star)


def _postvar(sum2, n, a, b):
    return (0.5 * sum2 + b) / (n / 2.0 + a - 1.0)


def _it_sol(s_data: np.ndarray, g_hat: np.ndarray, d_hat: np.ndarray,
            g_bar: float, t2: float, a: float, b: float,
            tol: float = 1e-4, max_iter: int = 200):
    """The empirical-Bayes fixed point from Johnson et al., section 3.

    Iterates the two posterior means to convergence. ``max_iter`` is a backstop:
    the iteration is a contraction in practice, and a matrix that does not
    converge in 200 rounds is a matrix with a degenerate feature in it.
    """
    n = np.sum(~np.isnan(s_data), axis=0).astype(float)
    g_old, d_old = g_hat.copy(), d_hat.copy()
    for _ in range(max_iter):
        g_new = _postmean(g_hat, g_bar, n, d_old, t2)
        sum2 = np.nansum((s_data - g_new[None, :]) ** 2, axis=0)
        d_new = _postvar(sum2, n, a, b)
        change = max(float(np.max(np.abs(g_new - g_old) / np.maximum(np.abs(g_old), 1e-12))),
                     float(np.max(np.abs(d_new - d_old) / np.maximum(np.abs(d_old), 1e-12))))
        g_old, d_old = g_new, d_new
        if change < tol:
            break
    return g_old, d_old


@dataclass
class BatchReference:
    """Frozen global parameters. The artefact that makes results stable.

    ``grand_mean``, ``beta`` and ``var_pooled`` are per feature; ``g_bar``,
    ``t2``, ``a_prior`` and ``b_prior`` are the empirical-Bayes hyperparameters.
    The hyperparameters are frozen as deliberately as the rest: leave them free
    and adding a batch re-estimates the prior, which moves every other batch's
    shrunk estimate, which is the drift this class exists to remove.
    """

    features: tuple[str, ...]
    grand_mean: np.ndarray
    beta: np.ndarray                    # (n_covariate_columns, n_features)
    var_pooled: np.ndarray
    covariates: tuple[str, ...]
    covariate_columns: tuple[str, ...]
    g_bar: float
    t2: float
    a_prior: float
    b_prior: float
    reference_batches: tuple[str, ...]
    n_reference_samples: int

    # -- persistence -------------------------------------------------------
    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p, grand_mean=self.grand_mean, beta=self.beta,
            var_pooled=self.var_pooled,
            meta=np.array(json.dumps({
                "features": list(self.features),
                "covariates": list(self.covariates),
                "covariate_columns": list(self.covariate_columns),
                "g_bar": self.g_bar, "t2": self.t2,
                "a_prior": self.a_prior, "b_prior": self.b_prior,
                "reference_batches": list(self.reference_batches),
                "n_reference_samples": self.n_reference_samples,
            })))
        return p

    @classmethod
    def read(cls, path: str | Path) -> BatchReference:
        z = np.load(Path(path), allow_pickle=False)
        m = json.loads(str(z["meta"]))
        return cls(features=tuple(m["features"]), grand_mean=z["grand_mean"],
                   beta=z["beta"], var_pooled=z["var_pooled"],
                   covariates=tuple(m["covariates"]),
                   covariate_columns=tuple(m["covariate_columns"]),
                   g_bar=m["g_bar"], t2=m["t2"], a_prior=m["a_prior"],
                   b_prior=m["b_prior"],
                   reference_batches=tuple(m["reference_batches"]),
                   n_reference_samples=m["n_reference_samples"])

    @property
    def digest(self) -> str:
        """Content hash, for the run manifest.

        A corrected matrix is a function of which reference corrected it, in the
        same way a score is a function of which coefficient file produced it.
        """
        h = hashlib.sha256()
        for arr in (self.grand_mean, self.beta, self.var_pooled):
            h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
        h.update(json.dumps([list(self.features), self.g_bar, self.t2,
                             self.a_prior, self.b_prior], sort_keys=True).encode())
        return h.hexdigest()

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (f"BatchReference({len(self.features)} features, "
                f"{len(self.reference_batches)} reference batch(es), "
                f"n={self.n_reference_samples}, {self.digest[:12]})")


def _batches(obs: pd.DataFrame, batch_col: str) -> pd.Series:
    if batch_col not in obs.columns:
        raise BatchError(f"no {batch_col!r} column in obs")
    b = obs[batch_col].astype(str)
    if b.isna().any() or (b == "nan").any():
        raise BatchError(f"{batch_col!r} has missing values; every sample must "
                         "belong to a named batch")
    return b


def _check_confounding(b: pd.Series, obs: pd.DataFrame, protect: Sequence[str]) -> None:
    """Refuse a design where batch and the variable of interest are the same thing.

    Correcting it out removes the effect along with the artefact, and the output
    looks fine. This is the one failure mode of batch correction that produces
    a clean-looking null result rather than an error.
    """
    for col in protect:
        if col not in obs.columns:
            continue
        tab = pd.crosstab(b, obs[col].astype(str))
        if (tab > 0).sum(axis=1).max() == 1 and tab.shape[1] > 1:
            raise BatchError(
                f"{col!r} is nested inside {b.name!r}: every batch contains only "
                f"one level of it.\n"
                "  Correcting this design removes the effect you are looking for "
                "along with the batch effect, and returns a clean null.\n"
                "  Pass protect=() to override, and say so in the methods.")


def fit_batch_reference(data: FalconData, *, batch_col: str,
                        covariates: Sequence[str] = (),
                        protect: Sequence[str] = ("condition", "group"),
                        min_batch: int = MIN_BATCH) -> BatchReference:
    """Fit the frozen global parameters on a reference cohort.

    Parameters
    ----------
    covariates
        Columns of ``obs`` whose effect should be preserved rather than removed
        -- typically age and sex. Numeric columns enter as themselves;
        categorical ones as dummies.
    protect
        Columns checked for being nested inside batch. See
        :func:`_check_confounding`; pass ``()`` to skip.
    """
    b = _batches(data.obs, batch_col)
    counts = b.value_counts()
    small = counts[counts < min_batch]
    if len(small):
        raise BatchError(
            f"batch(es) {list(small.index)[:5]} have fewer than {min_batch} samples.\n"
            "  A mean and variance estimated from that few is noise, and the "
            "correction will inject it into every feature.\n"
            "  Merge them, drop them, or lower min_batch and say so.")
    _check_confounding(b, data.obs, protect)

    X = data.X
    feats = tuple(map(str, X.columns))
    cov, cov_names = _design(data.obs, covariates)
    codes, levels = pd.factorize(b)
    n, n_b = len(b), len(levels)

    dummies = np.zeros((n, n_b))
    dummies[np.arange(n), codes] = 1.0
    design = np.hstack([dummies, cov])
    n_per = dummies.sum(axis=0)

    grand = np.empty(len(feats))
    beta = np.empty((cov.shape[1], len(feats)))
    var_pooled = np.empty(len(feats))

    for lo in range(0, len(feats), CHUNK):
        hi = min(lo + CHUNK, len(feats))
        y = X.iloc[:, lo:hi].to_numpy(dtype=np.float64)
        # Least squares rather than an explicit inverse: the batch-dummy design
        # is rank deficient the moment two batches have identical covariates,
        # and lstsq gives the minimum-norm solution instead of an exception.
        bhat, *_ = np.linalg.lstsq(design, y, rcond=None)
        grand[lo:hi] = (n_per / n) @ bhat[:n_b]
        if cov.shape[1]:
            beta[:, lo:hi] = bhat[n_b:]
        resid = y - design @ bhat
        var_pooled[lo:hi] = (resid ** 2).sum(axis=0) / n

    # Hyperparameters, from the reference batches only, and frozen from here on.
    g_hats, d_hats = [], []
    for k in range(n_b):
        rows = codes == k
        for lo in range(0, len(feats), CHUNK):
            hi = min(lo + CHUNK, len(feats))
            s = _standardise(X.iloc[rows, lo:hi].to_numpy(dtype=np.float64),
                             cov[rows], grand[lo:hi], beta[:, lo:hi],
                             var_pooled[lo:hi])
            g_hats.append(np.nanmean(s, axis=0))
            d_hats.append(np.nanvar(s, axis=0, ddof=1))
    g_all = np.concatenate(g_hats)
    d_all = np.concatenate(d_hats)
    d_all = d_all[np.isfinite(d_all) & (d_all > 0)]

    return BatchReference(
        features=feats, grand_mean=grand, beta=beta, var_pooled=var_pooled,
        covariates=tuple(covariates), covariate_columns=tuple(cov_names),
        g_bar=float(np.mean(g_all)), t2=float(np.var(g_all, ddof=1)),
        a_prior=_aprior(d_all), b_prior=_bprior(d_all),
        reference_batches=tuple(map(str, levels)), n_reference_samples=n)


def _standardise(y: np.ndarray, cov: np.ndarray, grand: np.ndarray,
                 beta: np.ndarray, var_pooled: np.ndarray) -> np.ndarray:
    mean = grand[None, :] + (cov @ beta if cov.shape[1] else 0.0)
    sd = np.sqrt(np.maximum(var_pooled, 1e-30))[None, :]
    return (y - mean) / sd


def apply_batch_reference(data: FalconData, ref: BatchReference, *,
                          batch_col: str, min_batch: int = MIN_BATCH) -> FalconData:
    """Correct a dataset against frozen parameters, one batch at a time.

    Every batch is standardised against the reference's global parameters and
    shrunk with the reference's priors, so a batch's corrected values depend on
    that batch and the reference alone. Adding a plate cannot move an earlier
    one -- there is no path through the arithmetic by which it could.
    """
    b = _batches(data.obs, batch_col)
    counts = b.value_counts()
    small = counts[counts < min_batch]
    if len(small):
        raise BatchError(
            f"batch(es) {list(small.index)[:5]} have fewer than {min_batch} samples")

    feats = [f for f in ref.features if f in data.X.columns]
    if not feats:
        raise BatchError(
            "this reference and this dataset share no features.\n"
            f"  The reference was fitted on {len(ref.features)} features "
            f"(e.g. {ref.features[0]}) and the data carries "
            f"{data.n_features} (e.g. {list(data.X.columns)[:1]}).")
    missing = len(ref.features) - len(feats)
    idx = {f: i for i, f in enumerate(ref.features)}
    take = np.array([idx[f] for f in feats])

    cov, cov_names = _design(data.obs, ref.covariates)
    if list(cov_names) != list(ref.covariate_columns):
        raise BatchError(
            f"covariate columns differ from the reference.\n"
            f"  reference: {list(ref.covariate_columns)}\n"
            f"  here:      {list(cov_names)}\n"
            "  A factor level present in one and not the other changes the "
            "design, so the frozen coefficients no longer mean the same thing.")

    out = data.X.copy()
    grand = ref.grand_mean[take]
    beta = ref.beta[:, take] if ref.beta.size else ref.beta
    vp = ref.var_pooled[take]

    for level in dict.fromkeys(b):
        rows = (b == level).to_numpy()
        y = data.X.loc[rows, feats].to_numpy(dtype=np.float64)
        s = _standardise(y, cov[rows], grand, beta, vp)
        g_hat = np.nanmean(s, axis=0)
        d_hat = np.nanvar(s, axis=0, ddof=1)
        d_hat = np.where(np.isfinite(d_hat) & (d_hat > 0), d_hat, 1.0)
        g_star, d_star = _it_sol(s, g_hat, d_hat, ref.g_bar, ref.t2,
                                 ref.a_prior, ref.b_prior)
        adj = ((s - g_star[None, :]) / np.sqrt(d_star)[None, :]) * np.sqrt(vp)[None, :]
        adj = adj + grand[None, :] + (cov[rows] @ beta if beta.size else 0.0)
        out.loc[rows, feats] = adj

    new = FalconData(X=out, obs=data.obs, modality=data.modality,
                     units=data.units, platform=data.platform, uns=dict(data.uns))
    new.uns["batch_reference"] = {
        "digest": ref.digest, "batch_col": batch_col,
        "features_corrected": len(feats), "features_not_in_reference": missing,
        "covariates": list(ref.covariates),
        "reference_batches": list(ref.reference_batches),
    }
    return new
