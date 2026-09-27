"""Fit a clock to one cohort, with every reported prediction made out of fold.

WHAT THIS IS FOR. When no published clock fits the assay a cohort has, the
honest alternative is a model fitted to that cohort and judged only on samples
it did not see. A model scored on its own training samples reports its fit,
not its accuracy, and in a cohort of a few hundred with more features than
people the two differ by a great deal.

THE ESTIMATOR. The elastic net as glmnet fits it (Friedman, Hastie &
Tibshirani 2010, J Stat Softw 33:1): for a gaussian outcome,

.. math::

    \\min_{\\beta_0, \\beta} \\frac{1}{2n}\\sum_i (y_i - \\beta_0 - x_i^\\top\\beta)^2
    + \\lambda\\Big(\\frac{1-\\alpha}{2}\\lVert\\beta\\rVert_2^2 + \\alpha\\lVert\\beta\\rVert_1\\Big)

with the predictors standardised to unit variance (the population SD, as
glmnet uses) and the coefficients returned on the original scale, solved by
cyclical coordinate descent along a decreasing grid of lambda. glmnet also
scales y to unit variance for a gaussian outcome before fitting and unscales
the coefficients after (its documentation says so); worked through, that
leaves the lasso term as written and divides the ridge term by the population
SD of y, and this does the same, so a lambda means what it means in glmnet. The grid,
the cross-validation and the choice of lambda are glmnet's: 100 values from
the smallest lambda that zeros every coefficient down to 1e-4 of it (1e-2 when
there are more features than samples), the fold-weighted CV error and its
standard error by ``cv.glmnet``'s rules, and ``lambda.min`` or ``lambda.1se``
chosen as it chooses them. ``python/tests/data/glmnet_reference.R`` holds
glmnet's own output for the test: the path agrees to about 1e-6 (glmnet's own
convergence at the smallest lambdas) and the chosen lambdas exactly.

NESTED, SO THE ERROR IS HONEST. ``fit_clock`` splits the cohort into outer
folds; within each training part it chooses lambda by inner cross-validation,
fits, and predicts the held-out fold. Every prediction it returns was made by a
model that never saw that sample, and the choice of lambda never saw it either.
The model it also returns, fitted on everyone, is for applying to new samples;
its accuracy is the out-of-fold one, not its fit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..core.errors import AnalysisError


def _standardise(X: np.ndarray):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)                      # population SD, as glmnet
    sd = np.where(sd > 0, sd, 1.0)
    return (X - mu) / sd, mu, sd


def lambda_grid(X: np.ndarray, y: np.ndarray, alpha: float, nlambda: int = 100,
                ratio: float | None = None) -> np.ndarray:
    """glmnet's lambda sequence for a gaussian outcome."""
    n, p = X.shape
    Z, _, _ = _standardise(X)
    lmax = np.max(np.abs(Z.T @ (y - y.mean()))) / (n * max(alpha, 1e-3))
    ratio = ratio if ratio is not None else (1e-4 if n > p else 1e-2)
    return np.exp(np.linspace(np.log(lmax), np.log(lmax * ratio), nlambda))


def elastic_net_path(X: np.ndarray, y: np.ndarray, alpha: float, lambdas,
                     tol: float = 1e-12, max_iter: int = 100_000):
    """Coefficients (intercept first, original scale) at each lambda, warm-started."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, p = X.shape
    Z, mu, sd = _standardise(X)
    yc = y - y.mean()
    sy = float(y.std()) or 1.0              # glmnet's internal scaling of y
    beta = np.zeros(p)
    r = yc.copy()
    out = np.empty((len(lambdas), p + 1))
    for k, lam in enumerate(lambdas):
        l1, l2 = lam * alpha, lam * (1.0 - alpha) / sy
        for _ in range(max_iter):
            delta = 0.0
            for j in range(p):
                old = beta[j]
                g = Z[:, j] @ r / n + old
                new = np.sign(g) * max(abs(g) - l1, 0.0) / (1.0 + l2)
                if new != old:
                    r -= Z[:, j] * (new - old)
                    beta[j] = new
                    delta = max(delta, abs(new - old))
            if delta < tol:
                break
        b = beta / sd
        out[k, 0] = y.mean() - mu @ b
        out[k, 1:] = b
    return out


@dataclass
class CVResult:
    lambdas: np.ndarray
    cvm: np.ndarray
    cvsd: np.ndarray
    lambda_min: float
    lambda_1se: float


def cv_elastic_net(X, y, alpha: float, foldid, lambdas=None) -> CVResult:
    """``cv.glmnet`` for a gaussian outcome: the grid from all of X, the error
    as the fold-size-weighted mean of the fold MSEs, and its standard error."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    foldid = np.asarray(foldid)
    lambdas = lambda_grid(X, y, alpha) if lambdas is None else np.asarray(lambdas)
    folds = np.unique(foldid)
    mse = np.empty((len(folds), len(lambdas)))
    w = np.empty(len(folds))
    for i, f in enumerate(folds):
        test = foldid == f
        B = elastic_net_path(X[~test], y[~test], alpha, lambdas)
        pred = B[:, :1].T + X[test] @ B[:, 1:].T
        mse[i] = ((pred - y[test, None]) ** 2).mean(axis=0)
        w[i] = test.sum()
    cvm = (w[:, None] * mse).sum(axis=0) / w.sum()
    cvsd = np.sqrt((w[:, None] * (mse - cvm) ** 2).sum(axis=0) / w.sum() / (len(folds) - 1))
    idmin = cvm <= cvm.min()
    lmin = float(lambdas[idmin].max())
    semin = (cvm + cvsd)[lambdas == lmin][0]
    l1se = float(lambdas[cvm <= semin].max())
    return CVResult(lambdas, cvm, cvsd, lmin, l1se)


@dataclass
class ClockFit:
    """A clock fitted to one cohort. ``predictions`` are out of fold."""

    predictions: pd.Series
    coefficients: pd.Series          # of the model fitted on everyone, intercept first
    alpha: float
    lambda_rule: str
    fold_lambdas: list[float]
    metrics: dict[str, float]
    notes: list[str] = field(default_factory=list)

    def predict(self, data) -> pd.Series:
        """Apply the model fitted on everyone to new samples."""
        feats = list(self.coefficients.index[1:])
        X = data.X.reindex(columns=feats)
        gone = [f for f in feats if X[f].isna().all()]
        if gone:
            raise AnalysisError(f"new data lack {len(gone)} of the model's features: "
                                f"{', '.join(gone[:5])}")
        X = X.fillna(X.mean())
        return pd.Series(self.coefficients.iloc[0] + X.to_numpy() @ self.coefficients.iloc[1:].to_numpy(),
                         index=data.X.index, name="cohort_clock")


def _folds(n: int, k: int, rng) -> np.ndarray:
    return rng.permutation(np.arange(n) % k)


def fit_clock(data, target: str = "age", *, alpha: float = 0.5, outer_folds: int = 10,
              inner_folds: int = 10, lambda_rule: str = "1se", seed: int = 20260926,
              features=None) -> ClockFit:
    """Fit an elastic net to predict ``obs[target]``, judged out of fold.

    Parameters
    ----------
    alpha
        The mixing parameter, 0.5 by default (elastic net); 1 is the lasso.
    lambda_rule
        ``"1se"`` (glmnet's default for prediction) or ``"min"``.
    features
        Columns of ``data.X`` to use; all by default.

    A missing value is filled with its feature's mean in the training part it
    falls in, never with a mean that includes the sample being predicted. The
    result is a model of this cohort, labelled as such, and not a published
    clock.
    """
    if lambda_rule not in ("1se", "min"):
        raise AnalysisError("lambda_rule is '1se' or 'min'")
    if target not in data.obs.columns:
        raise AnalysisError(f"no {target!r} column in obs")
    y = pd.to_numeric(data.obs[target], errors="coerce")
    X = data.X if features is None else data.X[list(features)]
    ok = y.notna().to_numpy()
    X, y = X.loc[ok].astype(float), y[ok].astype(float)
    X = X.loc[:, X.notna().any()]
    n = len(y)
    if n < 2 * outer_folds:
        raise AnalysisError(f"{n} samples with {target}; {outer_folds} outer folds need "
                            f"at least {2 * outer_folds}")
    rng = np.random.default_rng(seed)
    outer = _folds(n, outer_folds, rng)
    Xv, yv = X.to_numpy(), y.to_numpy()
    pred = np.empty(n)
    chosen = []
    for f in range(outer_folds):
        test = outer == f
        tr_mean = np.nanmean(Xv[~test], axis=0)
        Xtr = np.where(np.isnan(Xv[~test]), tr_mean, Xv[~test])
        Xte = np.where(np.isnan(Xv[test]), tr_mean, Xv[test])
        cv = cv_elastic_net(Xtr, yv[~test], alpha, _folds(int((~test).sum()), inner_folds, rng))
        lam = cv.lambda_1se if lambda_rule == "1se" else cv.lambda_min
        B = elastic_net_path(Xtr, yv[~test], alpha, cv.lambdas[cv.lambdas >= lam])[-1]
        pred[test] = B[0] + Xte @ B[1:]
        chosen.append(lam)

    full_mean = np.nanmean(Xv, axis=0)
    Xall = np.where(np.isnan(Xv), full_mean, Xv)
    cv = cv_elastic_net(Xall, yv, alpha, _folds(n, inner_folds, rng))
    lam = cv.lambda_1se if lambda_rule == "1se" else cv.lambda_min
    B = elastic_net_path(Xall, yv, alpha, cv.lambdas[cv.lambdas >= lam])[-1]
    coefs = pd.Series(B, index=["(Intercept)", *X.columns], name="coefficient")

    oof = pd.Series(pred, index=y.index, name="cohort_clock")
    metrics = {"n": float(n), "r": float(np.corrcoef(pred, yv)[0, 1]),
               "mae": float(np.mean(np.abs(pred - yv))),
               "n_nonzero": float((coefs.iloc[1:] != 0).sum())}
    notes = [f"A model of this cohort, fitted and judged here; not a published clock. "
             f"Predictions are out of fold ({outer_folds} outer folds, lambda by "
             f"{inner_folds}-fold inner CV, rule {lambda_rule}, alpha {alpha})."]
    return ClockFit(oof, coefs, alpha, lambda_rule, chosen, metrics, notes)
