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
from pathlib import Path

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
    """Coefficients (intercept first, original scale) at each lambda, warm-started.

    Coordinate descent as glmnet does it, including its active-set strategy:
    after a full sweep, only the non-zero coefficients are cycled until they
    converge, then one more full sweep confirms no other coefficient wants to
    enter. The answer is the same as sweeping every coefficient every time; the
    cost falls from the number of features to the number of non-zero ones,
    which is what makes a 450K-wide fit tractable. Ridge (``alpha == 0``) has
    no sparsity to exploit and a closed form, so it is solved exactly from one
    singular value decomposition instead.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, p = X.shape
    Z, mu, sd = _standardise(X)
    yc = y - y.mean()
    sy = float(y.std()) or 1.0              # glmnet's internal scaling of y
    lambdas = np.asarray(lambdas, dtype=np.float64)
    out = np.empty((len(lambdas), p + 1))

    if alpha == 0:
        U, S, Vt = np.linalg.svd(Z, full_matrices=False)
        uy = U.T @ yc
        for k, lam in enumerate(lambdas):
            l2 = lam / sy
            beta = Vt.T @ (S * uy / (S ** 2 + n * l2))
            b = beta / sd
            out[k, 0] = y.mean() - mu @ b
            out[k, 1:] = b
        return out

    beta = np.zeros(p)
    r = yc.copy()

    def sweep(idx, l1, l2):
        delta = 0.0
        for j in idx:
            old = beta[j]
            g = Z[:, j] @ r / n + old
            new = np.sign(g) * max(abs(g) - l1, 0.0) / (1.0 + l2)
            if new != old:
                r[:] -= Z[:, j] * (new - old)
                beta[j] = new
                delta = max(delta, abs(new - old))
        return delta

    everything = range(p)
    for k, lam in enumerate(lambdas):
        l1, l2 = lam * alpha, lam * (1.0 - alpha) / sy
        for _ in range(max_iter):
            if sweep(everything, l1, l2) < tol:
                break
            active = np.flatnonzero(beta)
            for _ in range(max_iter):
                if sweep(active, l1, l2) < tol:
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


def cv_elastic_net(X, y, alpha: float, foldid, lambdas=None, tol: float = 1e-12) -> CVResult:
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
        B = elastic_net_path(X[~test], y[~test], alpha, lambdas, tol=tol)
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
    #: Of the model fitted on everyone, intercept first. ``None`` for a learner
    #: that has no weight per feature (random forest, boosting, SVR, MLP).
    coefficients: pd.Series | None
    alpha: float
    lambda_rule: str
    fold_lambdas: list[float]
    metrics: dict[str, float]
    notes: list[str] = field(default_factory=list)
    learner: str = "elasticnet"
    target: str = "age"
    transform: str | None = None
    modality: str = "dna_methylation"
    platform: str | None = None
    #: Training means of the features, the fill for one a new dataset lacks.
    feature_means: pd.Series | None = None
    model: object = None

    def _features(self) -> list[str]:
        return (list(self.coefficients.index[1:]) if self.coefficients is not None
                else list(self.feature_means.index))

    def predict(self, data) -> pd.Series:
        """Apply the model fitted on everyone to new samples."""
        feats = self._features()
        X = data.X.reindex(columns=feats)
        gone = [f for f in feats if X[f].isna().all()]
        if gone and self.feature_means is None:
            raise AnalysisError(f"new data lack {len(gone)} of the model's features: "
                                f"{', '.join(gone[:5])}")
        fill = self.feature_means.reindex(feats) if self.feature_means is not None else X.mean()
        X = X.fillna(fill)
        if self.coefficients is not None:
            out = self.coefficients.iloc[0] + X.to_numpy() @ self.coefficients.iloc[1:].to_numpy()
        else:
            out = self.model.predict(X.to_numpy())
        if self.transform == "horvath":
            from .learners import horvath_inverse
            out = horvath_inverse(out)
        return pd.Series(out, index=data.X.index, name="cohort_clock")

    def to_clock(self, clock_id: str, outdir, *, name: str | None = None,
                 tissue=("whole blood",), platform=None, citation: str = "") -> Path:
        """Write this model as a clock FALCONAge can add and score.

        Writes ``<clock_id>.yaml``, ``<clock_id>.csv`` (the coefficients, intercept
        first) and ``<clock_id>_reference.csv`` (training means, used to fill a
        CpG a dataset lacks) into ``outdir``, and returns the YAML path. Load it
        with :func:`falconage.registry.add_clock` or ``falconage run
        --clocks-dir``. The entry records how the model was fitted, so the
        provenance travels with the weights.
        """
        import yaml

        if self.coefficients is None:
            raise AnalysisError(
                f"a {self.learner} model has no weight per feature, so it cannot be "
                "written as a clock file. Use a linear learner (elasticnet, lasso, "
                "ridge, screened_elasticnet, pc_elasticnet) or keep it in Python "
                "with ClockFit.predict().")
        d = Path(outdir)
        d.mkdir(parents=True, exist_ok=True)
        coef = self.coefficients
        nz = coef.iloc[1:][coef.iloc[1:] != 0]
        pd.concat([coef.iloc[:1], nz]).rename("coefficient").to_csv(
            d / f"{clock_id}.csv", index_label="feature_id")
        if self.feature_means is not None:
            self.feature_means.reindex(nz.index).rename("value").to_csv(
                d / f"{clock_id}_reference.csv", index_label="feature_id")
        is_age = self.target.lower() in ("age", "chronological_age", "age_years")
        entry = {
            "id": clock_id,
            "name": name or f"{clock_id} (fitted with FALCONAge)",
            "data_type": self.modality,
            "tissue": list(tissue),
            "platform": list(platform or ([self.platform] if self.platform else [])),
            "training_target": ["chronological age" if is_age else self.target],
            "scale_type": "age_years" if is_age else "relative_score",
            "unit": ["years" if is_age else "unitless"],
            "generation": "first" if is_age else "other",
            "model_type": f"linear ({self.learner})",
            "coefficients": f"{clock_id}.csv",
            "citation": citation,
            "notes": " ".join(self.notes),
            "fitted": {k: (round(v, 4) if isinstance(v, float) else v)
                       for k, v in self.metrics.items()},
        }
        if self.feature_means is not None:
            entry["reference"] = f"{clock_id}_reference.csv"
        if self.transform == "horvath":
            entry["postprocess"] = [{"op": "anti_log_linear", "adult_age": 20}]
        out = d / f"{clock_id}.yaml"
        out.write_text(yaml.safe_dump(entry, sort_keys=False, allow_unicode=True),
                       encoding="utf-8")
        return out


def _folds(n: int, k: int, rng) -> np.ndarray:
    return rng.permutation(np.arange(n) % k)


def fit_clock(data, target: str = "age", *, learner="elasticnet", alpha: float = 0.5,
              outer_folds: int = 10, inner_folds: int = 10, lambda_rule: str = "1se",
              groups: str | None = None, transform: str | None = None,
              seed: int = 20260926, features=None) -> ClockFit:
    """Fit a clock to predict ``obs[target]``, judged out of fold.

    Parameters
    ----------
    learner
        ``"elasticnet"`` (default, glmnet's, with ``alpha`` and ``lambda_rule``)
        or any name from :data:`falconage.analysis.learners.LEARNERS`. Use
        :func:`~falconage.analysis.learners.compare_learners` to choose one.
    alpha
        The elastic net's mixing parameter, 0.5 by default; 1 is the lasso.
    lambda_rule
        ``"1se"`` (glmnet's default for prediction) or ``"min"``.
    groups
        An ``obs`` column whose members always share a fold: the subject for
        repeated measures, so no person is in training and test at once.
    transform
        ``"horvath"`` fits Horvath's log-linear transform of age and inverts it.
    features
        Columns of ``data.X`` to use; all by default.

    Notes
    -----
    A missing value is filled with its feature's mean in the training part it
    falls in, never with a mean that includes the sample being predicted. The
    result is a model of this cohort, labelled as such, and not a published
    clock. ``ClockFit.to_clock()`` writes it as one FALCONAge can score.
    """
    from .learners import (ElasticNet, folds, horvath_inverse, horvath_transform,
                           make_learner, metrics)

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
    g = None if groups is None else data.obs.loc[ok, groups].to_numpy()

    def build():
        if learner == "elasticnet":
            return ElasticNet(alpha, lambda_rule, inner_folds, tol=1e-12)
        return make_learner(learner, inner_folds)

    rng = np.random.default_rng(seed)
    outer = folds(n, outer_folds, rng, g)
    Xv, yv = X.to_numpy(), y.to_numpy()
    yt = horvath_transform(yv) if transform == "horvath" else yv
    back = horvath_inverse if transform == "horvath" else (lambda v: v)
    pred = np.empty(n)
    chosen = []
    for f in range(outer_folds):
        test = outer == f
        tr_mean = np.nanmean(Xv[~test], axis=0)
        Xtr = np.where(np.isnan(Xv[~test]), tr_mean, Xv[~test])
        Xte = np.where(np.isnan(Xv[test]), tr_mean, Xv[test])
        m = build().fit(Xtr, yt[~test], rng)
        pred[test] = back(m.predict(Xte))
        chosen.append(getattr(m, "lambda_", float("nan")))

    full_mean = np.nanmean(Xv, axis=0)
    Xall = np.where(np.isnan(Xv), full_mean, Xv)
    final = build().fit(Xall, yt, rng)
    coefs = None
    if final.coef_ is not None:
        coefs = pd.Series(np.r_[final.coef_[0], final.coef_[1]],
                          index=["(Intercept)", *X.columns], name="coefficient")

    oof = pd.Series(pred, index=y.index, name="cohort_clock")
    mets = metrics(pred, yv)
    mets["n_nonzero"] = float((coefs.iloc[1:] != 0).sum()) if coefs is not None else float("nan")
    split = (f"grouped by {groups!r}" if groups else "random")
    notes = [f"A model of this cohort, fitted and judged here; not a published clock. "
             f"Predictions are out of fold ({outer_folds} outer folds, {split}; "
             f"learner {final.name}"
             + (f", lambda by {inner_folds}-fold inner CV, rule {lambda_rule}, alpha {alpha}"
                if learner == "elasticnet" else "")
             + (", Horvath age transform" if transform == "horvath" else "") + ")."]
    return ClockFit(oof, coefs, alpha, lambda_rule, chosen, mets, notes,
                    learner=final.name, target=target, transform=transform,
                    modality=data.modality, platform=data.platform,
                    feature_means=pd.Series(full_mean, index=X.columns),
                    model=final)
