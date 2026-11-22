"""Learners for building a clock, and an honest comparison between them.

No algorithm wins everywhere. Pre-selecting CpGs before an elastic net beat a
plain elastic net on blood (Li et al. 2022, doi:10.1371/journal.pcbi.1009938;
Bernabeu et al. 2023, doi:10.1186/s13073-023-01161-y); a neural network beat
it across datasets (de Lima Camillo et al. 2022, doi:10.1038/s41514-022-00085-y);
random forests and support vector regression did best on small targeted
panels. So the comparison is offered, and two rules are enforced while doing it:

* **Every step that looks at the outcome runs inside the training part.**
  Screening CpGs by their correlation with age on the whole cohort and then
  cross-validating leaks the held-out fold into the selection.
* **Picking the best learner is itself tuning.** The winner's cross-validated
  error is optimistic unless the choice is repeated inside an outer loop
  (Varma and Simon 2006, doi:10.1186/1471-2105-7-91), so
  :func:`compare_learners` reports that nested error separately, and it is the
  number to quote.

A third warning is printed rather than enforced: the most accurate
chronological-age predictor can be the worst biomarker, because the residual it
leaves has less biology in it (Zhang et al. 2019, doi:10.1186/s13073-019-0667-1).

The built-in learners need only numpy and scipy. Random forest, gradient
boosting, support vector regression and a multilayer perceptron come from
scikit-learn, installed with ``pip install "falconage[learn]"``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..core.errors import AnalysisError
from .elasticnet import cv_elastic_net, elastic_net_path

__all__ = ["LEARNERS", "ComparisonResult", "Learner", "compare_learners", "make_learner"]


# ---------------------------------------------------------------------------
# folds and metrics
# ---------------------------------------------------------------------------
def folds(n: int, k: int, rng, groups=None) -> np.ndarray:
    """Fold labels. With ``groups``, every member of a group shares a fold.

    Grouping is what keeps a person's repeated visits, or a batch or a whole
    cohort, on one side of every split; without it the model is tested on
    samples it has effectively already seen.
    """
    if groups is None:
        return rng.permutation(np.arange(n) % k)
    g = pd.Series(np.asarray(groups)).astype(str)
    uniq = g.unique()
    if len(uniq) < k:
        raise AnalysisError(f"{len(uniq)} groups cannot fill {k} folds; use fewer folds")
    order = rng.permutation(len(uniq))
    label = {u: i % k for u, i in zip(uniq[order], range(len(uniq)))}
    return g.map(label).to_numpy()


def metrics(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """MAE, MedAE, r, the slope of prediction on target, and bias by thirds.

    Correlation is inflated by a wide age range and says nothing about bias;
    median absolute error is in years and says nothing about ordering; the
    slope shows the shrinkage towards the mean every penalised model has. All
    of them are needed, and none of them alone is informative.
    """
    pred, y = np.asarray(pred, float), np.asarray(y, float)
    err = pred - y
    out = {"n": float(len(y)), "mae": float(np.mean(np.abs(err))),
           "medae": float(np.median(np.abs(err))),
           "r": float(np.corrcoef(pred, y)[0, 1]) if np.std(pred) > 0 else float("nan"),
           "slope": float(np.polyfit(y, pred, 1)[0]) if np.std(y) > 0 else float("nan")}
    cuts = np.quantile(y, [1 / 3, 2 / 3])
    for name, mask in (("bias_low", y <= cuts[0]),
                       ("bias_mid", (y > cuts[0]) & (y <= cuts[1])),
                       ("bias_high", y > cuts[1])):
        out[name] = float(np.mean(err[mask])) if mask.any() else float("nan")
    return out


# ---------------------------------------------------------------------------
# learners
# ---------------------------------------------------------------------------
class Learner:
    """``fit(X, y, rng)`` then ``predict(X)``. ``coef_`` is ``(intercept, beta)``
    on the original feature scale for a linear learner, else ``None``."""

    name = "learner"
    linear = False
    coef_: tuple[float, np.ndarray] | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, rng) -> Learner:  # pragma: no cover
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:  # pragma: no cover - overridden by nonlinear learners
            raise NotImplementedError
        b0, b = self.coef_
        return b0 + X @ b


class ElasticNet(Learner):
    """glmnet's elastic net, lambda chosen by inner cross-validation."""

    linear = True

    def __init__(self, alpha: float = 0.5, lambda_rule: str = "1se", inner_folds: int = 5,
                 tol: float | None = None):
        self.alpha, self.lambda_rule, self.inner_folds, self.tol = alpha, lambda_rule, inner_folds, tol
        self.name = {0.0: "ridge", 1.0: "lasso"}.get(alpha, f"elasticnet_{alpha:g}")

    def fit(self, X, y, rng):
        # glmnet's own convergence rule (thresh = 1e-7 on the squared change
        # relative to the outcome's variance) unless an absolute tol is given:
        # the 1e-12 fit_clock keeps for its glmnet reference test is far
        # tighter than the comparison needs and slow on collinear CpGs.
        tol = self.tol if self.tol is not None else float(np.sqrt(1e-7) * (np.std(y) or 1.0))
        self.tol = tol
        cv = cv_elastic_net(X, y, self.alpha, folds(len(y), self.inner_folds, rng), tol=tol)
        lam = cv.lambda_1se if self.lambda_rule == "1se" else cv.lambda_min
        B = elastic_net_path(X, y, self.alpha, cv.lambdas[cv.lambdas >= lam], tol=self.tol)[-1]
        self.coef_ = (float(B[0]), np.asarray(B[1:], float))
        self.lambda_ = float(lam)
        return self


class ScreenedElasticNet(Learner):
    """Keep the ``k`` features most correlated with the target, then elastic net.

    The screen is computed on the training part only, every time.
    """

    linear = True

    def __init__(self, k: int = 2000, alpha: float = 0.5, inner_folds: int = 5):
        self.k, self.alpha, self.inner_folds = k, alpha, inner_folds
        self.name = f"screen{k}_elasticnet"

    def fit(self, X, y, rng):
        Xc = X - X.mean(axis=0)
        yc = y - y.mean()
        sd = Xc.std(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.abs(Xc.T @ yc) / (np.where(sd > 0, sd, np.inf) * yc.std() * len(y))
        keep = np.argsort(-np.nan_to_num(r))[: min(self.k, X.shape[1])]
        en = ElasticNet(self.alpha, inner_folds=self.inner_folds).fit(X[:, keep], y, rng)
        beta = np.zeros(X.shape[1])
        beta[keep] = en.coef_[1]
        self.coef_ = (en.coef_[0], beta)
        return self


class PCElasticNet(Learner):
    """Principal components of the training data, then an elastic net on them.

    The recipe of the PC clocks (Higgins-Chen et al. 2022,
    doi:10.1038/s43587-022-00248-2): a component averages many CpGs, so
    technical noise in any one of them is diluted. The fitted model maps back
    to one weight per CpG, so it is still a linear clock.
    """

    linear = True

    def __init__(self, alpha: float = 0.5, inner_folds: int = 5):
        self.alpha, self.inner_folds = alpha, inner_folds
        self.name = "pc_elasticnet"

    def fit(self, X, y, rng):
        mu = X.mean(axis=0)
        U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
        keep = S > S[0] * 1e-10
        V = Vt[keep].T                                   # features x components
        en = ElasticNet(self.alpha, inner_folds=self.inner_folds).fit((X - mu) @ V, y, rng)
        b_pc0, b_pc = en.coef_
        beta = V @ b_pc
        self.coef_ = (float(b_pc0 - mu @ beta), beta)
        return self


class SklearnLearner(Learner):
    """A scikit-learn regressor. Not linear, so it cannot be written as a clock file."""

    def __init__(self, name: str, factory):
        self.name, self._factory = name, factory

    def fit(self, X, y, rng):
        self.model_ = self._factory(int(rng.integers(0, 2**31 - 1)))
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return np.asarray(self.model_.predict(X), float)


def _sk(name: str):
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:
        raise AnalysisError(
            f"the {name!r} learner needs scikit-learn: pip install \"falconage[learn]\"") from exc
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    return {
        "random_forest": lambda s: RandomForestRegressor(n_estimators=500, random_state=s, n_jobs=-1),
        "gradient_boosting": lambda s: HistGradientBoostingRegressor(random_state=s),
        "svr": lambda s: make_pipeline(StandardScaler(), SVR(C=10.0)),
        "mlp": lambda s: make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(64, 32), max_iter=2000, early_stopping=True, random_state=s)),
    }[name]


#: Every learner by name. The first four need nothing beyond numpy and scipy.
LEARNERS = ("elasticnet", "lasso", "ridge", "screened_elasticnet", "pc_elasticnet",
            "random_forest", "gradient_boosting", "svr", "mlp")
DEFAULT = ("elasticnet", "lasso", "ridge", "screened_elasticnet", "pc_elasticnet")


def make_learner(spec, inner_folds: int = 5) -> Learner:
    """A learner from its name, or a :class:`Learner` passed through."""
    if isinstance(spec, Learner):
        return spec
    if spec == "elasticnet":
        return ElasticNet(0.5, inner_folds=inner_folds)
    if spec == "lasso":
        return ElasticNet(1.0, inner_folds=inner_folds)
    if spec == "ridge":
        return ElasticNet(0.0, inner_folds=inner_folds)
    if spec == "screened_elasticnet":
        return ScreenedElasticNet(inner_folds=inner_folds)
    if spec == "pc_elasticnet":
        return PCElasticNet(inner_folds=inner_folds)
    if spec in ("random_forest", "gradient_boosting", "svr", "mlp"):
        return SklearnLearner(spec, _sk(spec))
    raise AnalysisError(f"unknown learner {spec!r}; choose from {', '.join(LEARNERS)}")


# ---------------------------------------------------------------------------
# the comparison
# ---------------------------------------------------------------------------
def horvath_transform(age: np.ndarray, adult_age: float = 20.0) -> np.ndarray:
    a = adult_age
    return np.where(age <= a, np.log(age + 1.0) - np.log(a + 1.0), (age - a) / (a + 1.0))


def horvath_inverse(y: np.ndarray, adult_age: float = 20.0) -> np.ndarray:
    a = adult_age
    return np.where(y < 0, (1.0 + a) * np.exp(y) - 1.0, (1.0 + a) * y + a)


def _xy(data, target, features):
    if target not in data.obs.columns:
        raise AnalysisError(f"no {target!r} column in obs")
    y = pd.to_numeric(data.obs[target], errors="coerce")
    X = data.X if features is None else data.X[list(features)]
    ok = y.notna().to_numpy()
    X, y = X.loc[ok].astype(float), y[ok].astype(float)
    X = X.loc[:, X.notna().any()]
    return X, y, ok


def _fill(train: np.ndarray, *others: np.ndarray):
    mean = np.nanmean(train, axis=0)
    out = [np.where(np.isnan(a), mean, a) for a in (train, *others)]
    return mean, out


@dataclass
class ComparisonResult:
    """What :func:`compare_learners` measured, every number out of fold."""

    table: pd.DataFrame
    predictions: pd.DataFrame
    #: The error of "pick the best learner", with the pick repeated inside each
    #: outer fold. This is the number to report, not the best row of ``table``.
    selected: dict[str, float]
    #: Which learner each outer fold picked.
    picks: list[str]
    notes: list[str] = field(default_factory=list)

    @property
    def best(self) -> str:
        """The learner with the lowest out-of-fold MAE."""
        return str(self.table["mae"].idxmin())

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (f"ComparisonResult({len(self.table)} learners, best {self.best!r}, "
                f"nested MAE {self.selected['mae']:.2f})")


def compare_learners(data, target: str = "age", *, learners=DEFAULT,
                     outer_folds: int = 5, inner_folds: int = 3, groups: str | None = None,
                     transform: str | None = None, features=None,
                     seed: int = 20261122) -> ComparisonResult:
    """Compare learners for building a clock, every number out of fold.

    Parameters
    ----------
    learners
        Names from :data:`LEARNERS` or :class:`Learner` objects. The default is
        the five that need nothing beyond numpy and scipy.
    groups
        An ``obs`` column whose members must share a fold: the subject for
        repeated measures, or the batch or cohort for leave-one-group-out.
    transform
        ``"horvath"`` fits Horvath's log-linear transform of age and inverts it,
        which matters for children; ``None`` fits the target as it is.

    Returns
    -------
    ComparisonResult
        ``table`` has one row per learner (MAE, MedAE, r, slope, bias in each
        third of the target range, seconds); ``selected`` is the nested error
        of choosing the best learner, which is the honest one to report.
    """
    X, y, ok = _xy(data, target, features)
    g = None if groups is None else data.obs.loc[ok, groups].to_numpy()
    rng = np.random.default_rng(seed)
    outer = folds(len(y), outer_folds, rng, g)
    Xv, yv = X.to_numpy(), y.to_numpy()
    yt = horvath_transform(yv) if transform == "horvath" else yv
    back = horvath_inverse if transform == "horvath" else (lambda v: v)
    names = [make_learner(s, inner_folds).name for s in learners]

    oof = {n: np.empty(len(yv)) for n in names}
    secs = dict.fromkeys(names, 0.0)
    picks, sel = [], np.empty(len(yv))
    for f in range(outer_folds):
        te = outer == f
        _, (Xtr, Xte) = _fill(Xv[~te], Xv[te])
        inner = folds(int((~te).sum()), inner_folds, rng,
                      None if g is None else g[~te])
        inner_mae = {}
        for spec, n in zip(learners, names):
            t0 = time.perf_counter()
            m = make_learner(spec, inner_folds).fit(Xtr, yt[~te], rng)
            oof[n][te] = back(m.predict(Xte))
            # the same learner's inner-CV error on the training part, for the pick
            errs = []
            for j in range(inner_folds):
                it = inner == j
                mi = make_learner(spec, inner_folds).fit(Xtr[~it], yt[~te][~it], rng)
                errs.append(np.abs(back(mi.predict(Xtr[it])) - yv[~te][it]))
            inner_mae[n] = float(np.mean(np.concatenate(errs)))
            secs[n] += time.perf_counter() - t0
        pick = min(inner_mae, key=inner_mae.get)
        picks.append(pick)
        sel[te] = oof[pick][te]

    rows = {n: {**metrics(oof[n], yv), "seconds": round(secs[n], 2)} for n in names}
    table = pd.DataFrame.from_dict(rows, orient="index").sort_values("mae")
    notes = [
        "Every prediction was made by a model that never saw the sample.",
        "Report the nested error in `selected`, not the best row of the table: "
        "choosing the best of several learners is itself a fitted choice.",
        "The most accurate chronological-age predictor is not necessarily the best "
        "biomarker; test the residual against an outcome on another cohort.",
    ]
    return ComparisonResult(
        table=table, predictions=pd.DataFrame(oof, index=y.index).assign(**{target: yv}),
        selected={**metrics(sel, yv)}, picks=picks, notes=notes)
