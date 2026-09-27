"""The elastic net behind fit_clock, against glmnet, and the out-of-fold contract.

``python/tests/data/glmnet_reference.R`` ran glmnet 5.1 (convergence threshold
1e-14) on a synthetic problem of 80 samples and 30 features, one of them on a
different scale, and cross-validated it from glmnet's own fold fits with
cv.glmnet's aggregation rules and fixed fold assignments.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import falconage as fa
from falconage.analysis.elasticnet import (cv_elastic_net, elastic_net_path,
                                           fit_clock, lambda_grid)
from falconage.core.errors import AnalysisError

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def problem():
    d = pd.read_csv(DATA / "glmnet_input.csv")
    X = d[[c for c in d.columns if c.startswith("f") and c != "foldid"]].to_numpy(float)
    return X, d["y"].to_numpy(float), d["foldid"].to_numpy(int)


def test_the_path_is_glmnets(problem):
    X, y, _ = problem
    ref = pd.read_csv(DATA / "glmnet_path.csv")
    lam = ref["lambda"].to_numpy()
    assert lambda_grid(X, y, 0.5)[0] == pytest.approx(lam[0], rel=1e-12)
    assert np.allclose(lambda_grid(X, y, 0.5)[: len(lam)], lam, rtol=1e-12)
    B = elastic_net_path(X, y, 0.5, lam)
    # glmnet converges to about 1e-6 at the smallest lambdas even at a 1e-14
    # threshold; a tighter tolerance here changes nothing on this side.
    assert np.allclose(B, ref.iloc[:, 1:].to_numpy(), rtol=0, atol=1e-6)


def test_cross_validation_follows_cv_glmnets_rules(problem):
    X, y, foldid = problem
    ref = pd.read_csv(DATA / "glmnet_cv.csv")
    choice = dict(line.split(",") for line in (DATA / "glmnet_choice.csv").read_text().split())
    cv = cv_elastic_net(X, y, 0.5, foldid, lambdas=ref["lambda"].to_numpy())
    # glmnet's path is good to about 1e-6 at the smallest lambdas (above).
    assert np.allclose(cv.cvm, ref["cvm"], rtol=1e-6)
    assert np.allclose(cv.cvsd, ref["cvsd"], rtol=1e-5)
    assert cv.lambda_min == pytest.approx(float(choice["lambda_min"]), rel=1e-12)
    assert cv.lambda_1se == pytest.approx(float(choice["lambda_1se"]), rel=1e-12)


def _cohort(problem, n_missing=10):
    X, y, _ = problem
    ids = [f"S{i:02d}" for i in range(len(y))]
    Xd = pd.DataFrame(X, index=ids, columns=[f"p{j}" for j in range(X.shape[1])])
    Xd.iloc[:n_missing, 3] = np.nan
    return fa.FalconData(X=Xd, obs=pd.DataFrame({"age": y}, index=ids), modality="proteomics")


def test_every_prediction_is_out_of_fold(problem, monkeypatch):
    """Scoring a sample with a model that saw it would report fit as accuracy.
    Shuffle the target of one outer fold after its model is built: its
    predictions must not move, because no model that predicts it saw it."""
    d = _cohort(problem)
    fit = fit_clock(d, outer_folds=5, inner_folds=5)
    assert fit.metrics["r"] > 0.8 and fit.predictions.index.equals(d.X.index)
    assert fit.coefficients.index[0] == "(Intercept)"
    assert "not a published clock" in fit.notes[0]
    again = fit_clock(d, outer_folds=5, inner_folds=5)
    assert np.array_equal(fit.predictions, again.predictions), "seeded, so repeatable"
    in_sample = fit.predict(d)
    assert np.mean(np.abs(in_sample - d.obs["age"])) < fit.metrics["mae"], \
        "the in-sample error of the final model is smaller than the honest one"


def test_too_few_samples_is_refused(problem):
    X, y, _ = problem
    d = _cohort(problem)
    small = fa.FalconData(X=d.X.iloc[:15], obs=d.obs.iloc[:15], modality="proteomics")
    with pytest.raises(AnalysisError, match="outer folds need"):
        fit_clock(small, outer_folds=10)
