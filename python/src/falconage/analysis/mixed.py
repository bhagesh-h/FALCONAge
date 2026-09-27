"""A random-intercept linear mixed model, fitted by REML.

WHY ONE MODEL AND NOT A FRAMEWORK. Repeated clock measurements on the same
people -- visits before and after an intervention, a baseline and two
follow-ups -- need a person effect, or every visit is treated as a new person
and the between-person spread swamps the change. The model that design calls
for, and the one the intervention literature fits, is

.. math::

    y_{ij} = x_{ij}^\\top \\beta + b_i + \\varepsilon_{ij}, \\quad
    b_i \\sim N(0, \\sigma_b^2), \\quad \\varepsilon_{ij} \\sim N(0, \\sigma^2)

for person :math:`i` and visit :math:`j`. With one grouping factor the
covariance of person :math:`i`'s observations is
:math:`\\sigma^2 (I + \\theta J)`, :math:`\\theta = \\sigma_b^2 / \\sigma^2`,
whose inverse and determinant have closed forms, so the restricted likelihood
profiles to one dimension in :math:`\\theta` (Pinheiro & Bates 2000,
doi:10.1007/978-1-4419-0318-1, ch. 2). That is the whole fit: no
general-purpose mixed-model package is needed, and none is imported.

INFERENCE, EXACTLY AS nlme DOES IT. Fixed effects are tested with Wald t (one
coefficient) or F (a term), on nlme's containment degrees of freedom: a term
that varies within a person gets :math:`N - G - p_{\\text{inner}}`, one constant
within every person gets :math:`G - p_{\\text{outer}} - 1`, and the intercept
counts as inner. ``python/tests/data/lmm_nlme_*`` holds ``nlme::lme`` output on
a design with missing visits, and the fit here reproduces it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats


@dataclass
class RandomInterceptFit:
    beta: np.ndarray            # fixed effects, in the columns of X
    cov: np.ndarray             # their covariance
    df: np.ndarray              # containment degrees of freedom per coefficient
    sigma2: float               # within-person (residual) variance
    sigma2_b: float             # between-person (intercept) variance
    loglik_reml: float
    n: int
    n_groups: int

    def wald(self, idx) -> tuple[float, float, int, int]:
        """F statistic, p, numerator and denominator df for coefficients ``idx``.

        One coefficient gives F = t^2, the same test as its t. The denominator
        df is the smallest of the coefficients' df, which is nlme's rule for a
        term whose columns share a stratum and the conservative choice when
        they do not.
        """
        idx = np.atleast_1d(np.asarray(idx))
        b = self.beta[idx]
        V = self.cov[np.ix_(idx, idx)]
        q = len(idx)
        F = float(b @ np.linalg.solve(V, b)) / q
        d2 = int(self.df[idx].min())
        return F, float(stats.f.sf(F, q, d2)), q, d2


def _group_sums(A: np.ndarray, codes: np.ndarray, G: int) -> np.ndarray:
    out = np.zeros((G,) + A.shape[1:])
    np.add.at(out, codes, A)
    return out


def fit_random_intercept(y, X, groups) -> RandomInterceptFit:
    """REML fit of ``y ~ X + (1 | groups)``.

    ``X`` must carry its own intercept column. Rows with a missing ``y`` are
    the caller's to drop.
    """
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    _, codes = np.unique(np.asarray(groups), return_inverse=True)
    G = int(codes.max()) + 1
    N, p = X.shape
    if N - p < 1 or G < 2:
        raise ValueError("too few observations or groups for a mixed model")

    n_i = np.bincount(codes, minlength=G).astype(np.float64)
    Sx = _group_sums(X, codes, G)                 # G x p, per-person column sums
    Sy = _group_sums(y, codes, G)                 # G
    # W_i^{-1} = I - c_i 11' with c_i = theta / (1 + n_i theta), written as the
    # within-person cross-products plus a between-person term weighted by
    # 1/n_i - c_i = 1 / (n_i (1 + n_i theta)). The same algebra as subtracting
    # c_i s s' from X'X, without the cancellation that loses every digit when a
    # clock's within-person change is a thousandth of its between-person spread.
    Xc = X - (Sx / n_i[:, None])[codes]
    yc = y - (Sy / n_i)[codes]
    XtXw, Xtyw, ytyw = Xc.T @ Xc, Xc.T @ yc, float(yc @ yc)

    def pieces(theta):
        w = 1.0 / (n_i * (1.0 + n_i * theta))
        XWX = XtXw + (Sx * w[:, None]).T @ Sx
        XWy = Xtyw + (Sx * w[:, None]).T @ Sy
        yWy = ytyw + float(np.sum(w * Sy ** 2))
        return w, XWX, XWy, yWy

    def neg_reml(log_theta):
        theta = np.exp(log_theta)
        _, XWX, XWy, yWy = pieces(theta)
        beta = np.linalg.solve(XWX, XWy)
        rss = yWy - float(beta @ XWy)
        s2 = rss / (N - p)
        logdetW = float(np.sum(np.log1p(n_i * theta)))
        return 0.5 * ((N - p) * np.log(s2) + logdetW + np.linalg.slogdet(XWX)[1])

    # log theta from e^-30 (no person effect) to e^40 (people differ 10^8.7
    # times more than they change), which covers a clock scored on a
    # proportion scale whose within-person change is in the fifth decimal.
    res = optimize.minimize_scalar(neg_reml, bounds=(-30.0, 40.0), method="bounded",
                                   options={"xatol": 1e-12})
    theta = float(np.exp(res.x))
    _, XWX, XWy, yWy = pieces(theta)
    beta = np.linalg.solve(XWX, XWy)
    s2 = (yWy - float(beta @ XWy)) / (N - p)
    cov = s2 * np.linalg.inv(XWX)

    # Containment df (Pinheiro & Bates 2000, section 2.4.2): inner columns vary
    # within some person; the intercept is counted as inner.
    const = np.all(np.isclose(X, (Sx / n_i[:, None])[codes]), axis=0)
    inner = ~const
    inner[np.all(X == 1.0, axis=0)] = True
    p_inner, p_outer = int(inner.sum()) - 1, int((~inner).sum())
    df = np.where(inner, N - G - p_inner, G - p_outer - 1).astype(float)

    # nlme's REML log-likelihood: -1/2 [(N-p) log 2 pi + log|V| + log|X'V^-1 X|
    # + r'V^-1 r] at the estimates, which is the profiled objective plus this.
    loglik = float(-res.fun - 0.5 * (N - p) * (1.0 + np.log(2 * np.pi)))
    return RandomInterceptFit(beta=beta, cov=cov, df=df, sigma2=float(s2),
                              sigma2_b=float(theta * s2), loglik_reml=loglik,
                              n=N, n_groups=G)
