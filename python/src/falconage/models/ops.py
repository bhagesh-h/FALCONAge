"""The preprocess and postprocess operation catalogue.

Every published clock is a chain of these around one linear combination. Keeping
them as named, declarative operations in the registry rather than as a Python
lambda per clock buys three things: the R side runs the identical chain through
the same core, a chain can be printed and audited without reading code, and a
constant that is wrong is wrong in one visible place.

Each op takes and returns an array. They are written against the ``xp`` module
handle from :mod:`falconage.core.backend`, so the same source runs on numpy and
on torch without a branch.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..core.errors import ScoringError

# ---------------------------------------------------------------------------
# postprocess
# ---------------------------------------------------------------------------


def add(x, value: float, xp=np):
    """Add the model intercept.

    A separate op rather than a synthetic feature with a constant 1.0 column,
    because the published files keep it separate: Horvath's Additional File 3
    lists ``(Intercept)`` in the same table as the CpGs, Levine's Table S6 lists
    it as row one, and the packages that fold it into the dot product then
    disagree about whether their feature count is 353 or 354.
    """
    return x + value


def multiply(x, value: float, xp=np):
    return x * value


def anti_log_linear(x, adult_age: float = 20.0, xp=np):
    """Horvath's inverse age transform.

    ``F(y) = (1+adult_age)·exp(y) - 1`` below zero and ``(1+adult_age)·y +
    adult_age`` at or above it: exponential through childhood, linear through
    adulthood, continuous and differentiable at the join. The linear branch is
    why a Horvath prediction of 0.5 is 30 years and not 6 months.

    Used by Horvath2013, SkinAndBlood, PedBE, Han, the PC variants, IntrinClock,
    CorticalClock and the three Pipek clocks.
    """
    a = adult_age
    return xp.where(x < 0, (1.0 + a) * xp.exp(x) - 1.0, (1.0 + a) * x + a)


def log_linear(x, adult_age: float = 20.0, xp=np):
    """The forward transform: the inverse of :func:`anti_log_linear`."""
    a = adult_age
    return xp.where(x <= a, xp.log(x + 1.0) - xp.log(a + 1.0), (x - a) / (a + 1.0))


def expit(x, xp=np):
    """Logistic. For the clocks whose output is a log-odds."""
    return 1.0 / (1.0 + xp.exp(-x))


def exp_op(x, xp=np):
    return xp.exp(x)


def clip(x, low: float | None = None, high: float | None = None, xp=np):
    if low is not None:
        x = xp.clip(x, low, None) if xp is np else xp.clamp(x, min=low)
    if high is not None:
        x = xp.clip(x, None, high) if xp is np else xp.clamp(x, max=high)
    return x


def divide_by(x, value: float, xp=np):
    if value == 0:
        raise ScoringError("divide_by with value 0")
    return x / value


def cox_to_years(x, cox_mean: float, cox_std: float, age_mean: float, age_std: float, xp=np):
    """Standardise a Cox linear predictor and re-express it as years.

    The GrimAge family's final step. The four constants are cohort statistics,
    not fitted parameters, and they are the reason a GrimAge implementation can
    reproduce the architecture exactly and still be off by a decade: they are
    published in the supplement, not in the coefficient file, and a package that
    copies the coefficients without them silently substitutes its own.
    """
    return ((x - cox_mean) / cox_std) * age_std + age_mean


# ---------------------------------------------------------------------------
# preprocess
# ---------------------------------------------------------------------------


def scale(x, centre=None, sd=None, xp=np):
    """Per-feature centring and scaling with stored statistics.

    The statistics travel with the model. Recomputing them from the sample at
    hand is the single most common way a reimplementation goes wrong: it makes
    the score depend on who else was in the batch, so the same person scored
    twice in two cohorts gets two answers.
    """
    if centre is not None:
        x = x - centre
    if sd is not None:
        x = x / sd
    return x


def quantile_normalize(x, reference=None, xp=np):
    """Map each sample's distribution onto a stored reference distribution.

    ``reference`` is the sorted vector of target values, one per feature. With
    no reference the columns are normalised to the average empirical
    distribution of the batch, which is only correct when the batch is the
    cohort the clock was fitted on -- almost never.
    """
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, axis=1)
    ranks = np.empty_like(order)
    n, m = x.shape
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(m)[None, :]
    target = np.sort(x, axis=0).mean(axis=0) if reference is None else np.asarray(reference)
    if target.shape[0] != m:
        raise ScoringError(
            f"quantile reference has {target.shape[0]} values for {m} features")
    return np.sort(target)[ranks]


def rank_normalize(x, xp=np):
    """Per-sample ranks rescaled to [0, 1]. Used by the BiT-age family."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(np.argsort(x, axis=1), axis=1).astype(np.float64)
    return order / max(x.shape[1] - 1, 1)


def binarize(x, threshold: float = 0.5, xp=np):
    """Threshold to 0/1. CellBiAge and BiT age score the binarised matrix."""
    return (x > threshold).astype(np.float64) if xp is np else (x > threshold).double()


def beta_to_m(x, alpha: float = 1e-6, xp=np):
    """Beta to M-value: ``log2(beta / (1 - beta))``.

    The offset keeps beta exactly 0 or 1 from producing an infinity. Real arrays
    do produce exact 0 and 1 after some normalisations, so this is not
    defensive programming -- it is the difference between a number and a NaN
    that propagates through the whole dot product.
    """
    b = xp.clip(x, alpha, 1.0 - alpha) if xp is np else xp.clamp(x, alpha, 1.0 - alpha)
    return xp.log2(b / (1.0 - b))


def m_to_beta(x, xp=np):
    two = 2.0 ** x
    return two / (1.0 + two)


def simplex_projection(x, xp=np):
    """Project each row onto the probability simplex.

    Cell-type deconvolution returns proportions, and a plain least-squares fit
    returns numbers that are occasionally negative and never sum to one. The
    Euclidean projection is the smallest correction that makes the answer a
    composition; clipping at zero and renormalising is not the same thing and
    gives a different answer.
    """
    v = np.asarray(x, dtype=np.float64)
    n, d = v.shape
    u = -np.sort(-v, axis=1)
    css = np.cumsum(u, axis=1) - 1.0
    ind = np.arange(1, d + 1)
    cond = u - css / ind > 0
    rho = d - 1 - np.argmax(cond[:, ::-1], axis=1)
    theta = css[np.arange(n), rho] / (rho + 1.0)
    return np.maximum(v - theta[:, None], 0.0)


# ---------------------------------------------------------------------------
# the remaining published output transforms
# ---------------------------------------------------------------------------
# Each of these is one clock family's final step. They are small, and that is
# the point: the alternative is a lambda per clock, where a wrong constant is
# invisible and the R side has to reimplement it.


def anti_logp2(x, xp=np):
    """``e^y - 2``. Mammalian1.

    The plus-two offset is inside the training target, so the inverse has to
    subtract it after exponentiating and not before.
    """
    return xp.exp(x) - 2.0


def anti_log_log(x, xp=np):
    """``e^(-e^(-y))``. The Mammalian2 relative-age step.

    A Gompertz inverse, and it maps the whole real line into (0, 1) -- the
    output is a relative age, not years, which is why Mammalian2 carries a
    separate step to put it on a species lifespan.
    """
    return xp.exp(-xp.exp(-x))


def one_minus(x, xp=np):
    """``1 - y``. HypoClock, whose score is a hypomethylation fraction."""
    return 1.0 - x


def days_to_weeks(x, xp=np):
    """``y / 7``. Bohlin and EPICGA predict gestational age in days."""
    return x / 7.0


def days_to_months(x, xp=np):
    """``y / 30.5``. Meer, a mouse clock trained in days and reported in months.

    30.5 and not 30 or 365.25/12: it is the constant the paper used, and a
    month is not a defined length so there is no more correct value to prefer.
    """
    return x / 30.5


def scale_and_shift(x, scale: float, offset: float, xp=np):
    """``y*s + o*s``. Pasta and PastaMouse.

    Note the offset is inside the scaling, which is unusual and is what the
    published form does. Writing it as the more natural ``y*s + o`` gives a
    different answer for every sample.
    """
    return x * scale + offset * scale


def petkovich_blood(x, xp=np):
    """Petkovich mouse blood, to months.

    ``age = ((y + 1.712) / 0.1666) ** (1 / 0.4185) / 30.5``

    The fractional power is only real for a non-negative base, and a strongly
    negative prediction makes it negative. Rather than return a NaN that
    propagates silently into a mean, the base is clipped at zero, which pins
    such a sample at age zero and is the honest floor for a quantity that
    cannot be less.
    """
    base = xp.maximum((x + 1.712) / 0.1666, 0.0)
    return base ** (1.0 / 0.4185) / 30.5


def stubbs_multitissue(x, xp=np):
    """Stubbs mouse multi-tissue, to months.

    ``age = (exp(0.1207*y^2 + 1.2424*y + 2.5440) - 3) * 7 / 30.5``

    Quadratic in the linear predictor, so it is not monotone: the minimum sits
    at y = -5.147 and predictions either side of it map to the same age. That
    is a property of the published model, not of this implementation.
    """
    return (xp.exp(0.1207 * x * x + 1.2424 * x + 2.5440) - 3.0) * 7.0 / 30.5


def mortality_to_phenoage(x, xp=np):
    """The Gompertz inversion at the end of clinical PhenoAge.

    ``m   = 1 - exp(-exp(xb) * (exp(120*g) - 1) / g)``  with ``g = 0.0076927``
    ``age = 141.50225 + ln(-0.00553 * ln(1 - m)) / 0.090165``

    0.090165 and not 0.09165. Both circulate; BioAge corrected to this one in
    April 2026 and the two differ by several years on the same input. The
    clock carries a ``known_discrepancies`` note saying so.

    ``m`` is clamped away from 0 and 1 before the log, because ``ln(1 - m)``
    diverges at 1 and a sample at either end is a very sick or very well
    person, not a reason to return an infinity.
    """
    gamma = 0.0076927
    m = 1.0 - xp.exp(-xp.exp(x) * (xp.exp(120.0 * gamma) - 1.0) / gamma)
    m = xp.clip(m, 1e-12, 1.0 - 1e-12)
    return 141.50225 + xp.log(-0.00553 * xp.log(1.0 - m)) / 0.090165


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------
POSTPROCESS: dict[str, Callable] = {
    "add": add,
    "multiply": multiply,
    "divide_by": divide_by,
    "anti_log_linear": anti_log_linear,
    "log_linear": log_linear,
    "expit": expit,
    "exp": exp_op,
    "clip": clip,
    "cox_to_years": cox_to_years,
    "anti_logp2": anti_logp2,
    "anti_log_log": anti_log_log,
    "one_minus": one_minus,
    "days_to_weeks": days_to_weeks,
    "days_to_months": days_to_months,
    "scale_and_shift": scale_and_shift,
    "petkovich_blood": petkovich_blood,
    "stubbs_multitissue": stubbs_multitissue,
    "mortality_to_phenoage": mortality_to_phenoage,
    # Aliases, because the literature names these three differently in
    # different places and a registry entry copied from a paper should not
    # fail on a synonym. Same function object, so they cannot drift.
    "anti_log": exp_op,
    "sigmoid": expit,
    "add_constant": add,
}

PREPROCESS: dict[str, Callable] = {
    "scale": scale,
    "quantile_normalize": quantile_normalize,
    "rank_normalize": rank_normalize,
    "binarize": binarize,
    "beta_to_m": beta_to_m,
    "m_to_beta": m_to_beta,
    "simplex_projection": simplex_projection,
}


def apply_chain(x, chain: tuple[dict[str, Any], ...], table: dict[str, Callable], xp=np):
    """Run a declarative op chain in order.

    ``chain`` is a sequence of ``{"op": name, **kwargs}`` mappings straight out
    of the registry. Unknown ops raise rather than being skipped: a chain that
    silently drops its transform still returns a plausible number.
    """
    for step in chain:
        step = dict(step)
        name = step.pop("op", None)
        if name is None:
            raise ScoringError(f"op chain step has no 'op' key: {step}")
        fn = table.get(name)
        if fn is None:
            raise ScoringError(
                f"unknown operation {name!r}\n  known: {', '.join(sorted(table))}")
        x = fn(x, xp=xp, **step)
    return x


def describe_chain(chain: tuple[dict[str, Any], ...]) -> str:
    """One-line human-readable rendering, for ``falconage clocks info``."""
    if not chain:
        return "identity"
    parts = []
    for step in chain:
        step = dict(step)
        name = step.pop("op")
        args = ", ".join(f"{k}={v}" for k, v in step.items())
        parts.append(f"{name}({args})" if args else f"{name}()")
    return " -> ".join(parts)
