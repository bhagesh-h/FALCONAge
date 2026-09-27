"""Horvath's gold-standard normalisation: ``BMIQcalibration``, ported.

WHAT IT DOES. Horvath (2013) trained his pan-tissue clock on data normalised
this way and recommends it before applying the clock; Knight et al. (2016)
did the same for their gestational-age clock, whose published code runs it
first. Each sample is calibrated to one fixed reference profile, the
``goldstandard2`` vector of 21,368 probes (Horvath 2013, Additional file 22,
CC BY 2.0), by the beta-mixture quantile method of Teschendorff et al. (2013):

1. fit a three-state beta mixture (unmethylated, hemimethylated, methylated)
   to the gold standard, and another to the sample;
2. assign each probe of the sample to a state;
3. map unmethylated and methylated probes through the sample's fitted beta
   distribution and back out through the gold standard's;
4. stretch the hemimethylated probes linearly into the gap left between.

It is a between-sample normalisation: every sample is pulled toward the same
reference, which is what the clocks trained on it expect. ``bmiq`` in this
package is the within-sample method (type II onto type I) and a different
thing.

THE PORT. ``BMIQcalibration`` is Horvath 2013 Additional file 24, R code that
fits the gold standard with RPMM's ``blc`` (``optim`` BFGS with numerical
gradients) and each sample with its own ``blc2`` (Nelder-Mead, 50 function
evaluations), on 20,000 probes drawn by ``set.seed(1); sample(...)``, and
places its thresholds with R's ``density()``. The fits stop early, so their
result depends on the exact optimiser path. This module therefore reproduces
each of those pieces as R computes them: the Mersenne-Twister stream and
rejection sampling behind ``sample()`` (R >= 3.6), ``optim``'s ``nmmin`` and
``vmmin``, and ``density.default``'s binned FFT estimate. What remains is
floating-point order (R sums in extended precision, and its ``dbeta`` and
``qbeta`` are other implementations), which
``python/tests/data/horvath_norm_reference.R`` measures against the unmodified
R code: agreement to about 1e-9 in beta.

The code is ported as it is, including two behaviours worth knowing:

* ``mean(max(a), min(b))`` in R passes ``min(b)`` as ``trim``, so each state
  threshold is the maximum of the lower state, not the midpoint of the gap.
* Probes are classified with ``<`` and ``>`` on the gold standard and ``<=``
  and ``>=`` on the sample.

MISSING VALUES. By default the calibration runs on the gold-standard probes
the data has, with the gold standard cut to match, as the wrapper Knight et
al. published with their clock does (``absent="drop"``); ``absent="fill"``
sets absent probes to their gold-standard value instead. A value missing in
one sample is set to its gold-standard value, as Horvath's code does for a
single sample or when too many are missing for k-nearest neighbours. Both
counts are recorded.

R VERSIONS. ``sample()`` draws differently since R 3.6.0 ("Rejection") than
before ("Rounding"); the code is the same, the 20,000 probes are not. The
default is today's R. ``sample_kind="Rounding"`` reproduces the published
output of code run on older R: Knight et al.'s test dataset gives 37.3665,
38.3463 and 39.322 weeks against their stated 37.366, 38.346 and 39.324 (the
remaining 0.002 is their k-nearest-neighbour imputation of 51 missing
values), and 37.372, 38.343 and 39.338 with today's draw.

MEASURED EFFECT. On 120 EPIC v1 whole-blood samples the Horvath 2013 age
moves by -1.87 years on average (SD 1.15) against the same betas unnormalised,
and this port reproduces R's result on them to 2e-8 years.
"""

from __future__ import annotations

import hashlib
import math
from functools import lru_cache
from importlib import resources

import numpy as np
import pandas as pd

from ..core.container import FalconData
from ..core.errors import DataError

__all__ = ["horvath_normalise", "bmiq_calibration", "goldstandard2"]

#: SHA-256 of the shipped gold standard (probe id and goldstandard2 columns of
#: Horvath 2013 Additional file 22; built by python/tools/build_references.py).
GOLDSTANDARD_FILE = "references/horvath2013_goldstandard2.csv.gz"
GOLDSTANDARD_SHA256 = "d9420f1d0c71721b00e74697aa0ccec0442f8cff1bd327db9aa0e9655448fbb7"


@lru_cache(maxsize=1)
def goldstandard2() -> pd.Series:
    """Horvath 2013's gold-standard beta for each of the 21,368 probes."""
    ref = resources.files("falconage.registry") / "data" / GOLDSTANDARD_FILE
    raw = ref.read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != GOLDSTANDARD_SHA256:
        raise DataError(f"{GOLDSTANDARD_FILE} has SHA-256 {got}, not the recorded "
                        f"{GOLDSTANDARD_SHA256}; the file was altered")
    with ref.open("rb") as fh:
        s = pd.read_csv(fh, compression="gzip", index_col="feature_id")["value"]
    return s.astype(np.float64)


# --------------------------------------------------------------------------- R's sample()

_I2_32M1 = 2.328306437080797e-10


class _RRandom:
    """R's default generator: Mersenne-Twister seeded by ``set.seed``, with
    ``unif_rand``'s fixup and ``R_unif_index``'s rejection sampling."""

    def __init__(self, seed: int, rounding: bool = False):
        self.rounding = rounding
        s = seed & 0xFFFFFFFF
        for _ in range(50):
            s = (69069 * s + 1) & 0xFFFFFFFF
        key = np.empty(625, dtype=np.uint32)
        for j in range(625):
            s = (69069 * s + 1) & 0xFFFFFFFF
            key[j] = s
        # dummy[0] is mti, set to 624 so the first draw regenerates the block
        self._bg = np.random.MT19937()
        self._bg.state = {"bit_generator": "MT19937",
                          "state": {"key": key[1:], "pos": 624}}

    def unif(self) -> float:
        x = float(self._bg.random_raw()) * 2.3283064365386963e-10
        if x <= 0.0:
            return 0.5 * _I2_32M1
        if 1.0 - x <= 0.0:
            return 1.0 - 0.5 * _I2_32M1
        return x

    def _rbits(self, bits: int) -> int:
        v = 0
        for _ in range(0, bits + 1, 16):
            v = 65536 * v + int(math.floor(self.unif() * 65536))
        return v & ((1 << bits) - 1)

    def unif_index(self, dn: int) -> int:
        if self.rounding:
            return int(math.floor(dn * self.unif()))
        if dn <= 0:
            return 0
        bits = int(math.ceil(math.log2(dn)))
        while True:
            dv = self._rbits(bits)
            if dn > dv:
                return dv


def r_sample(n: int, size: int, seed: int = 1, kind: str = "Rejection") -> np.ndarray:
    """``set.seed(seed); sample(n, size)``, 0-based. ``kind`` is R's
    ``sample.kind``: "Rejection" since R 3.6.0, "Rounding" before it."""
    rng = _RRandom(seed, rounding=(kind == "Rounding"))
    x = list(range(n))
    out = np.empty(size, dtype=np.int64)
    m = n
    for i in range(size):
        j = rng.unif_index(m)
        out[i] = x[j]
        m -= 1
        x[j] = x[m]
    return out


# --------------------------------------------------------------------------- R's optim()

_BIG = 1.0e35
_RELTOL = math.sqrt(np.finfo(float).eps)     # optim's default reltol


def nmmin(fn, par, maxit: int = 500, abstol: float = -np.inf, reltol: float = _RELTOL,
          alpha: float = 1.0, beta: float = 0.5, gamma: float = 2.0) -> np.ndarray:
    """``optim(method = "Nelder-Mead")``: R's nmmin, step for step."""
    n = len(par)
    B = np.array(par, dtype=np.float64)
    if maxit <= 0:
        return B
    f = fn(B)
    if not np.isfinite(f):
        raise FloatingPointError("function cannot be evaluated at initial parameters")
    funcount = 1
    convtol = reltol * (abs(f) + reltol)
    n1, C = n + 1, n + 2
    P = np.zeros((n + 1, n + 2))
    P[n1 - 1, 0] = f
    P[:n, 0] = B
    L = 1
    size = 0.0
    step = 0.0
    for i in range(n):
        if 0.1 * abs(B[i]) > step:
            step = 0.1 * abs(B[i])
    if step == 0.0:
        step = 0.1
    for j in range(2, n1 + 1):
        P[:n, j - 1] = B
        trystep = step
        while P[j - 2, j - 1] == B[j - 2]:
            P[j - 2, j - 1] = B[j - 2] + trystep
            trystep *= 10
        size += trystep
    oldsize = size
    calcvert = True
    while True:
        if calcvert:
            for j in range(n1):
                if j + 1 != L:
                    B = P[:n, j].copy()
                    f = fn(B)
                    if not np.isfinite(f):
                        f = _BIG
                    funcount += 1
                    P[n1 - 1, j] = f
            calcvert = False
        VL = P[n1 - 1, L - 1]
        VH = VL
        H = L
        for j in range(1, n1 + 1):
            if j != L:
                f = P[n1 - 1, j - 1]
                if f < VL:
                    L, VL = j, f
                if f > VH:
                    H, VH = j, f
        if VH <= VL + convtol or VL <= abstol:
            break
        for i in range(n):
            temp = -P[i, H - 1]
            for j in range(n1):
                temp += P[i, j]
            P[i, C - 1] = temp / n
        B = (1.0 + alpha) * P[:n, C - 1] - alpha * P[:n, H - 1]
        f = fn(B)
        if not np.isfinite(f):
            f = _BIG
        funcount += 1
        VR = f
        if VR < VL:
            P[n1 - 1, C - 1] = f
            newB = gamma * B + (1 - gamma) * P[:n, C - 1]
            P[:n, C - 1] = B
            B = newB
            f = fn(B)
            if not np.isfinite(f):
                f = _BIG
            funcount += 1
            if f < VR:
                P[:n, H - 1] = B
                P[n1 - 1, H - 1] = f
            else:
                P[:n, H - 1] = P[:n, C - 1]
                P[n1 - 1, H - 1] = VR
        else:
            if VR < VH:
                P[:n, H - 1] = B
                P[n1 - 1, H - 1] = VR
            B = (1 - beta) * P[:n, H - 1] + beta * P[:n, C - 1]
            f = fn(B)
            if not np.isfinite(f):
                f = _BIG
            funcount += 1
            if f < P[n1 - 1, H - 1]:
                P[:n, H - 1] = B
                P[n1 - 1, H - 1] = f
            elif VR >= VH:
                calcvert = True
                size = 0.0
                for j in range(n1):
                    if j + 1 != L:
                        for i in range(n):
                            P[i, j] = beta * (P[i, j] - P[i, L - 1]) + P[i, L - 1]
                            size += abs(P[i, j] - P[i, L - 1])
                if size < oldsize:
                    oldsize = size
                else:
                    break
        if funcount > maxit:
            break
    return P[:n, L - 1].copy()


def _numgrad(fn, b: np.ndarray, ndeps: float = 1e-3) -> np.ndarray:
    """optim's finite-difference gradient (central, step ``ndeps``)."""
    g = np.empty_like(b)
    x = b.copy()
    for i in range(len(b)):
        x[i] = b[i] + ndeps
        v1 = fn(x)
        x[i] = b[i] - ndeps
        v2 = fn(x)
        g[i] = (v1 - v2) / (2 * ndeps)
        if not np.isfinite(g[i]):
            raise FloatingPointError(f"non-finite finite-difference value [{i + 1}]")
        x[i] = b[i]
    return g


def vmmin(fn, par, maxit: int = 100, abstol: float = -np.inf,
          reltol: float = _RELTOL) -> np.ndarray:
    """``optim(method = "BFGS")`` without a gradient: R's vmmin, step for step."""
    stepredn, acctol, reltest = 0.2, 0.0001, 10.0
    b = np.array(par, dtype=np.float64)
    n = len(b)
    if maxit <= 0:
        return b
    f = fn(b)
    if not np.isfinite(f):
        raise FloatingPointError("initial value in 'vmmin' is not finite")
    Fmin = f
    g = _numgrad(fn, b)
    gradcount = 1
    it = 1
    ilast = gradcount
    Bm = np.eye(n)
    t = np.zeros(n)
    X = np.zeros(n)
    c = np.zeros(n)
    while True:
        if ilast == gradcount:
            Bm = np.eye(n)
        X = b.copy()
        c = g.copy()
        gradproj = 0.0
        for i in range(n):
            s = 0.0
            for j in range(i + 1):
                s -= Bm[i, j] * g[j]
            for j in range(i + 1, n):
                s -= Bm[j, i] * g[j]
            t[i] = s
            gradproj += s * g[i]
        if gradproj < 0.0:
            steplength = 1.0
            accpoint = False
            while True:
                count = 0
                for i in range(n):
                    b[i] = X[i] + steplength * t[i]
                    if reltest + X[i] == reltest + b[i]:
                        count += 1
                if count < n:
                    f = fn(b)
                    accpoint = bool(np.isfinite(f)) and f <= Fmin + gradproj * steplength * acctol
                    if not accpoint:
                        steplength *= stepredn
                if count == n or accpoint:
                    break
            enough = (f > abstol) and abs(f - Fmin) > reltol * (abs(Fmin) + reltol)
            if not enough:
                count = n
                Fmin = f
            if count < n:
                Fmin = f
                g = _numgrad(fn, b)
                gradcount += 1
                it += 1
                D1 = 0.0
                for i in range(n):
                    t[i] = steplength * t[i]
                    c[i] = g[i] - c[i]
                    D1 += t[i] * c[i]
                if D1 > 0:
                    D2 = 0.0
                    for i in range(n):
                        s = 0.0
                        for j in range(i + 1):
                            s += Bm[i, j] * c[j]
                        for j in range(i + 1, n):
                            s += Bm[j, i] * c[j]
                        X[i] = s
                        D2 += s * c[i]
                    D2 = 1.0 + D2 / D1
                    for i in range(n):
                        for j in range(i + 1):
                            Bm[i, j] += (D2 * t[i] * t[j] - X[i] * t[j] - t[i] * X[j]) / D1
                else:
                    ilast = gradcount
            else:
                if ilast < gradcount:
                    count = 0
                    ilast = gradcount
        else:
            count = 0
            if ilast == gradcount:
                count = n
            else:
                ilast = gradcount
        if it >= maxit:
            break
        if gradcount - ilast > 2 * n:
            ilast = gradcount
        if not (count != n or ilast != gradcount):
            break
    return b


# --------------------------------------------------------------------------- R's density()

def _seq(lo: float, hi: float, n: int) -> np.ndarray:
    """``seq.int(lo, hi, length.out = n)`` as R's C code fills it."""
    out = np.empty(n)
    out[0] = lo
    if n > 1:
        out[n - 1] = hi
    if n > 2:
        by = (hi - lo) / (n - 1)
        for i in range(1, n - 1):
            out[i] = lo + i * by if i < n // 2 else hi - (n - 1 - i) * by
    return out


def density_mode(x: np.ndarray) -> float:
    """``d <- density(x); d$x[which.max(d$y)]`` for R >= 4.4 defaults."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    nx = x.size
    n_user = n = 512
    hi = float(np.std(x, ddof=1))
    q75, q25 = np.percentile(x, [75, 25])
    lo_ = min(hi, (q75 - q25) / 1.34)
    if not lo_:
        lo_ = hi or abs(x[0]) or 1.0
    bw = 0.9 * lo_ * nx ** (-0.2)
    frm, to = x.min() - 3 * bw, x.max() + 3 * bw
    lo, up = frm - 4 * bw, to + 4 * bw
    y = np.zeros(2 * n)
    xdelta = (up - lo) / (n - 1)
    w = 1.0 / nx
    xpos = (x - lo) / xdelta
    ix = np.floor(xpos).astype(np.int64)
    fx = xpos - ix
    for i, f in zip(ix, fx):           # BinDist, in input order
        if 0 <= i <= n - 2:
            y[i] += w * (1 - f)
            y[i + 1] += w * f
        elif i == -1:
            y[0] += w * f
        elif i == n - 1:
            y[i] += w * (1 - f)
    kords = _seq(0.0, (2 * n - 1) / (n - 1) * (up - lo), 2 * n)
    kords[n + 1:2 * n] = -kords[n - 1:0:-1]
    kords = (1 / math.sqrt(2 * math.pi)) * np.exp(-0.5 * (kords / bw) ** 2) / bw
    conv = np.fft.ifft(np.fft.fft(y) * np.conj(np.fft.fft(kords))) * (2 * n)
    dens = np.maximum(0.0, conv.real[:n] / (2 * n))
    xords = _seq(lo, up, n)
    xs = _seq(frm, to, n_user)
    return float(xs[int(np.argmax(np.interp(xs, xords, dens)))])


# --------------------------------------------------------------------------- the mixture fits

def _logdbeta(y: np.ndarray, a: float, b: float) -> np.ndarray:
    from scipy.stats import beta as beta_dist

    with np.errstate(all="ignore"):
        return beta_dist.logpdf(y, a, b)


def _beta_est(y, w, weights, optimiser: str) -> tuple[float, float]:
    """RPMM's ``betaEst`` (BFGS) or Horvath's ``betaEst2`` (Nelder-Mead, 50)."""
    obs = np.isfinite(y)
    if obs.sum() <= 1:
        return 1.0, 1.0
    y, w, weights = y[obs], w[obs], weights[obs]
    N = np.sum(weights * w)
    p = np.sum(weights * w * y) / N
    v = np.sum(weights * w * y * y) / N - p * p
    with np.errstate(all="ignore"):
        logab = np.log([p, 1 - p]) + np.log(max(1e-6, p * (1 - p) / v - 1))
    if obs.sum() == 2:
        return tuple(np.exp(logab))
    ww = w * weights

    def objf(lab):
        with np.errstate(over="ignore"):
            ab = np.exp(lab)
        d = _logdbeta(y, ab[0], ab[1])
        return -np.nansum(ww * d)

    try:
        if optimiser == "bfgs":
            par = vmmin(objf, logab)
        else:
            par = nmmin(objf, logab, maxit=50)
    except (FloatingPointError, ValueError):
        return 1.0, 1.0
    return tuple(np.exp(par))


def _blc(Y: np.ndarray, w: np.ndarray, maxiter: int, tol: float, optimiser: str) -> dict:
    """RPMM's ``blc`` and Horvath's ``blc2``, for one column of data."""
    Y = Y.astype(np.float64).copy()
    Ymn = np.nanmin(Y[Y > 0])
    Ymx = np.nanmax(Y[Y < 1])
    Y = np.maximum(Y, Ymn / 2)
    Y = np.minimum(Y, 1 - (1 - Ymx) / 2)
    obs = np.isfinite(Y)
    K = w.shape[1]
    weights = np.ones(len(Y))
    a = np.full(K, np.inf)
    b = np.full(K, np.inf)
    mu = np.full(K, np.inf)
    eta = None
    for _ in range(maxiter):
        eta = (weights[:, None] * w).sum(axis=0) / weights.sum()
        mu0 = mu.copy()
        for k in range(K):
            a[k], b[k] = _beta_est(Y, w[:, k], weights, optimiser)
            mu[k] = a[k] / (a[k] + b[k])
        ll = np.zeros((len(Y), K))
        for k in range(K):
            ll[obs, k] = _logdbeta(Y[obs], a[k], b[k])
        wmax = ll.max(axis=1)
        w = eta * np.exp(ll - wmax[:, None])
        w = w / w.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore"):
            crit = np.max(np.abs(mu - mu0))
        if crit < tol:
            break
    return {"a": a.copy(), "b": b.copy(), "eta": eta, "mu": mu.copy(), "w": w}


def _init_weights(x: np.ndarray, th) -> np.ndarray:
    w0 = np.zeros((len(x), 3))
    w0[x <= th[0], 0] = 1
    w0[(x > th[0]) & (x <= th[1]), 1] = 1
    w0[x > th[1], 2] = 1
    return w0


def _mode_or_value(v: np.ndarray) -> float:
    return float(v[0]) if v.size == 1 else density_mode(v)


def _calibrate_unit_interval(M: np.ndarray) -> np.ndarray:
    """``CalibrateUnitInterval``: rescale a sample outside [0, 1] onto [0.001, 0.999]."""
    M = M.copy()
    for i in range(M.shape[0]):
        lo, hi = np.nanmin(M[i]), np.nanmax(M[i])
        if (lo < 0 or hi > 1) and np.isfinite(lo) and np.isfinite(hi):
            slope = (0.999 - 0.001) / (hi - lo)
            M[i] = 0.001 - slope * lo + slope * M[i]
    return M


def bmiq_calibration(M: np.ndarray, gold: np.ndarray, *, nfit: int = 20000,
                     niter: int = 5, tol: float = 0.001, sample_kind: str = "Rejection",
                     detail: bool = False):
    """``BMIQcalibration(datM, goldstandard.beta)`` for samples in rows.

    ``M`` is samples by probes, in the gold standard's probe order, with no
    missing values. Returns the calibrated matrix (and the fitted mixtures
    when ``detail``).
    """
    from scipy.stats import beta as beta_dist

    M = np.asarray(M, dtype=np.float64)
    beta1 = np.asarray(gold, dtype=np.float64)
    if M.shape[1] != beta1.size:
        raise DataError("the matrix and the gold standard have different probe counts")
    M = _calibrate_unit_interval(M)
    rand = r_sample(beta1.size, min(nfit, beta1.size), seed=1, kind=sample_kind)

    em1 = _blc(beta1[rand], _init_weights(beta1[rand], (0.2, 0.75))[:, :],
               niter, tol, "bfgs")
    cls = em1["w"].argmax(axis=1)
    b1r = beta1[rand]
    th1 = (b1r[cls == 0].max(), b1r[cls == 1].max())    # see the module note on trim
    class1 = np.full(beta1.size, 2)
    class1[beta1 < th1[0]] = 1
    class1[beta1 > th1[1]] = 3
    mod1U = _mode_or_value(beta1[class1 == 1])
    mod1M = _mode_or_value(beta1[class1 == 3])

    out = M.copy()
    fits = []
    for s in range(M.shape[0]):
        beta2 = M[s]
        mod2U = density_mode(beta2[beta2 < 0.4])
        mod2M = density_mode(beta2[beta2 > 0.6])
        th2 = (th1[0] + (mod2U - mod1U), th1[1] + (mod2M - mod1M))
        b2r = beta2[rand]
        em2 = _blc(b2r, _init_weights(b2r, th2), niter, tol, "nm")
        cls2 = em2["w"].argmax(axis=1)
        if (cls2 == 1).any():
            sth = (b2r[cls2 == 0].max(), b2r[cls2 == 1].max())
        else:
            sth = (0.5 * b2r[cls2 == 0].max() + 0.5 * b2r[cls2 == 2].mean(),
                   1 / 3 * b2r[cls2 == 0].max() + 2 / 3 * b2r[cls2 == 2].mean())
        class2 = np.full(beta2.size, 2)
        class2[beta2 <= sth[0]] = 1
        class2[beta2 >= sth[1]] = 3
        av2 = em2["mu"]
        a1, bb1, a2, bb2 = em1["a"], em1["b"], em2["a"], em2["b"]
        nb = beta2.copy()
        selU = np.flatnonzero(class2 == 1)
        selUR = selU[beta2[selU] > av2[0]]
        selUL = selU[beta2[selU] < av2[0]]
        nb[selUR] = beta_dist.isf(beta_dist.sf(beta2[selUR], a2[0], bb2[0]), a1[0], bb1[0])
        nb[selUL] = beta_dist.ppf(beta_dist.cdf(beta2[selUL], a2[0], bb2[0]), a1[0], bb1[0])
        selM = np.flatnonzero(class2 == 3)
        selMR = selM[beta2[selM] > av2[2]]
        selML = selM[beta2[selM] < av2[2]]
        nb[selMR] = beta_dist.isf(beta_dist.sf(beta2[selMR], a2[2], bb2[2]), a1[2], bb1[2])
        selH = np.concatenate([np.flatnonzero(class2 == 2), selML])
        minH, maxH = np.nanmin(beta2[selH]), np.nanmax(beta2[selH])
        deltaUH = -np.nanmax(beta2[selU]) + np.nanmin(beta2[selH])
        deltaHM = -np.nanmax(beta2[selH]) + np.nanmin(beta2[selMR])
        nmaxH = np.nanmin(nb[selMR]) - deltaHM
        nminH = np.nanmax(nb[selU]) + deltaUH
        hf = (nmaxH - nminH) / (maxH - minH)
        nb[selH] = nminH + hf * (beta2[selH] - minH)
        out[s] = nb
        if detail:
            fits.append({"a": a2.tolist(), "b": bb2.tolist(), "mu": av2.tolist(),
                         "thresholds": [float(v) for v in sth], "hf": float(hf)})
    if detail:
        return out, {"gold": {"a": em1["a"].tolist(), "b": em1["b"].tolist(),
                              "mu": em1["mu"].tolist(), "thresholds": [float(v) for v in th1]},
                     "samples": fits}
    return out


def horvath_normalise(data: FalconData, *, absent: str = "drop",
                      sample_kind: str = "Rejection") -> FalconData:
    """Calibrate each sample to Horvath's gold standard (opt-in).

    Parameters
    ----------
    absent
        ``"drop"`` (default) calibrates on the gold-standard probes the data
        has, as Knight et al.'s published wrapper does; ``"fill"`` sets the
        absent ones to their gold-standard value first.
    sample_kind
        R's ``sample.kind`` for the 20,000-probe draw: ``"Rejection"`` (R 3.6.0
        and later, the default) or ``"Rounding"`` (earlier R).

    Only gold-standard probes are calibrated; other columns pass through
    unchanged. A missing value is set to its gold-standard value first. Both
    counts are in ``uns["horvath_normalisation"]``.

    Use it before a clock whose authors normalised this way: Horvath 2013 and
    Knight 2016 (gestational age) say so, and Knight's published code runs it.
    """
    if data.modality != "dna_methylation":
        raise DataError("Horvath's normalisation is for DNA methylation betas")
    if absent not in ("drop", "fill"):
        raise DataError("absent is 'drop' or 'fill'")
    if sample_kind not in ("Rejection", "Rounding"):
        raise DataError("sample_kind is 'Rejection' or 'Rounding'")
    full = goldstandard2()
    is_absent = ~full.index.isin(data.X.columns)
    gold = full[~is_absent] if absent == "drop" else full
    if len(gold) < 1000:
        raise DataError(f"the data carry {int((~is_absent).sum())} of the 21,368 gold-standard "
                        "probes; the calibration fits a mixture to the whole profile and "
                        "needs most of them")
    X = data.X.reindex(columns=gold.index).to_numpy(dtype=np.float64)
    missing = np.isnan(X) & gold.index.isin(data.X.columns)[None, :]
    X = np.where(np.isnan(X), gold.to_numpy()[None, :], X)
    Y = bmiq_calibration(X, gold.to_numpy(), sample_kind=sample_kind)
    newX = data.X.copy()
    keep = gold.index.isin(data.X.columns)
    newX.loc[:, gold.index[keep]] = Y[:, keep]
    out = FalconData(X=newX, obs=data.obs, modality=data.modality, units=data.units,
                     platform=data.platform, uns=dict(data.uns))
    out.uns["horvath_normalisation"] = {
        "reference": "Horvath 2013 Genome Biol 14:R115, Additional files 22 and 24",
        "n_gold_standard_probes": int(full.size),
        "n_absent_from_data": int(is_absent.sum()),
        "absent": absent,
        "sample_kind": sample_kind,
        "filled_with_gold_standard": {str(k): int(v) for k, v in zip(data.X.index, missing.sum(axis=1))},
    }
    return out
