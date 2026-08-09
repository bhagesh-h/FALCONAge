"""BMIQ: put type II probes on the type I scale.

WHY THE TWO PROBE TYPES NEED RECONCILING. An Infinium type I probe reads both of
its beads in one colour; a type II probe reads methylated in green and
unmethylated in red. The chemistries differ, and the consequence is not subtle:
type II betas are compressed toward the middle, so the same underlying
methylation reads lower at the top of the range and higher at the bottom. On the
corpus's EPIC v1 arrays the median type I beta is 0.095 and the median type II
beta is 0.646 -- most of that is which probes are where on the genome, and some
of it is the chemistry, and nothing downstream can separate them.

It matters here because **clock coefficients were fitted on corrected data**. A
clock trained on BMIQ-normalised betas and applied to raw ones is being asked
about a slightly different quantity at every type II probe, which is most of
them: 723,760 of 865,918 on EPIC v1.

THE METHOD (Teschendorff et al., Bioinformatics 2013;29:189-196)

1. Fit a three-state beta mixture -- unmethylated, hemimethylated, methylated --
   to the type I betas, and another to the type II betas.
2. Assign every probe to the state it most likely belongs to.
3. For the unmethylated and methylated states, map each type II probe through
   its own fitted distribution and back out through the corresponding type I
   one, so a probe at the 30th percentile of type II's unmethylated state comes
   out at the 30th percentile of type I's.
4. Rescale the hemimethylated state linearly into the gap the other two leave.

Step 3 is the whole idea: a quantile map *within a state*, not across the whole
distribution. Mapping the two distributions wholesale would also erase the real
difference in which genomic regions each chemistry covers.

FITTED BY EM WITH METHOD-OF-MOMENTS M-STEPS. The original uses RPMM's
beta-mixture fitter. There is no RPMM here and porting one is not the point: a
three-component beta mixture is a short EM, and a beta distribution's two
parameters follow from a weighted mean and variance in closed form. That is what
this does, and the fit is checked against a mixture drawn with known parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..core.container import FalconData
from ..core.errors import DataError

__all__ = ["BetaMixture", "bmiq", "fit_beta_mixture"]

#: Betas are clamped inside this before fitting. A beta distribution has no
#: density at exactly 0 or 1 and real matrices contain both.
EPS = 1e-4


@dataclass
class BetaMixture:
    """A fitted three-state beta mixture: weights, and (a, b) per state."""

    weights: np.ndarray            # (3,)
    a: np.ndarray                  # (3,)
    b: np.ndarray                  # (3,)
    n: int
    iterations: int
    converged: bool

    def responsibilities(self, x: np.ndarray) -> np.ndarray:
        from scipy.stats import beta as beta_dist

        x = np.clip(np.asarray(x, dtype=np.float64), EPS, 1 - EPS)
        logp = np.stack([np.log(max(w, 1e-300)) + beta_dist.logpdf(x, ai, bi)
                         for w, ai, bi in zip(self.weights, self.a, self.b)], axis=1)
        logp -= logp.max(axis=1, keepdims=True)
        p = np.exp(logp)
        return p / p.sum(axis=1, keepdims=True)

    def state(self, x: np.ndarray) -> np.ndarray:
        """Hard assignment: 0 unmethylated, 1 hemimethylated, 2 methylated."""
        return self.responsibilities(x).argmax(axis=1)

    @property
    def means(self) -> np.ndarray:
        return self.a / (self.a + self.b)


def _mom(x: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Beta parameters from a weighted mean and variance, in closed form.

    ``a = m(m(1-m)/v - 1)``, ``b = (1-m)(m(1-m)/v - 1)``. Guarded because a
    component that collapses onto a handful of near-identical probes has a
    variance that rounds to zero, and the closed form then divides by it.
    """
    tot = float(w.sum())
    if tot < 1e-8:
        return 1.0, 1.0
    m = float((w * x).sum() / tot)
    v = float((w * (x - m) ** 2).sum() / tot)
    m = min(max(m, EPS), 1 - EPS)
    v = max(min(v, m * (1 - m) - 1e-9), 1e-9)
    k = m * (1 - m) / v - 1.0
    return max(m * k, 1e-3), max((1 - m) * k, 1e-3)


def fit_beta_mixture(x: np.ndarray, *, max_iter: int = 200,
                     tol: float = 1e-5, seed: int = 0) -> BetaMixture:
    """Three-state beta mixture by expectation-maximisation.

    Initialised by splitting at 1/3 and 2/3 rather than at random. The states
    are not exchangeable -- they mean unmethylated, hemimethylated and
    methylated -- so a random start can converge to a relabelled solution that
    then maps the wrong state onto the wrong one, and the output looks
    plausible. A fixed, ordered start makes the fit reproducible and keeps the
    labels meaning what they say.
    """
    from scipy.stats import beta as beta_dist

    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 100:
        raise DataError(
            f"a three-state beta mixture from {x.size} values is not a fit; "
            "BMIQ needs a full array, not a probe subset")
    x = np.clip(x, EPS, 1 - EPS)

    init = [x < 1 / 3, (x >= 1 / 3) & (x <= 2 / 3), x > 2 / 3]
    weights = np.array([max(m.mean(), 1e-3) for m in init])
    weights /= weights.sum()
    a = np.empty(3)
    b = np.empty(3)
    for k, m in enumerate(init):
        sel = x[m] if m.sum() > 10 else x
        a[k], b[k] = _mom(sel, np.ones_like(sel))

    prev = -np.inf
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        logp = np.stack([np.log(max(w, 1e-300)) + beta_dist.logpdf(x, ai, bi)
                         for w, ai, bi in zip(weights, a, b)], axis=1)
        mx = logp.max(axis=1, keepdims=True)
        lse = mx[:, 0] + np.log(np.exp(logp - mx).sum(axis=1))
        ll = float(lse.sum())
        r = np.exp(logp - lse[:, None])

        weights = np.clip(r.mean(axis=0), 1e-6, None)
        weights /= weights.sum()
        for k in range(3):
            a[k], b[k] = _mom(x, r[:, k])

        if abs(ll - prev) < tol * max(abs(ll), 1.0):
            converged = True
            break
        prev = ll

    return BetaMixture(weights=weights, a=a, b=b, n=int(x.size),
                       iterations=it, converged=converged)


def _map_state(x: np.ndarray, src: tuple[float, float],
               dst: tuple[float, float]) -> np.ndarray:
    """Quantile map through two fitted beta distributions.

    Where a value sits in the source distribution, put it at the same place in
    the target one. Clamped away from the tails: the inverse CDF of a beta at
    exactly 0 or 1 is 0 or 1, and a single probe landing there would come back
    as a hard 0 or 1 that no assay produced.
    """
    from scipy.stats import beta as beta_dist

    q = beta_dist.cdf(np.clip(x, EPS, 1 - EPS), src[0], src[1])
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return beta_dist.ppf(q, dst[0], dst[1])


def bmiq(data: FalconData, *, probe_type: pd.Series | None = None,
         platform: str | None = None, max_iter: int = 200) -> FalconData:
    """Normalise type II probes onto the type I scale, sample by sample.

    Parameters
    ----------
    probe_type
        ``"I"``/``"II"`` per feature. Taken from the array manifest when not
        given, which needs the manifest fetch.
    platform
        Overrides the dataset's declared platform for the manifest lookup.

    Notes
    -----
    Fitted per sample, because the type I/type II offset is a property of a
    chip and a run rather than of a cohort. Fitting once across samples would
    be faster and would smear one array's dye batch onto every other.

    The correction is applied to type II probes only. Type I probes are the
    reference and come out unchanged, which is worth stating because it means
    BMIQ never moves a clock built purely on type I probes.
    """
    if probe_type is None:
        from .manifest import load_manifest

        plat = platform or data.platform
        if not plat:
            raise DataError(
                "bmiq needs to know which probes are type I and which are type "
                "II.\n  Declare the platform on the data, pass platform=, or "
                "pass probe_type= directly.")
        man = load_manifest(plat)
        probe_type = man["type"]

    types = probe_type.reindex(data.X.columns)
    is_i = (types == "I").to_numpy()
    is_ii = (types == "II").to_numpy()
    if is_i.sum() < 1000 or is_ii.sum() < 1000:
        raise DataError(
            f"this matrix has {int(is_i.sum())} type I and {int(is_ii.sum())} "
            "type II probes with a known design.\n"
            "  BMIQ fits a mixture to each and needs a full array; on a "
            "targeted panel or a clock-sized subset there is nothing to fit.")

    X = data.X.to_numpy(dtype=np.float64).copy()
    notes = []
    for i in range(X.shape[0]):
        row = X[i]
        xi = row[is_i]
        xii = row[is_ii]
        if not (np.isfinite(xi).sum() > 100 and np.isfinite(xii).sum() > 100):
            notes.append({"sample": str(data.X.index[i]), "status": "skipped: too few finite betas"})
            continue

        mi = fit_beta_mixture(xi, max_iter=max_iter)
        mii = fit_beta_mixture(xii, max_iter=max_iter)

        # Order both fits by mean so state k means the same thing in each. The
        # EM start is ordered, but a degenerate array can still swap two
        # components, and a mismap here is silent and large.
        oi = np.argsort(mi.means)
        oii = np.argsort(mii.means)

        finite = np.isfinite(xii)
        state = np.full(xii.shape, -1)
        state[finite] = mii.state(xii[finite])

        out = xii.copy()
        for k in (0, 2):                       # unmethylated and methylated
            sel = finite & (state == oii[k])
            if sel.sum() == 0:
                continue
            out[sel] = _map_state(xii[sel],
                                  (mii.a[oii[k]], mii.b[oii[k]]),
                                  (mi.a[oi[k]], mi.b[oi[k]]))

        # The hemimethylated state has no stable shape to map through -- it is
        # the shoulder between two modes -- so it is stretched linearly into
        # whatever gap the corrected U and M states left.
        mid = finite & (state == oii[1])
        if mid.sum():
            lo_src, hi_src = np.nanmin(xii[mid]), np.nanmax(xii[mid])
            corrected_u = out[finite & (state == oii[0])]
            corrected_m = out[finite & (state == oii[2])]
            lo_dst = float(np.nanmax(corrected_u)) if corrected_u.size else lo_src
            hi_dst = float(np.nanmin(corrected_m)) if corrected_m.size else hi_src
            if hi_dst > lo_dst and hi_src > lo_src:
                out[mid] = lo_dst + (xii[mid] - lo_src) * (hi_dst - lo_dst) / (hi_src - lo_src)

        row[is_ii] = np.clip(out, 0.0, 1.0)
        X[i] = row
        notes.append({
            "sample": str(data.X.index[i]), "status": "ok",
            "type_i_converged": mi.converged, "type_ii_converged": mii.converged,
            "type_i_means": [round(v, 4) for v in np.sort(mi.means)],
            "type_ii_means": [round(v, 4) for v in np.sort(mii.means)],
        })

    out_data = FalconData(
        X=pd.DataFrame(X, index=data.X.index, columns=data.X.columns),
        obs=data.obs, modality=data.modality, units=data.units,
        platform=data.platform, uns=dict(data.uns))
    out_data.uns["bmiq"] = {
        "n_type_i": int(is_i.sum()), "n_type_ii": int(is_ii.sum()),
        "per_sample": notes,
        "reference": "Teschendorff et al., Bioinformatics 2013;29:189-196",
    }
    return out_data
