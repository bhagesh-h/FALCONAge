"""Disorder as the signal: entropy, drift, and the noise barometer.

Every clock in the registry models the *mean* beta at a site. Aging also raises
the *variance*, and there is now direct evidence that a large part of what a
clock reads is the variance rather than the mean.

Tong et al. (Nature Aging 4:886-901, 2024) found that roughly 66 to 75 per cent
of Horvath2013's accuracy against chronological age is reproducible by a purely
stochastic model of methylation change, about 90 per cent for Zhang, and about
63 per cent for PhenoAge. The gradient is the useful part: the more accurately a
clock predicts chronological age, the more of it is entropy. A clock is
therefore not the only way to read an aging methylome, and on the evidence it
may not be the most informative one.

Nothing here is a clock. These are properties of a methylation matrix, they
carry no coefficients, and no ``scale_type`` gate applies because none of them
claims to be an age. That also means none of them is comparable across
datasets that were processed differently: entropy moves with normalisation,
with probe panel, and with detection-p filtering, so the only defensible
comparison is within one processed matrix.

The four readouts, and what each is for
---------------------------------------
:func:`entropy`
    Per sample. How far the methylome sits from fully committed. One number,
    bounded 0 to 1, and the closest thing here to a summary statistic.
:func:`drift`
    Per sample. How far this sample sits from the cohort's own centroid. The
    per-sample analogue of the barometer, and the one to regress on an outcome.
:func:`noise_barometer`
    Per group. Mei et al.'s statistic (Aging 15:8552-8575, 2023): the summed
    per-site standard deviation over sites whose variance rises with age.
    A property of a group, not of a person.
:func:`variable_sites`
    Per site. Which cytosines actually become more variable with age, which is
    the selection step the barometer depends on and the one most often skipped.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from ..core.errors import AnalysisError, DataError

__all__ = ["drift", "entropy", "noise_barometer", "variable_sites"]

#: Beta exactly 0 or 1 makes ``x·log x`` a 0·(-inf). The limit is 0 and that is
#: what the entropy sum should use, so the clamp only has to be small enough not
#: to move a real value: at 1e-9 the per-site contribution is 2e-8, which is
#: eleven orders of magnitude below a site that carries any information at all.
_CLAMP = 1e-9

_METHYLATION = ("dna_methylation", "rrbs")


def _beta_matrix(data, features: Sequence[str] | None,
                 *, complete: bool, what: str) -> pd.DataFrame:
    """The beta matrix these functions are allowed to work on.

    Shared by all four, because all four fail the same three ways: a modality
    that is not beta, values outside the unit interval, and a feature panel that
    differs between samples.
    """
    if data.modality not in _METHYLATION:
        raise DataError(
            f"{what} is defined on methylation beta, not on {data.modality}.\n"
            "  Entropy and the barometer both assume the value at a feature is "
            "a proportion of methylated molecules bounded by 0 and 1. A "
            "clinical panel in mg/dL has no such bound and the formula returns "
            "a number anyway, which is the failure mode worth refusing.")

    X = data.X if features is None else data.X.reindex(columns=list(features))
    if X.shape[1] == 0:
        raise DataError(f"{what} got no features to work on")

    finite = X.to_numpy(dtype=np.float64)
    seen = finite[np.isfinite(finite)]
    if seen.size and (seen.min() < -1e-6 or seen.max() > 1.0 + 1e-6):
        raise DataError(
            f"{what} needs beta values in [0, 1]; this matrix spans "
            f"{seen.min():.3g} to {seen.max():.3g}.\n"
            "  M-values are the usual cause. Convert with "
            "beta = 2**m / (2**m + 1) before calling.")

    if complete:
        keep = X.columns[X.notna().all(axis=0)]
        if len(keep) == 0:
            raise DataError(
                f"{what} with complete_sites=True found no site covered in "
                "every sample.\n"
                "  Either impute first, or pass complete_sites=False and read "
                "the n_sites column before comparing samples to each other.")
        X = X[keep]
    return X


def entropy(data, *, features: Sequence[str] | None = None,
            complete_sites: bool = True) -> pd.DataFrame:
    """Normalised Shannon entropy of the methylome, one value per sample.

    .. math::

        S = \\frac{1}{N \\log \\tfrac{1}{2}}
            \\sum_{i=1}^{N} \\Big[ \\beta_i \\log \\beta_i
                                 + (1-\\beta_i) \\log (1-\\beta_i) \\Big]

    Bounded 0 to 1 by construction. A site at beta 0 or 1 is fully committed and
    contributes nothing; a site at 0.5 is maximally uncertain and contributes
    its full share. So ``S = 0`` is a methylome where every site is decided and
    ``S = 1`` is one where none of them is.

    The normalisation by :math:`N \\log \\tfrac{1}{2}` is what makes the number
    a *rate* rather than a total, and it is the reason two samples measured on
    different panels still cannot be compared: the sites differ, so the two
    means are over different populations.

    Parameters
    ----------
    complete_sites
        Restrict to sites covered in every sample, which is the default because
        entropy is a mean over sites and a mean over *different* sites is not a
        comparison. Set ``False`` to use each sample's own coverage, in which
        case read ``n_sites`` before comparing any two rows.

    Returns
    -------
    ``entropy`` and ``n_sites`` per sample. ``n_sites`` is not decoration: with
    ``complete_sites=False`` it is the column that says whether the comparison
    you are about to make is valid.
    """
    X = _beta_matrix(data, features, complete=complete_sites, what="entropy")
    b = np.clip(X.to_numpy(dtype=np.float64), _CLAMP, 1.0 - _CLAMP)
    seen = np.isfinite(b)

    terms = np.where(seen, b * np.log(b) + (1.0 - b) * np.log(1.0 - b), 0.0)
    n = seen.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = terms.sum(axis=1) / (n * np.log(0.5))
    s = np.where(n > 0, s, np.nan)

    return pd.DataFrame({"entropy": s, "n_sites": n}, index=X.index)


def drift(data, *, features: Sequence[str] | None = None,
          reference: pd.Series | None = None,
          statistic: str = "mean_abs") -> pd.DataFrame:
    """Per-sample distance from the cohort centroid: the drift score.

    The barometer below is a property of a group. This is its per-sample
    analogue, and it is the one that goes into a regression against an outcome,
    because a group-level statistic has no per-person value to associate.

    When ``reference`` is not supplied the centroid is computed **leave-one-out**
    -- each sample is scored against the mean of every *other* sample. Scoring a
    sample against a mean it helped compute shrinks its own distance, and in a
    small cohort that shrinkage is large enough to invert a group difference.

    Parameters
    ----------
    reference
        Per-site reference level, indexed by feature. Supply a young-cohort
        centroid to get drift-from-young, which is the quantity most of the
        drift literature means. Omit it for drift-from-this-cohort, which is
        the quantity that is defined without a second dataset.
    statistic
        ``"mean_abs"``
            Mean absolute deviation. Bounded 0 to 1, in beta units, and robust.
        ``"rmse"``
            Root mean squared deviation. Weights a few large departures more
            heavily, which is the right choice if the hypothesis is about
            focal change rather than diffuse change.

    Returns
    -------
    ``drift`` and ``n_sites`` per sample.
    """
    if statistic not in {"mean_abs", "rmse"}:
        raise AnalysisError(f"statistic must be 'mean_abs' or 'rmse', not {statistic!r}")

    X = _beta_matrix(data, features, complete=reference is None, what="drift")
    vals = X.to_numpy(dtype=np.float64)
    n_samples = vals.shape[0]

    if reference is not None:
        ref = reference.reindex(X.columns).to_numpy(dtype=np.float64)
        if not np.isfinite(ref).any():
            raise AnalysisError(
                "the reference shares no features with this matrix")
        centre = np.broadcast_to(ref, vals.shape)
    else:
        if n_samples < 3:
            raise AnalysisError(
                f"a leave-one-out centroid needs at least 3 samples, got {n_samples}.\n"
                "  With two samples each is scored against the other and the "
                "two drift scores are identical by construction.")
        total = vals.sum(axis=0)
        centre = (total - vals) / (n_samples - 1)

    d = vals - centre
    ok = np.isfinite(d)
    n = ok.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        if statistic == "mean_abs":
            score = np.where(ok, np.abs(d), 0.0).sum(axis=1) / n
        else:
            score = np.sqrt(np.where(ok, d ** 2, 0.0).sum(axis=1) / n)
    score = np.where(n > 0, score, np.nan)

    return pd.DataFrame({"drift": score, "n_sites": n}, index=X.index)


def variable_sites(data, *, age_col: str = "age", bins: int = 3,
                   min_per_bin: int = 5, alpha: float = 0.05,
                   features: Sequence[str] | None = None) -> pd.DataFrame:
    """Which cytosines become more *variable* with age, by Brown-Forsythe.

    The barometer is defined over "cytosines whose variance rises with age", and
    that clause is the whole statistic: summed over every site, the sum is
    dominated by sites that are simply variable, most of which have nothing to
    do with age. This function is the selection step, split out so it can be
    inspected rather than assumed.

    The test is Brown-Forsythe -- Levene's test using the median rather than the
    mean -- across age bins. Bartlett's test is the more common choice in this
    literature and is the wrong one here: it assumes normality within each
    group, and beta values near 0 or 1 are visibly not normal, so Bartlett
    reports variance differences that are distributional artefacts.

    A significant Brown-Forsythe says the variances differ between bins. It does
    **not** say variance *rises*, and a site whose variance falls with age is
    just as significant. ``direction`` carries the sign of the trend in
    per-bin standard deviation, and ``rising`` is the conjunction that the
    barometer actually wants.

    Parameters
    ----------
    bins
        Age bins, cut at equal-count quantiles. Three is the default because it
        keeps per-bin counts up while still admitting a monotone trend; more
        bins resolve the shape of the trend and cost power.

    Returns
    -------
    Per feature: ``statistic``, ``p``, ``q`` (Benjamini-Hochberg), ``direction``
    (+1 if the per-bin SD trends up with age, -1 down, 0 flat), ``rising``
    (``q <= alpha`` and ``direction > 0``), and the per-bin standard deviations.
    """
    from ..analysis import _bh  # BH is already implemented once; do not repeat it

    if age_col not in data.obs.columns:
        raise AnalysisError(f"no {age_col!r} column in obs")
    if bins < 2:
        raise AnalysisError(f"bins must be at least 2, got {bins}")

    X = _beta_matrix(data, features, complete=False, what="variable_sites")
    age = pd.to_numeric(data.obs[age_col], errors="coerce")
    usable = age.notna()
    if usable.sum() < bins * min_per_bin:
        raise AnalysisError(
            f"{usable.sum()} samples carry an age; {bins} bins of at least "
            f"{min_per_bin} need {bins * min_per_bin}.")

    X, age = X.loc[usable], age[usable]
    codes = pd.qcut(age, q=bins, labels=False, duplicates="drop")
    groups = [np.flatnonzero(codes.to_numpy() == k) for k in range(int(codes.max()) + 1)]
    groups = [g for g in groups if len(g) >= min_per_bin]
    if len(groups) < 2:
        raise AnalysisError(
            f"after binning, only {len(groups)} bin(s) reached min_per_bin="
            f"{min_per_bin}. Ages here may be too clustered to bin.")

    vals = X.to_numpy(dtype=np.float64)
    n_feat = vals.shape[1]
    stat = np.full(n_feat, np.nan)
    pval = np.full(n_feat, np.nan)
    sds = np.full((len(groups), n_feat), np.nan)

    for k, idx in enumerate(groups):
        sub = vals[idx]
        with np.errstate(invalid="ignore"):
            sds[k] = np.nanstd(sub, axis=0, ddof=1)

    # Brown-Forsythe per site. scipy's levene takes 1-D arrays, so this loops;
    # the loop is over sites and each call is over a few dozen values, which is
    # fast enough that vectorising it would trade clarity for nothing.
    for j in range(n_feat):
        parts = []
        for idx in groups:
            col = vals[idx, j]
            col = col[np.isfinite(col)]
            if col.size >= 2 and np.ptp(col) > 0:
                parts.append(col)
        if len(parts) < 2:
            continue
        try:
            s, p = stats.levene(*parts, center="median")
        except ValueError:  # pragma: no cover - degenerate input
            continue
        stat[j], pval[j] = s, p

    q = np.full(n_feat, np.nan)
    tested = np.isfinite(pval)
    if tested.any():
        q[tested] = _bh(pval[tested])

    # Direction from the rank correlation of per-bin SD against bin order, which
    # asks whether the SD trends rather than whether the endpoints differ.
    order = np.arange(len(groups), dtype=np.float64)
    direction = np.zeros(n_feat, dtype=np.int8)
    if len(groups) >= 3:
        for j in range(n_feat):
            col = sds[:, j]
            if np.isfinite(col).sum() >= 3 and np.ptp(col[np.isfinite(col)]) > 0:
                ok = np.isfinite(col)
                rho, _ = stats.spearmanr(order[ok], col[ok])
                if np.isfinite(rho) and rho != 0:
                    direction[j] = 1 if rho > 0 else -1
    else:
        with np.errstate(invalid="ignore"):
            delta = sds[-1] - sds[0]
        direction = np.where(np.isfinite(delta) & (delta > 0), 1,
                             np.where(np.isfinite(delta) & (delta < 0), -1, 0)).astype(np.int8)

    out = pd.DataFrame(
        {"statistic": stat, "p": pval, "q": q, "direction": direction,
         "rising": (q <= alpha) & (direction > 0)},
        index=X.columns)
    for k in range(len(groups)):
        out[f"sd_bin{k}"] = sds[k]
    return out


def noise_barometer(data, *, group: str | None = None,
                    sites: Sequence[str] | None = None,
                    age_col: str = "age", min_group: int = 5,
                    normalise: bool = True) -> pd.DataFrame:
    """Mei et al.'s noise barometer: summed per-site SD over age-variable sites.

    A property of a **group of samples**, not of a person. There is no
    per-sample barometer, because a standard deviation needs more than one
    observation; :func:`drift` is the per-sample quantity.

    Two things make a barometer comparable between groups, and both are easy to
    lose. The sites must be the same, which is why ``sites`` is computed once
    over the whole dataset rather than per group. And the group sizes must be
    close, because the sampling error of an SD scales with :math:`1/\\sqrt{n-1}`
    and a small group's summed SD is noisier, not larger in expectation. Group
    sizes are returned so the second can be checked rather than hoped for.

    Parameters
    ----------
    sites
        The cytosines to sum over. Omit to select them here with
        :func:`variable_sites`, which needs ``age_col``. Pass an explicit list
        to reuse a selection made on a different cohort, which is the only way
        the statistic transfers.
    normalise
        Return the *mean* per-site SD alongside the sum. The published statistic
        is the sum; the mean is what lets two selections of different size be
        compared, and both are returned because they answer different questions.

    Returns
    -------
    One row per group: ``barometer`` (the sum), ``mean_sd``, ``n_samples``,
    ``n_sites``.
    """
    X = _beta_matrix(data, None, complete=False, what="noise_barometer")

    if sites is None:
        table = variable_sites(data, age_col=age_col)
        chosen = list(table.index[table["rising"].to_numpy(dtype=bool)])
        if not chosen:
            raise AnalysisError(
                "no site's variance rose significantly with age, so the "
                "barometer has nothing to sum over.\n"
                "  That is a result, not a failure: inspect variable_sites() "
                "directly. Passing an explicit `sites=` list overrides this.")
    else:
        chosen = [s for s in sites if s in X.columns]
        if not chosen:
            raise AnalysisError(
                "none of the requested sites is in this matrix")

    X = X[chosen]
    if group is None:
        labels = pd.Series("all", index=X.index)
    else:
        if group not in data.obs.columns:
            raise AnalysisError(f"no {group!r} column in obs")
        labels = data.obs.loc[X.index, group].astype(str)

    rows = {}
    for name, idx in labels.groupby(labels).groups.items():
        sub = X.loc[idx]
        if len(sub) < min_group:
            continue
        with np.errstate(invalid="ignore"):
            sd = np.nanstd(sub.to_numpy(dtype=np.float64), axis=0, ddof=1)
        ok = np.isfinite(sd)
        rows[name] = {
            "barometer": float(sd[ok].sum()),
            "mean_sd": float(sd[ok].mean()) if ok.any() else np.nan,
            "n_samples": int(len(sub)),
            "n_sites": int(ok.sum()),
        }

    if not rows:
        raise AnalysisError(
            f"no group reached min_group={min_group} samples")

    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = group or "group"
    return out if normalise else out.drop(columns=["mean_sd"])
