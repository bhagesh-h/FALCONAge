"""How much of a score is measurement noise.

WHY THIS EXISTS. Every implementation of every aging clock, including this one
until now, returns a point estimate. The field's own literature says that number
is not interpretable on its own: technical replicates of the *same DNA* differ by
up to 9 years on six prominent clocks (Nat Aging 2022, s43587-022-00248-2), and
only 18% of CpGs on the 450K array reach ICC >= 0.5 in whole blood (Sugden 2020).
A reader given "DNAmAge = 54.2" has no way to know whether a 1.5-year change
between two draws is biology or the assay.

For a linear clock the answer is one line of algebra. With ``score = f(b + wᵀβ)``
and per-probe measurement variance ``σ²ⱼ``,

    Var(score) = f'(raw)² · Σⱼ wⱼ² σ²ⱼ

and ``σ²ⱼ`` follows from the probe's test-retest reliability and the spread of
that probe in the cohort at hand: ``σ²ⱼ = s²ⱼ (1 − ICCⱼ)``. Both pieces are
available -- the ICCs ship with the registry, the ``s²ⱼ`` come from the user's
own matrix -- so the interval costs one matrix-vector product.

TWO SOURCES, AND THEY ANSWER SLIGHTLY DIFFERENT QUESTIONS.

``"probe"``
    The expression above. Per sample as well as per clock, because a sample with
    more imputed features gets a wider interval -- an imputed probe carries no
    information about *this* sample, so it contributes its full between-sample
    variance rather than a reduced one. This is the more useful answer and the
    one to prefer.
``"clock"``
    ``SE = s · sqrt(1 − ICC)`` from the clock-level technical ICC in the
    registry, with ``s`` the SD of the scores in this cohort. One number per
    clock. Available for the eight clocks with a published figure, and it is a
    check on the probe path rather than a substitute: the two are derived from
    different measurements and should land in the same neighbourhood.

THE ASSUMPTION IN THAT SUM, STATED BECAUSE IT IS LOAD-BEARING. Writing
``Var(Σ wⱼ βⱼ) = Σ wⱼ² σ²ⱼ`` drops the cross term
``2 Σ_{j<k} wⱼ w_k Cov(εⱼ, ε_k)``: it treats the measurement errors on different
probes as independent. They are not. Chip position, plate, and scanner drift all
move many probes together, and those covariances are positive far more often than
negative. **So the SE reported here is a lower bound.** It is not a small
correction either -- a clock with hundreds of positively correlated terms can
have a true SE some multiple of this one.

The alternative is not available: estimating a 319,607 x 319,607 error covariance
would need replicate designs nobody publishes. What is available is the
comparison -- ``source="clock"`` derives the same quantity from a *published*
clock-level ICC measured on real technical replicates, which does capture the
correlated part. When the two disagree by a lot, the clock-level number is the
one to trust and the gap is the correlated component this path cannot see.
``icc_from_replicates`` closes it properly if you have run duplicates.

WHAT THIS IS NOT. It is measurement error, not prediction error. A clock can be
perfectly repeatable and still be a poor estimate of anything -- see
:func:`falconage.uncertainty.conformal_interval` for the other question. And it
says nothing about biological variability: the same person sampled a fortnight
later is a different measurement of a different thing.
"""

from __future__ import annotations

import functools
import gzip
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..core.errors import FalconError
from ..models import ops
from ..registry.registry import DATA_DIR

from .variance import (VarianceComponents, VarianceError,  # noqa: E402
                       variance_components)

__all__ = [
    "SEResult", "VarianceComponents", "VarianceError", "conformal_interval",
    "icc_from_replicates", "interval", "load_conformal", "load_probe_icc",
    "probe_icc_source", "technical_se", "variance_components",
]

ICC_FILE = DATA_DIR / "probe_icc.csv.gz"
CONFORMAL_FILE = DATA_DIR / "conformal.csv"


class UncertaintyError(FalconError):
    """Raised when an interval cannot be computed honestly."""


@functools.lru_cache(maxsize=1)
def load_probe_icc() -> pd.Series:
    """Per-probe test-retest ICC, indexed by feature id.

    Cached for the process. The file is 1.7 MB gzipped and about 284,000 rows;
    parsing it per call would dominate a scoring run.
    """
    if not ICC_FILE.exists():
        raise UncertaintyError(
            f"the bundled reliability table is missing from {ICC_FILE}.\n"
            "  Rebuild it with python/tools/build_probe_icc.py, or pass your own "
            "with technical_se(icc=<Series>).")
    df = pd.read_csv(ICC_FILE, comment="#")
    return pd.Series(df["icc"].to_numpy(dtype=np.float64),
                     index=df["feature_id"].astype(str), name="icc")


@functools.lru_cache(maxsize=1)
def probe_icc_source() -> dict[str, str]:
    """Provenance of the bundled table, for the run manifest.

    An interval whose source is not recorded is worse than no interval: a reader
    cannot tell whether it came from a published replicate study or from a
    default somebody typed.
    """
    out: dict[str, str] = {}
    if not ICC_FILE.exists():
        return out
    import hashlib

    with gzip.open(ICC_FILE, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            body = line[1:].strip()
            key, _, val = body.partition(": ")
            if val:
                out[key] = val
            elif "citation" not in out:
                out["citation"] = body      # the first bare line
            else:
                out["method"] = body        # and the second, which is the ICC model
    out["sha256"] = hashlib.sha256(ICC_FILE.read_bytes()).hexdigest()
    return out


def icc_from_replicates(data, subject_col: str, *,
                        features: Sequence[str] | None = None) -> pd.Series:
    """Per-probe ICC(1,1) computed from the user's own technical replicates.

    For labs that ran duplicates. A one-way random-effects, single-measurement
    ICC is the right model when the "raters" are interchangeable array positions
    rather than named assessors:

        ICC(1,1) = (MSB - MSW) / (MSB + (k-1)·MSW)

    with ``k`` the average replicates per subject. Negative values are kept, not
    clipped to zero: a negative ICC means the within-subject spread exceeded the
    between-subject spread, which is a real and reportable state of affairs for
    a probe that measures nothing.

    Preferred over the bundled table whenever it is available, because it is
    *this* laboratory's noise on *this* platform rather than a published cohort's.
    """
    if subject_col not in data.obs.columns:
        raise UncertaintyError(f"no {subject_col!r} column in obs")
    groups = data.obs[subject_col].astype(str)
    counts = groups.value_counts()
    if (counts > 1).sum() < 3:
        raise UncertaintyError(
            f"only {(counts > 1).sum()} subject(s) have more than one sample; "
            "an ICC from fewer than three replicated subjects is not an estimate")

    X = data.X if features is None else data.X.reindex(columns=list(features))
    vals = X.to_numpy(dtype=np.float64)
    codes, uniq = pd.factorize(groups)
    n_g = len(uniq)
    n = len(codes)

    grand = np.nanmean(vals, axis=0)
    sums = np.zeros((n_g, vals.shape[1]))
    cnts = np.zeros((n_g, 1))
    np.add.at(sums, codes, np.nan_to_num(vals))
    np.add.at(cnts, codes, 1.0)
    means = sums / np.maximum(cnts, 1)

    ssb = (cnts * (means - grand) ** 2).sum(axis=0)
    ssw = ((vals - means[codes]) ** 2).sum(axis=0)
    df_b, df_w = n_g - 1, n - n_g
    if df_b < 1 or df_w < 1:
        raise UncertaintyError("not enough degrees of freedom for an ICC")
    msb, msw = ssb / df_b, ssw / df_w
    k = n / n_g

    with np.errstate(invalid="ignore", divide="ignore"):
        icc = (msb - msw) / (msb + (k - 1.0) * msw)
    return pd.Series(icc, index=X.columns, name="icc")


class SEResult:
    """Standard errors, plus how much of each was guessed.

    ``se`` is samples x clocks in each clock's own unit. ``diagnostics`` says,
    per clock, how many of its features had a published ICC and how many fell
    back to the cohort median -- the number that decides whether the interval is
    a measurement or an indication.
    """

    def __init__(self, se: pd.DataFrame, diagnostics: pd.DataFrame,
                 refused: dict[str, str], source: dict[str, Any]):
        self.se, self.diagnostics = se, diagnostics
        self.refused, self.source = refused, source

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (f"SEResult({self.se.shape[0]} samples x {self.se.shape[1]} clocks, "
                f"{len(self.refused)} refused)")


def technical_se(result, data=None, *, source: str = "auto",
                 icc: pd.Series | None = None, registry=None) -> SEResult:
    """Standard error on each score attributable to assay measurement noise.

    Parameters
    ----------
    result
        A :class:`~falconage.score.FalconResult`.
    data
        The :class:`~falconage.core.FalconData` that was scored. Required for
        ``source="probe"``; the per-feature spread has to come from the matrix,
        not from the scores.
    source
        ``"probe"``, ``"clock"``, or ``"auto"`` (probe where possible, clock
        otherwise, and a recorded refusal where neither is available).
    icc
        Per-feature ICC, overriding the bundled table. Pass the output of
        :func:`icc_from_replicates` to use your own duplicates.

    Notes
    -----
    An imputed feature contributes its **full** between-sample variance rather
    than the reduced ``s²(1-ICC)``. This is deliberate and it is the part of the
    calculation most likely to look wrong: an imputed value is the cohort mean
    or a published reference, so it carries no information about the sample in
    front of you, and treating it as a well-measured probe would make the
    interval *narrower* for worse data.
    """
    if source not in ("auto", "probe", "clock"):
        raise UncertaintyError(f"source={source!r}; expected auto, probe or clock")
    reg = registry if registry is not None else result.registry

    table = icc if icc is not None else None
    if table is None and source in ("auto", "probe") and data is not None:
        try:
            table = load_probe_icc()
        except UncertaintyError:
            table = None

    se: dict[str, np.ndarray] = {}
    diag: list[dict[str, Any]] = []
    refused: dict[str, str] = {}

    for cid in result.scores.columns:
        clock = reg.get(cid)
        want_probe = source in ("auto", "probe") and data is not None and table is not None
        if want_probe:
            try:
                col, d = _probe_se(result, data, reg, cid, table)
                se[cid] = col
                diag.append(d)
                continue
            except UncertaintyError as exc:
                if source == "probe":
                    refused[cid] = str(exc).splitlines()[0]
                    continue
        if source in ("auto", "clock"):
            try:
                col, d = _clock_se(result, clock, cid)
                se[cid] = col
                diag.append(d)
                continue
            except UncertaintyError as exc:
                refused[cid] = str(exc).splitlines()[0]
                continue
        refused[cid] = "no usable reliability source"

    frame = pd.DataFrame(se, index=result.scores.index)
    dg = pd.DataFrame(diag).set_index("clock") if diag else pd.DataFrame()

    # The clock-level ICC this cohort implies, which is the number the
    # literature quotes and the one a reader can compare against a paper:
    # ICC = 1 - var_technical / var_total. Reported as measured here rather than
    # written into the registry, because it is a property of this dataset --
    # a cohort with a narrow age range has less between-sample variance and will
    # honestly report a lower ICC for the same assay.
    if not dg.empty:
        implied, sds, ratios = [], [], []
        for cid in dg.index:
            tot = float(result.scores[cid].var(ddof=1))
            tech = float(np.mean(frame[cid].to_numpy() ** 2))
            implied.append(round(1.0 - tech / tot, 4) if tot > 0 else np.nan)
            sds.append(round(float(np.sqrt(tot)), 4))
            # Unit-free, so clocks measured in years, kilobases and pace ratios
            # can be compared on one axis. Plotting the raw SE side by side
            # would be the units error LEGAL_OPS exists to prevent, committed in
            # a figure instead of in arithmetic.
            ratios.append(round(float(np.sqrt(tech / tot)), 4) if tot > 0 else np.nan)
        dg["cohort_sd"] = sds
        dg["se_over_sd"] = ratios
        dg["implied_cohort_icc"] = implied

    src = probe_icc_source() if icc is None else {"source": "user-supplied"}
    # Cached on the result so summary(), the report and the plots can reach it
    # without being handed the matrix a second time, and recorded in the
    # manifest so an interval always travels with the table it came from.
    result.se = frame
    result.manifest.config["technical_se"] = {
        "source": source, "reliability": src,
        "refused": {k: v for k, v in refused.items()},
    }
    return SEResult(frame, dg, refused, src)


def _probe_se(result, data, reg, cid: str, table: pd.Series):
    """The Σ wⱼ² σ²ⱼ path. Returns (per-sample SE, diagnostics row)."""
    from ..models.linear import align

    if not reg.has_coefficients(cid):
        raise UncertaintyError(f"{cid}: no coefficients, so there are no weights to square")
    if not reg.has_coefficient_vector(cid):
        raise UncertaintyError(
            f"{cid} is a network: the probe path propagates each probe's noise "
            "through its own weight, and a network has no per-probe weight. Its "
            "interval has to come from a published clock-level reliability "
            "figure or from replicates.")
    if reg.get(cid).preprocess:
        raise UncertaintyError(
            f"{cid}: preprocess chains are not yet supported on the probe path")

    feats, coefs = reg.coefficients(cid)
    feats = list(feats)
    w = np.asarray(coefs, dtype=np.float64)

    al = align(data, feats, imputation=result.manifest.config.get("imputation", "reference"),
               coefficients=w)
    x = al.matrix                                  # samples x features, imputed
    observed = data.X.reindex(columns=feats).to_numpy(dtype=np.float64)
    imputed = np.isnan(observed)

    # Between-sample variance per feature, from the observed values only. A
    # feature the cohort does not carry has no spread to measure, so it falls
    # back to the array-wide median spread rather than to zero -- zero would
    # make a wholly absent probe look perfectly measured.
    # A feature with one observed value has no variance to estimate, and numpy
    # says so once per such column. On a 319,607-probe BLUP clock that is tens
    # of thousands of identical lines burying the warnings that matter. The
    # fallback below is what handles the case; the message adds nothing.
    import warnings as _w

    with np.errstate(invalid="ignore"), _w.catch_warnings():
        _w.simplefilter("ignore", RuntimeWarning)
        s2 = np.nanvar(observed, axis=0, ddof=1)
    fallback_s2 = float(np.nanmedian(s2)) if np.isfinite(s2).any() else 0.0
    s2 = np.where(np.isfinite(s2), s2, fallback_s2)

    icc = table.reindex(feats).to_numpy(dtype=np.float64)
    n_known = int(np.isfinite(icc).sum())
    median_icc = float(np.nanmedian(icc)) if n_known else 0.0
    icc = np.where(np.isfinite(icc), icc, median_icc)
    icc = np.clip(icc, 0.0, 1.0)

    # Present features: the reduced variance. Imputed: the full variance.
    var_present = s2 * (1.0 - icc)
    per_cell = np.where(imputed, s2[None, :], var_present[None, :])

    raw_var = (per_cell * (w ** 2)[None, :]).sum(axis=1)

    # Delta method through the postprocess chain, evaluated at each sample's
    # own raw score -- anti_log_linear's slope differs by a factor of e^x below
    # zero, so a single slope for the cohort would be wrong for anyone young.
    raw = x @ w
    _, slope = ops.chain_derivative(raw, reg.get(cid).postprocess, ops.POSTPROCESS)
    se = np.sqrt(np.maximum(raw_var, 0.0)) * np.abs(np.asarray(slope, dtype=np.float64))

    return se, {
        "clock": cid, "method": "probe",
        "n_features": len(feats),
        "n_icc_published": n_known,
        "n_icc_imputed": len(feats) - n_known,
        "median_icc": round(float(np.median(icc)), 4),
        "n_features_imputed_mean": round(float(imputed.sum(axis=1).mean()), 2),
        "median_se": round(float(np.median(se)), 4),
    }


def _clock_se(result, clock, cid: str):
    """``SE = s·sqrt(1-ICC)`` from the clock-level published reliability."""
    r = clock.reliability
    if r.technical_icc is None:
        raise UncertaintyError(
            f"{cid}: no published technical ICC in the registry, and no per-probe "
            "table was usable. An interval would be invented rather than measured.")
    s = float(result.scores[cid].std(ddof=1))
    icc = float(np.clip(r.technical_icc, 0.0, 1.0))
    val = s * np.sqrt(max(1.0 - icc, 0.0))
    n = result.scores.shape[0]
    return np.full(n, val), {
        "clock": cid, "method": "clock",
        "n_features": clock.n_features,
        "n_icc_published": 1, "n_icc_imputed": 0,
        "median_icc": round(icc, 4),
        "n_features_imputed_mean": np.nan,
        "median_se": round(float(val), 4),
    }


@functools.lru_cache(maxsize=1)
def load_conformal() -> pd.DataFrame:
    """Calibrated prediction-interval half-widths, one row per clock and level.

    Built by ``python/tools/build_conformal.py`` from healthy-control blood
    samples in the test corpus. Empty when the table has not been derived.
    """
    if not CONFORMAL_FILE.exists():
        return pd.DataFrame(columns=["clock", "age_band", "level", "half_width",
                                     "median_bias", "mae", "bias_within_interval",
                                     "n_calibration", "exact"])
    return pd.read_csv(CONFORMAL_FILE, comment="#")


def conformal_interval(result, *, level: float = 0.90,
                       clocks: Sequence[str] | None = None) -> pd.DataFrame:
    """How far the prediction is likely to be from chronological age.

    A different question from :func:`technical_se`, with a larger answer.
    Technical error asks how much the *assay* would move the number on a repeat;
    this asks how wrong the number is likely to be, which includes everything
    the clock never learned.

    Split conformal: the half-width is a quantile of the absolute residual on a
    calibration set of healthy blood samples with known ages, so on any sample
    exchangeable with that cohort the interval contains the truth at the stated
    rate. No distribution is assumed and the guarantee is finite-sample.

    Two things it will tell you that are easy to miss:

    * ``median_bias`` -- a clock systematically above or below chronological age
      on the calibration cohort. Ying's DamAge and AdaptAge are tens of years
      off because they are causality-partitioned components on an age-like
      scale, not age predictors.
    * ``exchangeable`` -- always ``False`` here, and deliberately. The coverage
      guarantee is conditional on the new samples being drawn like the
      calibration ones: adult, blood, overwhelmingly European ancestry. Nothing
      in this function can verify that, so it declines to imply it.
    """
    tab = load_conformal()
    if tab.empty:
        raise UncertaintyError(
            "no conformal calibration is bundled.\n"
            "  Build it with python/tools/build_conformal.py, which needs the "
            "test corpus (see test/data/README.md).")
    use = list(clocks) if clocks else list(result.scores.columns)
    rows = []
    for cid in use:
        m = tab[(tab["clock"] == cid) & (tab["age_band"] == "all")
                & (np.isclose(tab["level"], level))]
        if m.empty:
            continue
        r = m.iloc[0]
        for sid in result.scores.index:
            v = float(result.scores.at[sid, cid])
            rows.append({
                "sample_id": sid, "clock": cid, "value": v,
                "lo": v - float(r["half_width"]), "hi": v + float(r["half_width"]),
                "half_width": float(r["half_width"]), "level": level,
                "median_bias": float(r["median_bias"]), "mae": float(r["mae"]),
                "bias_within_interval": bool(r["bias_within_interval"]),
                "n_calibration": int(r["n_calibration"]),
                "exchangeable": False,
            })
    if not rows:
        raise UncertaintyError(
            f"none of {use[:4]} has a conformal calibration at level {level}.\n"
            "  Only clocks whose scale is age in years are calibrated: a band "
            "on a log-hazard is not an age interval, and quoting one in years "
            "would invent the units.")
    return pd.DataFrame(rows)


def interval(result, data=None, *, level: float = 0.95, **kw) -> pd.DataFrame:
    """Score, lower and upper bound, long form.

    A normal interval on the score scale. Justified by the propagation being a
    sum of many small independent contributions, which is where the central
    limit theorem is at its most comfortable; the exception is a clock with a
    handful of features, where the interval should be read as indicative.
    """
    from scipy.stats import norm

    z = float(norm.ppf(0.5 + level / 2.0))
    res = technical_se(result, data, **kw)
    rows = []
    for cid in res.se.columns:
        for sid in result.scores.index:
            v, s = float(result.scores.at[sid, cid]), float(res.se.at[sid, cid])
            rows.append({"sample_id": sid, "clock": cid, "value": v,
                         "se": s, "lo": v - z * s, "hi": v + z * s,
                         "level": level})
    out = pd.DataFrame(rows)
    out.attrs["refused"] = res.refused
    out.attrs["source"] = res.source
    return out
