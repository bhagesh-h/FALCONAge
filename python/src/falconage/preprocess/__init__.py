"""Turn raw or public data into something a clock can be scored on."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.container import FalconData
from ..core.units import MARKERS, canonical_name, check_plausible, convert, require_units
from .batch import BatchError, BatchReference, apply_batch_reference, fit_batch_reference
from .bmiq import BetaMixture, bmiq, fit_beta_mixture
from .idat import RawSignal, dye_bias, idat_to_betas, noob, poobah, read_idat_dir
from .manifest import fetch_manifest, load_manifest, manifest_record
from .masks import apply_mask, load_mask, mask_report, masked_probes
from .methylation import (
    QCReport,
    aggregate_replicate_probes,
    clip_betas,
    ensure_platform,
    harmonise_probe_ids,
    prepare,
    qc,
)
from .proteomic import prepare_proteomic, read_olink, read_somascan
from .transcriptomic import (median_centre, prepare_transcriptomic, read_counts,
                             rle_normalise, yugene)

__all__ = [
    "BatchError", "BatchReference", "BetaMixture", "QCReport", "RawSignal",
    "bmiq", "fit_beta_mixture",
    "aggregate_replicate_probes",
    "apply_batch_reference", "apply_mask", "clip_betas", "dye_bias", "ensure_platform",
    "load_mask", "mask_report", "masked_probes",
    "fetch_manifest", "fit_batch_reference",
    "harmonise_probe_ids", "idat_to_betas", "impute", "load_manifest",
    "manifest_record", "noob", "poobah",
    "prepare", "prepare_clinical", "probe_loss",
    "median_centre", "prepare_proteomic", "prepare_transcriptomic",
    "qc", "read_counts", "read_idat_dir", "read_olink", "read_somascan",
    "rle_normalise", "yugene",
]


def prepare_clinical(data: FalconData, units: dict[str, str] | None = None,
                     *, target: dict[str, str] | None = None) -> FalconData:
    """Rename to canonical markers and convert to the units the models expect.

    Raises :class:`~falconage.core.errors.UnitsNotDeclaredError` when ``units``
    is missing or incomplete, with the dict to supply. That refusal is the point
    of the module -- see :mod:`falconage.core.units`.
    """
    from ..models.clinical import PHENOAGE_UNITS

    tgt = target or PHENOAGE_UNITS
    declared = require_units(units, list(data.X.columns))

    cols: dict[str, np.ndarray] = {}
    notes: list[str] = []
    for col in data.X.columns:
        key = str(col).strip().lower().replace(" ", "_").replace("-", "_")
        try:
            name = canonical_name(key)
        except Exception:
            continue                       # not a recognised marker; pass through below
        want = tgt.get(name, MARKERS[name].canonical)
        have = declared.get(name, want)
        v = convert(data.X[col].to_numpy(dtype=np.float64), have, want)
        cols[name] = v
        if have != want:
            notes.append(f"{name}: {have} -> {want}")
        w = check_plausible(name, v)
        if w:
            notes.append("WARNING " + w)

    if not cols:
        from ..core.errors import DataError

        raise DataError(
            "no recognised clinical markers in " + ", ".join(map(str, data.X.columns[:8]))
            + f"\n  known: {', '.join(sorted(MARKERS))}"
        )

    X = pd.DataFrame(cols, index=data.X.index)
    # Age lives in obs as well as X: the models take it as a term, the analyses
    # take it as a covariate, and having it in one place only guarantees one of
    # them goes looking in the wrong place.
    obs = data.obs.copy()
    if "age" in X.columns and "age" not in obs.columns:
        obs["age"] = X["age"]

    out = FalconData(X=X, obs=obs, modality="clinical_chemistry",
                     units={k: tgt.get(k, MARKERS[k].canonical) for k in cols},
                     uns=dict(data.uns))
    out.uns["unit_conversions"] = notes
    return out


def impute(data: FalconData, how: str = "median") -> FalconData:
    """Dataset-level imputation, applied before any clock sees the matrix.

    This is the coarsest of the three imputation stages and the one to reach
    for last. Clock-level imputation (in :func:`falconage.models.linear.align`)
    knows which features a given clock needs and can use the values that
    clock's authors published; this one only knows the column.

    ``how="none"`` is a no-op, and is the right choice when you want the
    coverage check to fail rather than to be papered over.
    """
    if how == "none":
        return data
    X = data.X
    if how == "median":
        filled = X.fillna(X.median(axis=0))
    elif how == "mean":
        filled = X.fillna(X.mean(axis=0))
    else:
        raise ValueError("how must be 'median', 'mean' or 'none'")
    # A column that is entirely NaN has no median; leave it NaN so coverage
    # counts it absent rather than filling the whole matrix with one number.
    out = FalconData(X=filled, obs=data.obs, modality=data.modality, units=data.units,
                     platform=data.platform, uns=dict(data.uns))
    out.uns["dataset_imputation"] = how
    return out


def _load_platform_bias() -> dict[tuple[str, str], dict]:
    """The measured cost of probe loss, keyed by (clock, platform).

    Cached for the process. Empty when the table has not been derived, which is
    the honest state for a build that has never seen the corpus -- an absent row
    means "not measured", never "no bias".
    """
    global _PLATFORM_BIAS
    if _PLATFORM_BIAS is None:
        from ..registry.registry import DATA_DIR

        p = DATA_DIR / "platform_bias.csv"
        if not p.exists():
            _PLATFORM_BIAS = {}
        else:
            df = pd.read_csv(p, comment="#")
            _PLATFORM_BIAS = {
                (str(r.clock), str(r.platform)): {
                    "median_shift": float(r.median_shift),
                    "ci_lo": float(r.ci_lo), "ci_hi": float(r.ci_hi),
                    "probes_retained": int(r.probes_retained),
                    "probes_total": int(r.probes_total),
                    "unit": str(r.unit),
                }
                for r in df.itertuples()
            }
    return _PLATFORM_BIAS


_PLATFORM_BIAS: dict[tuple[str, str], dict] | None = None

#: Above this, in the clock's own unit, the shift is worth interrupting a run
#: for. One year on an age clock is the smallest difference anybody reports.
BIAS_WARN = 1.0


def probe_loss(data: FalconData, clocks: str | list[str] = "all",
               *, registry=None, top: int = 3) -> pd.DataFrame:
    """What each clock has lost on this dataset, before scoring anything.

    One row per clock: how many of its features are present, and -- for the
    clocks whose coefficients are available -- how much of the model's total
    weight those present features carry, plus the heaviest probes that are
    missing.

    WHY BOTH NUMBERS. A count treats every probe as interchangeable and an
    elastic-net's weights are nothing like uniform, so "92% of probes present"
    covers both "the 8% missing are negligible" and "the 8% missing carry a
    third of the model". EPIC v2 dropped probes that several first-generation
    clocks lean on, which is why those clocks shift on v2 arrays while the
    principal-component versions barely move (Life Science Alliance
    2025;8:e202403155) -- the same probe loss, very different consequences.

    Run this before ``score``, on an array you have not used before. It costs
    one alignment per clock and answers "will this dataset support these
    clocks" without producing a number anyone can quote.

    Parameters
    ----------
    clocks
        ``"all"``, ``"scoreable"`` for the ones whose coefficients are
        available, or an explicit list.
    top
        How many of the heaviest absent features to name per clock.

    Returns
    -------
    One row per clock, worst mass coverage first. ``mass_coverage`` is ``NaN``
    for a clock whose coefficients are not available -- the weights are what
    the column is computed from, so there is no honest value without them.
    """
    from ..models.linear import align
    from ..registry import load as _load

    reg = registry if registry is not None else _load()

    if clocks == "all":
        chosen = [c.id for c in reg if c.data_type == data.modality]
    elif clocks == "scoreable":
        chosen = [c.id for c in reg
                  if c.data_type == data.modality and reg.has_coefficients(c.id)]
    else:
        chosen = list(clocks)

    rows = []
    for cid in chosen:
        c = reg.get(cid)
        if c.formula:
            continue
        try:
            feats, coefs = reg.coefficients(cid)
        except Exception:
            # Tier B and C: the feature list itself is not available, so there
            # is nothing to align against. Say so rather than omitting the row,
            # because a clock silently missing from this table reads as "fine".
            rows.append({"clock": cid, "tier": c.availability,
                         "n_features": c.n_features, "n_present": None,
                         "coverage": np.nan, "mass_coverage": np.nan,
                         "heaviest_absent": "coefficients not available"})
            continue

        al = align(data, list(feats), imputation="none", coefficients=coefs)
        bias = _load_platform_bias().get((cid, data.platform or ""), {})
        rows.append({
            "clock": cid,
            "tier": c.availability,
            "n_features": len(feats),
            "n_present": int(al.present.sum()),
            "coverage": round(al.coverage, 4),
            "mass_coverage": (np.nan if al.mass_coverage is None
                              else round(al.mass_coverage, 4)),
            "bias_years": bias.get("median_shift", np.nan),
            "bias_ci": (f"{bias['ci_lo']:g} to {bias['ci_hi']:g}"
                        if bias else ""),
            "heaviest_absent": ", ".join(
                f"{f} ({s:.1%})" for f, s in al.missing_mass[:top]) or "",
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Worst first, and by weight rather than by count -- the ordering that
    # matches which clock this dataset actually damages most.
    return df.sort_values(["mass_coverage", "coverage"],
                          na_position="last").set_index("clock")
