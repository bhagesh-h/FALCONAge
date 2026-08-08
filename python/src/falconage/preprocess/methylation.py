"""Methylation preprocessing: get a matrix into the shape every clock assumes.

The clocks assume beta values in [0, 1], keyed by bare ``cg`` identifiers, with
the platform known. Real data violates all three, and each violation has a
different failure mode:

* **EPIC v2 suffixes** make every clock see zero overlap and score entirely
  imputed values. Loud when you look at the coverage report, invisible when you
  do not.
* **Cross-reactive and SNP-overlapping probes** move a beta by up to 0.3 for
  reasons that have nothing to do with methylation, and several of them are in
  clock feature lists.
* **Undetected platform** means the coverage warning cannot say what it should
  have been.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..core.container import FalconData
from ..core.errors import PlatformError

EPICV2_SUFFIX = re.compile(r"^(cg\d+|ch\.\d+\.\d+[FR]?)_[A-Z]{2}\d+$")


def aggregate_replicate_probes(data: FalconData, how: str = "mean") -> FalconData:
    """Strip EPIC v2 probe suffixes and collapse the replicates.

    EPIC v2 renamed ``cg00000029`` to ``cg00000029_TC21`` and, for about 5,000
    loci, put two or three differently-suffixed probes on the array. Every clock
    in the registry matches on the bare identifier, so without this step an
    EPIC v2 dataset has zero features in common with any of them -- and because
    the imputation step then fills all of them, the clock returns a number
    rather than an error.

    ``how="mean"`` averages replicates, which is what Illumina's own
    recommendation and every published EPIC v2 harmonisation do. ``how="first"``
    keeps the lowest-suffix probe, which is reproducible but arbitrary.
    """
    cols = [str(c) for c in data.X.columns]
    if not any(EPICV2_SUFFIX.match(c) for c in cols[:5000]):
        return data

    base = [EPICV2_SUFFIX.match(c).group(1) if EPICV2_SUFFIX.match(c) else c for c in cols]
    X = data.X.copy()
    X.columns = base
    n_before = X.shape[1]
    if how == "mean":
        X = X.T.groupby(level=0).mean().T
    elif how == "first":
        X = X.loc[:, ~pd.Index(base).duplicated()]
    else:
        raise ValueError("how must be 'mean' or 'first'")

    out = FalconData(X=X, obs=data.obs, modality=data.modality, units=data.units,
                     platform=data.platform or "EPICv2", uns=dict(data.uns))
    out.uns["epicv2_aggregation"] = {
        "how": how, "features_before": n_before, "features_after": X.shape[1],
        "collapsed": n_before - X.shape[1],
    }
    return out


def harmonise_probe_ids(data: FalconData) -> FalconData:
    """Strip whitespace and quoting, drop control and rs probes.

    ``rs`` probes are genotyping controls, not methylation, and a few of them
    sit inside clock feature lists in packages that never filtered them.
    """
    X = data.X.copy()
    X.columns = [str(c).strip().strip('"') for c in X.columns]
    keep = [c for c in X.columns if not str(c).startswith("rs")]
    dropped = X.shape[1] - len(keep)
    X = X[keep]
    out = FalconData(X=X, obs=data.obs, modality=data.modality, units=data.units,
                     platform=data.platform, uns=dict(data.uns))
    if dropped:
        out.uns["dropped_rs_probes"] = dropped
    return out


def ensure_platform(data: FalconData) -> FalconData:
    from ..io.methylation import detect_platform

    if data.platform:
        return data
    plat = detect_platform(data.X.columns)
    if plat is None:
        raise PlatformError(
            f"cannot identify the platform from {data.n_features} features.\n"
            "  27K, 450K and EPIC v1 share a probe namespace and are told apart "
            "by count alone, so a filtered matrix falls between the windows. Set "
            "data.platform explicitly -- it is used for the coverage warning, not "
            "for the arithmetic, so an approximate answer is fine."
        )
    data.platform = plat
    return data


def clip_betas(data: FalconData, lo: float = 0.0, hi: float = 1.0) -> FalconData:
    """Clip into the beta range, and count what needed clipping.

    Normalised matrices routinely contain a handful of values a hair outside
    [0, 1] from floating-point error. Many outside it means the matrix is not
    betas, which :func:`falconage.io.methylation.read_betas` already refuses.
    """
    X = data.X
    n_out = int(((X < lo) | (X > hi)).to_numpy().sum())
    out = FalconData(X=X.clip(lo, hi), obs=data.obs, modality=data.modality,
                     units=data.units, platform=data.platform, uns=dict(data.uns))
    if n_out:
        out.uns["clipped_values"] = n_out
    return out


@dataclass
class QCReport:
    """Per-sample and per-feature quality, computed before any scoring."""

    per_sample: pd.DataFrame
    per_feature: pd.DataFrame
    platform: str | None
    n_samples: int
    n_features: int
    warnings: list[str]

    def summary(self) -> pd.Series:
        return pd.Series({
            "platform": self.platform or "unknown",
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "median_sample_missingness": float(self.per_sample["missing_fraction"].median()),
            "features_all_missing": int((self.per_feature["missing_fraction"] == 1.0).sum()),
            "warnings": len(self.warnings),
        })

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"QCReport({self.n_samples} samples, {len(self.warnings)} warning(s))"


def qc(data: FalconData, *, sample_missing_threshold: float = 0.1) -> QCReport:
    """Look at the data before scoring it.

    Reports rather than fixes. A sample that is 40% missing may be a failed
    array or may be a 27K matrix aligned against an EPIC feature space, and the
    right response differs -- so this says what it sees and leaves the decision
    where it belongs.
    """
    X = data.X
    miss_s = X.isna().mean(axis=1)
    miss_f = X.isna().mean(axis=0)

    per_sample = pd.DataFrame({
        "missing_fraction": miss_s,
        "mean_beta": X.mean(axis=1),
        "sd_beta": X.std(axis=1),
    })
    per_feature = pd.DataFrame({
        "missing_fraction": miss_f,
        "mean_beta": X.mean(axis=0),
    })

    warnings: list[str] = []
    bad = per_sample.index[miss_s > sample_missing_threshold]
    if len(bad):
        warnings.append(
            f"{len(bad)} sample(s) above {sample_missing_threshold:.0%} missing: "
            + ", ".join(map(str, bad[:5])) + ("..." if len(bad) > 5 else ""))

    dead = int((miss_f == 1.0).sum())
    if dead:
        warnings.append(f"{dead} feature(s) are missing in every sample; they count "
                        "as absent for coverage, not as present-and-NaN")

    if data.modality == "dna_methylation":
        mean_all = float(np.nanmean(X.to_numpy(dtype=np.float64)))
        if not (0.3 < mean_all < 0.7):
            warnings.append(
                f"mean beta across the matrix is {mean_all:.3f}; a whole-array mean "
                "outside 0.3-0.7 usually means a normalisation problem or a "
                "non-blood tissue, and every clock's intercept assumes otherwise")

    if "sex" in data.obs.columns:
        warnings.extend(_sex_check(data))

    return QCReport(per_sample, per_feature, data.platform, data.n_samples,
                    data.n_features, warnings)


#: A handful of X-linked probes whose beta separates the sexes cleanly on every
#: Illumina platform. Not a clock -- a plausibility check on the sample sheet,
#: which is the single most common metadata error in public series.
_X_PROBES = ["cg12653510", "cg05533223", "cg03691818", "cg26355737", "cg09516963"]


def _sex_check(data: FalconData) -> list[str]:
    probes = [p for p in _X_PROBES if p in data.X.columns]
    if len(probes) < 3:
        return []
    score = data.X[probes].mean(axis=1)
    declared = data.obs["sex"].astype(str).str.upper().str[0]
    if declared.isin(["M", "F"]).sum() < 4:
        return []
    mf = score[declared == "M"].median(), score[declared == "F"].median()
    if not np.isfinite(mf).all() or abs(mf[0] - mf[1]) < 0.05:
        return ["sex check inconclusive: the X-linked probes do not separate the "
                "declared groups, so either the labels or the probes are unusable here"]
    cut = (mf[0] + mf[1]) / 2.0
    predicted = np.where((score < cut) == (mf[0] < mf[1]), "M", "F")
    known = declared.isin(["M", "F"]).to_numpy()
    mismatch = int(((predicted != declared.to_numpy()) & known).sum())
    if mismatch:
        return [f"{mismatch} sample(s) disagree with the declared sex on X-linked "
                "probe methylation; check the sample sheet before trusting any "
                "sex-stratified result"]
    return []


def prepare(data: FalconData, *, aggregate_epicv2: bool = True,
            clip: bool = True) -> FalconData:
    """The standard chain: harmonise ids, collapse v2 replicates, clip, identify."""
    out = harmonise_probe_ids(data)
    if aggregate_epicv2:
        out = aggregate_replicate_probes(out)
    if clip:
        out = clip_betas(out)
    try:
        out = ensure_platform(out)
    except PlatformError:
        pass  # a warning at score time is better than a hard stop here
    return out
