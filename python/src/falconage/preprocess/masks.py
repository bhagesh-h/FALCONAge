"""Probe masks, and the question nobody asks before applying one.

WHAT A MASK IS. Some Infinium probes do not measure what their name says: they
map to more than one place in the genome, they overlap a common SNP at the
interrogated base, they sit in a repeat, or their extension base is polymorphic.
Zhou, Laird and Shen catalogued all of it and publish a recommended general
mask per platform (Nucleic Acids Res 2017;45:e22). On HM450 it flags 29,504
probes -- six percent of the array.

THE QUESTION. Masking is standard and correct for an EWAS: a hit at a probe that
is really measuring a SNP is not a finding. **Scoring a pre-fitted clock is a
different situation.** Every clock in this registry was trained on unmasked
data, before most of these masks existed. Its coefficients were fitted with
those probes in, whatever the probes were measuring. Removing them at score time
does not restore a truer number -- it deletes inputs the model expects and hands
the gap to the imputer.

So this module does not mask anything by default. It fetches the published
masks, reports what applying one would cost each clock in features and in
coefficient mass, and applies it when asked. The report is the point: a clock
that loses two percent of its weight is a different decision from one that
loses a fifth.
"""

from __future__ import annotations

import functools

import pandas as pd

from ..core.container import FalconData
from ..core.errors import DataError

__all__ = ["MASK_SOURCES", "apply_mask", "load_mask", "mask_report", "masked_probes"]

_BASE = ("https://github.com/zhou-lab/InfiniumAnnotationV1/raw/main/Anno")

#: Platform -> the published mask file. EPIC v2 masks are keyed by the full v2
#: probe name; :func:`load_mask` reduces them to the bare identifier so they
#: line up with the feature ids clocks use.
MASK_SOURCES: dict[str, str] = {
    "450K":   f"{_BASE}/HM450/HM450.hg38.mask.tsv.gz",
    "EPICv1": f"{_BASE}/EPIC/EPIC.hg38.mask.tsv.gz",
    "EPICv2": f"{_BASE}/EPICv2/EPICv2.hg38.mask.tsv.gz",
}

CITATION = ("Zhou W, Laird PW, Shen H. Comprehensive characterization, "
            "annotation and innovative use of Infinium DNA methylation BeadChip "
            "probes. Nucleic Acids Res 2017;45:e22.")


@functools.lru_cache(maxsize=4)
def load_mask(platform: str) -> pd.DataFrame:
    """The published mask for one platform, indexed by probe.

    Only flagged probes appear -- a probe absent from the table is unmasked.
    Columns: ``general`` (the authors' recommended mask) and ``reasons`` (the
    categories that flagged it, comma separated).
    """
    if platform not in MASK_SOURCES:
        raise DataError(
            f"no published mask for {platform!r}. Known: {', '.join(MASK_SOURCES)}.")
    from ..download import fetch

    df = pd.read_csv(fetch(MASK_SOURCES[platform]), sep="\t", low_memory=False)
    if "Probe_ID" not in df.columns:
        raise DataError(f"the {platform} mask file has no Probe_ID column")

    ids = df["Probe_ID"].astype(str)
    # EPIC v2 names probes cg00000029_II_F_C_rep1_EPIC; every clock's feature
    # list uses the bare form, and a mask keyed the long way masks nothing.
    bare = ids.str.split("_", n=1).str[0]
    general = (df["M_general"] if "M_general" in df.columns
               else pd.Series(True, index=df.index)).fillna(False).astype(bool)
    reasons = (df["maskUniq"] if "maskUniq" in df.columns
               else pd.Series("", index=df.index)).fillna("").astype(str)

    out = pd.DataFrame({"general": general.to_numpy(), "reasons": reasons.to_numpy()},
                       index=pd.Index(bare, name="feature_id"))
    # A bare id can appear more than once on v2 (replicate probes). Masked if
    # any of its copies is masked -- the conservative direction, and the one
    # that matches what a user means by "mask this CpG".
    return out.groupby(level=0).agg({"general": "max", "reasons": "first"})


def masked_probes(platform: str, *, kind: str = "general") -> frozenset[str]:
    """The set of probe ids the mask flags."""
    m = load_mask(platform)
    if kind == "general":
        return frozenset(m.index[m["general"]].astype(str))
    if kind == "any":
        return frozenset(m.index.astype(str))
    raise DataError(f"mask kind={kind!r}; expected 'general' or 'any'")


def mask_report(platform: str, *, clocks: str | list[str] = "scoreable",
                kind: str = "general", registry=None) -> pd.DataFrame:
    """What applying this mask would cost each clock.

    One row per clock: how many of its probes the mask flags, and -- the number
    that decides it -- what share of the model's total ``|coefficient|`` those
    probes carry. Two clocks losing the same probe count are not in the same
    position if one of them leans on the probes it is losing.

    Run this before :func:`apply_mask`, not after.
    """
    import numpy as np

    from ..registry import load as _load

    reg = registry if registry is not None else _load()
    bad = masked_probes(platform, kind=kind)

    if clocks == "scoreable":
        chosen = [c.id for c in reg if reg.has_coefficients(c.id)]
    elif clocks == "all":
        chosen = [c.id for c in reg]
    else:
        chosen = list(clocks)

    rows = []
    for cid in chosen:
        try:
            feats, coefs = reg.coefficients(cid)
        except Exception:  # noqa: BLE001 - no coefficients to weigh
            continue
        feats = list(feats)
        w = np.abs(np.asarray(coefs, dtype=float))
        hit = np.array([f in bad for f in feats])
        total = float(w.sum())
        rows.append({
            "clock": cid, "n_features": len(feats),
            "n_masked": int(hit.sum()),
            "frac_masked": round(float(hit.mean()), 4),
            "mass_masked": round(float(w[hit].sum() / total) if total else 0.0, 4),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("mass_masked", ascending=False).set_index("clock")


def apply_mask(data: FalconData, *, platform: str | None = None,
               kind: str = "general") -> FalconData:
    """Set masked probes to NaN. Not called anywhere automatically.

    .. warning::

       Every clock in this registry was fitted on unmasked data, before most of
       these masks existed. Removing a probe at score time does not recover a
       truer number: it deletes an input the coefficients expect and hands the
       gap to the imputer. Run :func:`mask_report` first and decide with the
       coefficient-mass column in front of you.

       Masking is the right default for an EWAS and a real decision for a clock.
    """
    plat = platform or data.platform
    if not plat:
        raise DataError("apply_mask needs a platform; declare one or pass platform=")
    bad = masked_probes(plat, kind=kind)
    cols = [c for c in data.X.columns if str(c) in bad]
    X = data.X.copy()
    X[cols] = float("nan")

    out = FalconData(X=X, obs=data.obs, modality=data.modality, units=data.units,
                     platform=data.platform, uns=dict(data.uns))
    out.uns["probe_mask"] = {
        "platform": plat, "kind": kind, "n_masked": len(cols),
        "frac_masked": round(len(cols) / max(data.n_features, 1), 6),
        "source": MASK_SOURCES[plat], "citation": CITATION,
    }
    return out
