"""Illumina array manifests: fetched and cached, never vendored.

WHY THIS IS A FETCH AND NOT A FILE IN THE WHEEL. The manifests are Illumina's,
they are 11-25 MB compressed each, and their licence is not one FALCONAge can
redistribute under. ``methylprep`` (MIT) resolved the same problem the same way:
it downloads them from a public S3 bucket and caches them in the user's home.
This uses the identical bucket and the identical filenames, through the
package's existing :func:`falconage.download.fetch`, so a manifest is fetched
once per machine, checksummed, and reused.

That means the raw-IDAT path needs a network the first time and never again.
Everything else in FALCONAge works offline; this one step does not, and the
error says so rather than failing at a parse.

WHAT A MANIFEST IS FOR. An IDAT is keyed by *bead address*, not by probe. Two
addresses per Infinium type I probe and one per type II, and which channel a
type I probe reads in is a property of its chemistry, not of the file. Nothing
downstream -- background correction, detection, dye bias, betas -- can be
computed without that mapping, which is why this is the first module in the
raw-data chain rather than an optional extra.
"""

from __future__ import annotations

import functools
import hashlib
from pathlib import Path

import pandas as pd

from ..core.errors import DataError

__all__ = ["MANIFESTS", "detect_manifest_platform", "fetch_manifest",
           "load_manifest", "manifest_record"]

BUCKET = "https://s3.amazonaws.com/array-manifest-files"

#: Platform -> (filename, approximate address count in an IDAT of that array).
#: The counts are what :func:`detect_manifest_platform` matches against; they
#: are the number of bead addresses on the chip, not the number of probes,
#: which is smaller because every type I probe uses two.
MANIFESTS: dict[str, tuple[str, int]] = {
    "27K":    ("hm27.hg19.manifest.csv.gz", 27_578),
    "450K":   ("HumanMethylation450k_15017482_v3.csv.gz", 622_399),
    "EPICv1": ("HumanMethylationEPIC_manifest_v2.csv.gz", 1_051_815),
    "EPICv2": ("CombinedManifestEPIC_manifest_CoreColumns_v2.csv.gz", 1_105_209),
}

#: The five columns the chain needs. The files carry coordinates and old-build
#: coordinates too; they are read but not required, because a manifest revision
#: that drops a genome build should not stop anybody scoring a clock.
REQUIRED = ("IlmnID", "AddressA_ID", "AddressB_ID", "Infinium_Design_Type",
            "Color_Channel")


def fetch_manifest(platform: str) -> Path:
    """Download the manifest for a platform into the cache, or return it."""
    if platform not in MANIFESTS:
        raise DataError(
            f"no manifest known for platform {platform!r}.\n"
            f"  Known: {', '.join(MANIFESTS)}.\n"
            "  A platform FALCONAge cannot name is one it cannot map addresses "
            "to probes for, so the raw-IDAT path is closed for it. Supply "
            "betas instead.")
    from ..download import fetch

    name = MANIFESTS[platform][0]
    try:
        return fetch(f"{BUCKET}/{name}")
    except Exception as exc:  # noqa: BLE001 - the message matters more than the type
        raise DataError(
            f"could not fetch the {platform} manifest ({name}).\n"
            f"  {exc}\n"
            "  This is the one step in FALCONAge that needs a network. It runs "
            "once per machine and the result is cached; every other path works "
            "offline.") from exc


@functools.lru_cache(maxsize=4)
def load_manifest(platform: str) -> pd.DataFrame:
    """Address-to-probe mapping for one platform, indexed by probe id.

    Columns: ``address_a``, ``address_b`` (nullable), ``type`` (``I``/``II``)
    and ``channel`` (``Grn``/``Red``, empty for type II). Cached for the
    process -- parsing 850,000 rows per call would dominate a run.

    Type II probes have no ``address_b`` and no channel: they read methylated
    in green and unmethylated in red at a single address. Type I probes have
    two addresses in *one* channel, and the other channel at those same
    addresses is the out-of-band signal that :mod:`falconage.preprocess.idat`
    uses as its background null.
    """
    p = fetch_manifest(platform)
    df = pd.read_csv(p, low_memory=False)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise DataError(
            f"the {platform} manifest is missing {missing}.\n"
            f"  It has {list(df.columns)[:8]}...\n"
            "  Delete the cached copy and refetch; a truncated download is the "
            "usual cause.")

    # `.to_numpy()` on every column, not the Series. Handing pandas a dict of
    # Series together with an explicit `index=` makes it *reindex* them onto
    # that index rather than relabel them -- and since the source frame has a
    # RangeIndex and the target is probe ids, nothing aligns and every value
    # becomes NaN. The manifest then loads as zero probes and the failure
    # surfaces two modules later as "none of the manifest's addresses are in
    # this IDAT", which is a true statement about entirely the wrong thing.
    idx = pd.Index(df["IlmnID"].astype(str), name="feature_id")
    out = pd.DataFrame({
        "address_a": pd.to_numeric(df["AddressA_ID"], errors="coerce").to_numpy(),
        "address_b": pd.to_numeric(df["AddressB_ID"], errors="coerce").to_numpy(),
        "type": df["Infinium_Design_Type"].astype(str).str.strip().to_numpy(),
        "channel": df["Color_Channel"].fillna("").astype(str).str.strip().to_numpy(),
    }, index=idx)

    # EPIC v2 names probes `cg00000029_II_F_C_rep1_EPIC` and carries the bare
    # identifier alongside. Every clock's feature list uses the bare form, so
    # the bare form is the index and the full name is kept beside it -- the
    # replicate structure is real and `aggregate_replicate_probes` still needs
    # to see it.
    if "trimmed_id" in df.columns:
        out["probe_name"] = out.index.to_numpy()
        out.index = pd.Index(df["trimmed_id"].astype(str), name="feature_id")

    out = out[out["address_a"].notna()]
    out = out[out["type"].isin(["I", "II"])]
    # A type I probe with no second address cannot be resolved into a
    # methylated and an unmethylated bead, so it is not a measurement.
    bad = (out["type"] == "I") & out["address_b"].isna()
    if bad.any():
        out = out[~bad]
    out = out[~out.index.duplicated(keep="first")]
    return out


def manifest_record(platform: str) -> dict[str, str]:
    """What the run manifest records about the mapping that produced a beta.

    A beta matrix is a function of which manifest turned addresses into probes,
    in exactly the way a score is a function of which coefficient file was used.
    Recording one and not the other would be an odd place to stop.
    """
    name, _ = MANIFESTS[platform]
    p = fetch_manifest(platform)
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    m = load_manifest(platform)
    return {
        "platform": platform, "file": name, "url": f"{BUCKET}/{name}",
        "sha256": h.hexdigest(), "n_probes": str(len(m)),
        "n_type_i": str(int((m["type"] == "I").sum())),
        "n_type_ii": str(int((m["type"] == "II").sum())),
        "redistributed": "no -- fetched from Illumina's public bucket and cached",
    }


def detect_manifest_platform(n_addresses: int, *, tolerance: float = 0.03,
                             margin: float = 2.0) -> str:
    """Which array an IDAT with this many bead addresses came from.

    Matched on address count, with two conditions rather than one: the best
    candidate must be within ``tolerance``, **and** it must be at least
    ``margin`` times closer than the runner-up.

    The second condition is not defensive padding. EPIC v1 and EPIC v2 carry
    1,051,815 and 1,105,209 addresses -- five percent apart, not the factor that
    separates 27K from 450K from EPIC. A single tolerance wide enough to absorb
    chip-to-chip variation is also wide enough to match both, and the failure
    would be silent: the wrong manifest maps addresses to the wrong probes and
    returns a full matrix of plausible wrong betas that no downstream check can
    catch. A count sitting between the two is genuinely ambiguous, and the
    honest response is to ask.
    """
    errs = sorted(((abs(n_addresses - expect) / expect, name)
                   for name, (_, expect) in MANIFESTS.items()))
    (best_err, best), (next_err, runner_up) = errs[0], errs[1]
    known = ", ".join(f"{k} ~{v[1]:,}" for k, v in MANIFESTS.items())

    if best_err > tolerance:
        raise DataError(
            f"{n_addresses:,} bead addresses matches no known array within "
            f"{tolerance:.0%} (closest is {best} at {best_err:.1%} off).\n"
            f"  Known: {known}.\n"
            "  Pass platform= explicitly if you know which array this is. "
            "Guessing would map addresses to the wrong probes and return a full "
            "matrix of plausible wrong numbers.")
    if best_err > 0 and next_err < margin * best_err:
        raise DataError(
            f"{n_addresses:,} bead addresses is ambiguous between {best} "
            f"({best_err:.1%} off) and {runner_up} ({next_err:.1%} off).\n"
            f"  Known: {known}.\n"
            "  EPIC v1 and v2 are only five percent apart in address count, so "
            "a count between them cannot be resolved from the count alone. "
            "Pass platform= explicitly.")
    return best
