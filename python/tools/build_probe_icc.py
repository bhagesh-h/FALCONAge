"""Derive the bundled per-probe reliability table.

WHAT THIS PRODUCES. ``registry/data/probe_icc.csv`` -- one row per CpG that at
least one registry clock uses, carrying the probe's published test-retest
intraclass correlation. ``falconage.uncertainty.technical_se`` reads it to turn
a point estimate into an interval.

WHERE THE NUMBERS COME FROM. Zhang et al., *Critical evaluation of the
reliability of DNA methylation probes on the Illumina MethylationEPIC v1.0
BeadChip microarrays*, Epigenetics 2024;19(1), doi:10.1080/15592294.2024.2333660.
69 blood DNA samples from ADNI, each assayed twice on EPIC v1, preprocessed with
SeSAMe 2. ICC is the two-way random-effects, absolute-agreement, single-rating
model -- ICC(2,1) in Shrout-Fleiss terms -- over 640,960 probes.

    https://github.com/TransBioInfoLab/DNAm-reliability
    results/ADNI_ICC_annotation.xlsx

Released CC BY 4.0, so the derived subset ships in the wheel with attribution
recorded in the file header and in the registry's source record.

WHY A SUBSET, GZIPPED, TWO COLUMNS. The published table is a 72 MB xlsx over the
whole array with nine columns. Three reductions, each measured:

* restricted to probes a *scoreable* clock names -- 283,860 of 640,960. Not
  26,000 as first assumed: ``zhangblup`` is a BLUP clock over 319,607 probes and
  dominates the union on its own.
* ``feature_id,icc`` only. The published mean and SD of beta are the ADNI
  cohort's, and the propagation uses the *user's* cohort SD, so carrying them
  would be two columns of the wrong number.
* gzipped, because a column of cg identifiers compresses well.

72 MB xlsx -> 10.7 MB csv -> the shipped file. Everything else about the
published table stays where it is published.

WHY NOT AT RUN TIME. Reading it needs openpyxl and 72 MB of network. Deriving
once and checksumming the result is the same discipline the coefficient files
already follow: the artefact ships, the recipe is in the repository, and CI
checks they still agree.

Usage
-----
    python python/tools/build_probe_icc.py                # fetch, derive, write
    python python/tools/build_probe_icc.py --source X.xlsx  # from a local copy
    python python/tools/build_probe_icc.py --check        # CI: fail if stale
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))

TARGET = (ROOT / "python" / "src" / "falconage" / "registry" / "data"
          / "probe_icc.csv.gz")

SOURCE_URL = ("https://github.com/TransBioInfoLab/DNAm-reliability/raw/main/"
              "results/ADNI_ICC_annotation.xlsx")
CITATION = ("Zhang W, Young JI, Gomez L, Schmidt MA, Lukacsovich D, Varma A, "
            "Chen XS, Kunkle B, Martin ER, Wang L. Critical evaluation of the "
            "reliability of DNA methylation probes on the Illumina "
            "MethylationEPIC v1.0 BeadChip microarrays. Epigenetics "
            "2024;19(1). doi:10.1080/15592294.2024.2333660")

#: The published sheet puts a title in row 1 and the real header in row 2.
_COLS = ["cpg", "seqnames", "start", "end", "annotation", "mean_beta", "sd_beta",
         "icc", "icc_class"]


def registry_features() -> set[str]:
    """Every feature a clock that can actually be scored offline names.

    Restricted to ``has_coefficients``. Taking every clock in the registry
    instead was measured at 283,860 probes and a 12 MB file, because the
    scaffolded PC clocks declare feature lists in the tens of thousands and
    their weights do not ship. Twelve megabytes in a wheel to carry reliability
    figures for probes no bundled clock reads is the wrong trade.

    A user who registers a licensed tier C weight file can pass their own table
    to ``technical_se(icc=...)``; the function says so when a probe is missing.
    """
    import falconage as fa

    reg = fa.registry.load()
    feats: set[str] = set()
    for c in reg:
        if not reg.has_coefficients(c.id):
            continue
        try:
            feats.update(reg.feature_ids(c.id))
        except Exception:  # noqa: BLE001 - nothing resolvable for this clock
            continue
    return feats


def read_published(source: str | Path) -> pd.DataFrame:
    df = pd.read_excel(source, skiprows=2, header=None, names=_COLS,
                       usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9])
    df = df.dropna(subset=["cpg", "icc"])
    for col in ("mean_beta", "sd_beta", "icc"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["icc"])


def render(source: str | Path) -> tuple[bytes, dict]:
    pub = read_published(source)
    want = registry_features()
    sub = pub[pub["cpg"].isin(want)][["cpg", "icc"]].sort_values("cpg")
    sub.columns = ["feature_id", "icc"]
    # Four decimals. ICC runs -0.28 to 1.00 and the estimate behind it comes
    # from 69 pairs, so the fifth decimal is noise being stored at full price.
    sub["icc"] = sub["icc"].round(4)

    buf = io.StringIO()
    buf.write(f"# {CITATION}\n")
    buf.write(f"# source: {SOURCE_URL}\n")
    buf.write("# licence: CC BY 4.0\n")
    buf.write("# ICC(2,1), two-way random effects, absolute agreement, single "
              "rating; 69 subjects assayed twice on EPIC v1, whole blood\n")
    buf.write(f"# rows: {len(sub)} of {len(pub)} published probes, restricted to "
              "features named by a clock that can be scored offline\n")
    sub.to_csv(buf, index=False, lineterminator="\n")

    # mtime=0 so the same input always produces the same bytes; otherwise the
    # --check guard would fail on every rebuild for no reason.
    import gzip

    raw = buf.getvalue().encode("utf-8")
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode="wb", compresslevel=9, mtime=0) as gz:
        gz.write(raw)

    stats = {
        "published": len(pub),
        "registry_features": len(want),
        "matched": len(sub),
        "median_icc": float(sub["icc"].median()),
        "frac_icc_ge_0.5": float((sub["icc"] >= 0.5).mean()),
        "plain_bytes": len(raw),
    }
    return out.getvalue(), stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", default=None,
                    help="local xlsx; default fetches SOURCE_URL into the cache")
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if the bundled table is out of date")
    args = ap.parse_args(argv)

    src = args.source
    if src is None:
        from falconage.download import fetch

        src = fetch(SOURCE_URL)

    blob, stats = render(src)
    current = TARGET.read_bytes() if TARGET.exists() else None

    if args.check:
        if current != blob:
            print(f"{TARGET.relative_to(ROOT)} is stale; run "
                  "python/tools/build_probe_icc.py")
            return 1
        print(f"{TARGET.relative_to(ROOT)} is current ({stats['matched']} probes)")
        return 0

    TARGET.write_bytes(blob)
    digest = hashlib.sha256(blob).hexdigest()
    print(f"wrote {TARGET.relative_to(ROOT)}")
    print(f"  {stats['matched']} of {stats['registry_features']} registry features "
          f"matched, from {stats['published']} published probes")
    print(f"  median ICC {stats['median_icc']:.3f}; "
          f"{stats['frac_icc_ge_0.5']:.1%} at ICC >= 0.5")
    print(f"  {stats['plain_bytes'] / 1e6:.1f} MB plain -> "
          f"{len(blob) / 1e6:.1f} MB gzipped")
    print(f"  sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
