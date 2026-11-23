#!/usr/bin/env python3
"""Write ``registry/data/cpg_positions.csv.gz``: where every clock CpG sits.

One row per CpG that a catalogue methylation clock uses, with three columns:

``chr``, ``pos``
    GRCh37/hg19 coordinates from Illumina's EPIC v1 manifest, then 450K, then
    27K for probes only the older arrays carry. Empty for a CpG no manifest
    places. The coverage Manhattan plot draws from these.
``arrays``
    Which Illumina arrays carry the CpG, as a bit mask: 1 = 27K, 2 = 450K,
    4 = EPIC v1, 8 = EPIC v2. Scoring uses it to tell a clock CpG the array
    never had (platform loss) from one the array had and the data lacks
    (removed after measurement, usually by probe QC).

Only these derived columns are shipped. The manifests are fetched at build
time only: 27K, 450K and EPIC v1 from the public bucket methylprep uses, and
EPIC v2 from the Zhou lab InfiniumAnnotation repository at a pinned commit
(the same source the methylarray pipeline pins for its EPIC v2 masks). Run once after the catalogue changes (needs a network):

    python python/tools/build_cpg_positions.py
"""

from __future__ import annotations

import gzip
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))

import falconage as fa  # noqa: E402
from falconage.download import fetch  # noqa: E402

OUT = ROOT / "python" / "src" / "falconage" / "registry" / "data" / "cpg_positions.csv.gz"
BUCKET = "https://s3.amazonaws.com/array-manifest-files"
MANIFESTS = {
    "27K": "hm27.hg19.manifest.csv.gz",
    "450K": "HumanMethylation450k_15017482_v3.csv.gz",
    "EPICv1": "HumanMethylationEPIC_manifest_v2.csv.gz",
}
EPICV2_URL = ("https://raw.githubusercontent.com/zhou-lab/InfiniumAnnotationV1/"
              "175566c157b8cb902ba6d9d7525547fca41adbe5/Anno/EPICv2/"
              "EPICv2.hg38.manifest.tsv.gz")
#: Bit per array in the ``arrays`` column. Keep in step with
#: ``falconage.preprocess.ARRAY_BITS``.
BITS = {"27K": 1, "450K": 2, "EPICv1": 4, "EPICv2": 8}
CHR_COLS = ("CHR", "Chr", "Chromosome", "chromosome")
POS_COLS = ("MAPINFO", "MapInfo", "Coordinate", "pos")
#: EPIC v2 names a probe `cg00000029_TC21`; clocks use the bare identifier.
_SUFFIX = re.compile(r"_[A-Za-z]{2}\d+$")


def _manifest(platform: str) -> pd.DataFrame:
    if platform == "EPICv2":
        return pd.read_csv(fetch(EPICV2_URL), sep="\t", low_memory=False)
    return pd.read_csv(fetch(f"{BUCKET}/{MANIFESTS[platform]}"), low_memory=False)


def _ids(platform: str, df: pd.DataFrame) -> pd.Index:
    if platform == "EPICv2":
        ids = df["Probe_ID"].astype(str).str.replace(_SUFFIX, "", regex=True)
    else:
        ids = df["IlmnID"].astype(str)
    return pd.Index(ids.unique())


def _positions(df: pd.DataFrame) -> pd.DataFrame:
    c = next(x for x in CHR_COLS if x in df.columns)
    p = next(x for x in POS_COLS if x in df.columns)
    out = pd.DataFrame({"feature_id": df["IlmnID"].astype(str),
                        "chr": df[c].astype(str).str.replace("chr", "", regex=False),
                        "pos": pd.to_numeric(df[p], errors="coerce")})
    out = out.dropna(subset=["pos"])
    out = out[out["chr"].isin([str(i) for i in range(1, 23)] + ["X", "Y"])]
    out["pos"] = out["pos"].astype("int64")
    return out.drop_duplicates("feature_id").set_index("feature_id")


def main() -> int:
    reg = fa.registry.load()
    wanted: set[str] = set()
    for c in reg:
        if c.data_type != "dna_methylation":
            continue
        try:
            wanted.update(f for f in reg.feature_ids(c.id)
                          if not f.lower().startswith("(intercept"))
        except Exception:  # noqa: BLE001 - a clock without per-CpG features
            continue
    wanted_idx = pd.Index(sorted(wanted))

    arrays = pd.Series(0, index=wanted_idx, dtype="int64")
    pos = pd.DataFrame(columns=["chr", "pos"])
    for platform in ("EPICv1", "450K", "27K", "EPICv2"):
        df = _manifest(platform)
        carried = wanted_idx.intersection(_ids(platform, df))
        arrays.loc[carried] |= BITS[platform]
        print(f"{platform}: carries {len(carried)} clock CpGs")
        if platform != "EPICv2":     # GRCh38 coordinates; the table is hg19
            p = _positions(df)
            new = p.loc[p.index.intersection(wanted_idx).difference(pos.index)]
            pos = pd.concat([pos, new])

    table = pd.DataFrame({"arrays": arrays}).join(pos, how="left")
    table = table[(table["arrays"] > 0) | table["pos"].notna()]
    table["pos"] = table["pos"].astype("Int64")
    table.index.name = "feature_id"
    table = table[["chr", "pos", "arrays"]].sort_index()
    with gzip.open(OUT, "wt", encoding="utf-8", compresslevel=9) as fh:
        table.to_csv(fh)
    print(f"wrote {OUT.relative_to(ROOT)}: {len(table)} of {len(wanted)} clock CpGs "
          f"({int(table['pos'].notna().sum())} placed on the genome)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
