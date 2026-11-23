#!/usr/bin/env python3
"""Write ``registry/data/cpg_positions.csv.gz``: where every clock CpG sits.

The coverage Manhattan plot draws each clock's CpGs along the genome, and it
has to do that offline, before any clock is scored. So the chromosome and
position of every CpG a catalogue clock uses is shipped with the package,
taken from Illumina's public array manifests (GRCh37/hg19 coordinates): EPIC v1
first, then 450K, then 27K for probes only the older arrays carry.

Run once after the catalogue changes (needs a network the first time; the
manifests are cached):

    python python/tools/build_cpg_positions.py
"""

from __future__ import annotations

import gzip
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))

import falconage as fa  # noqa: E402
from falconage.preprocess.manifest import fetch_manifest  # noqa: E402

OUT = ROOT / "python" / "src" / "falconage" / "registry" / "data" / "cpg_positions.csv.gz"
CHR_COLS = ("CHR", "Chr", "Chromosome", "chromosome")
POS_COLS = ("MAPINFO", "MapInfo", "Coordinate", "pos")


def _positions(platform: str) -> pd.DataFrame:
    df = pd.read_csv(fetch_manifest(platform), low_memory=False)
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
            wanted.update(reg.feature_ids(c.id))
        except Exception:  # noqa: BLE001 - a clock without per-CpG features
            continue
    table = pd.DataFrame(columns=["chr", "pos"])
    for platform in ("EPICv1", "450K", "27K"):
        pos = _positions(platform)
        new = pos.loc[pos.index.intersection(sorted(wanted)).difference(table.index)]
        table = pd.concat([table, new])
        print(f"{platform}: {len(new)} positions added")
    table.index.name = "feature_id"
    table = table.sort_index()
    with gzip.open(OUT, "wt", encoding="utf-8", compresslevel=9) as fh:
        table.to_csv(fh)
    print(f"wrote {OUT.relative_to(ROOT)}: {len(table)} of {len(wanted)} clock CpGs placed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
