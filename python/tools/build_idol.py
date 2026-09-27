#!/usr/bin/env python3
"""Extract the six-cell IDOL reference for blood deconvolution.

WHAT IT IS. Salas et al. 2018 (Genome Biology 19:64) chose, with IDOL, the 450
EPIC CpGs whose methylation best separates CD8+ T, CD4+ T, NK, B cells,
monocytes and neutrophils, and published the mean beta of each cell type at
each one from their immunomagnetically sorted reference arrays. The authors'
Bioconductor package FlowSorted.Blood.EPIC (GPL-3) ships that table as
``IDOLOptimizedCpGs.compTable``, and a 350-CpG version for the 450K array as
``IDOLOptimizedCpGs450klegacy.compTable``. Its documentation gives the way to
use the table on a beta matrix: ``projectCellType_CP(betas[IDOL CpGs, ],
compTable, nonnegative = TRUE, lessThanOne = FALSE)`` on Noob-processed betas,
which is what ``falconage.models.deconvolution`` implements.

The twelve-cell library (FlowSorted.BloodExtended.EPIC, Salas et al. 2022) is
under a Dartmouth research-use source-code licence and is not built here.

Usage
-----
    python python/tools/build_idol.py            # needs Rscript
    python python/tools/build_idol.py --check
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import build_references as br  # noqa: E402

OUT = br.ROOT / "python" / "src" / "falconage" / "registry" / "data" / "coefficients"
COMMIT = "b837178f81082ba386a13d5669d08dd8d3df6222"
RAW = f"https://raw.githubusercontent.com/immunomethylomics/FlowSorted.Blood.EPIC/{COMMIT}/data/"

br.SOURCES.update({
    "idol_epic": {"url": RAW + "IDOLOptimizedCpGs.compTable.rda",
                  "sha256": "ab1ae36984a473aaa653ea0192d12da6cd60bd9254b46244c3a042b08ebf389b"},
    "idol_450k": {"url": RAW + "IDOLOptimizedCpGs450klegacy.compTable.rda",
                  "sha256": "a057fdc254263cc285abdde99600b43176f2afe969f6ec69168e380171c4a723"},
})
TABLES = {
    "idol_epic.csv": ("idol_epic", "e$IDOLOptimizedCpGs.compTable"),
    "idol_450klegacy.csv": ("idol_450k", "e$IDOLOptimizedCpGs450klegacy.compTable"),
}


def render(frame) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["feature_id", *frame.columns])
    for fid, row in frame.iterrows():
        w.writerow([fid, *(repr(float(v)) for v in row)])
    return buf.getvalue().encode("utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--cache", default=None, help="where to keep the downloads")
    args = ap.parse_args(argv)
    cache = Path(args.cache) if args.cache else Path.home() / ".cache" / "falconage-references"
    stale = []
    for name, (key, expr) in TABLES.items():
        data = render(br.rdata_matrix(br.fetch(key, cache), expr))
        p = OUT / name
        if args.check:
            if not p.exists() or p.read_bytes() != data:
                stale.append(name)
            continue
        p.write_bytes(data)
        print(f"  wrote {name}: sha256 {hashlib.sha256(data).hexdigest()}")
    if args.check:
        if stale:
            print(f"stale: {', '.join(stale)}; run python/tools/build_idol.py")
            return 1
        print(f"{len(TABLES)} IDOL tables are current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
