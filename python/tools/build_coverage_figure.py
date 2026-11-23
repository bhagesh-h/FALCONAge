#!/usr/bin/env python3
"""Draw the catalogue's coverage map against the EPIC v1 array for the docs.

Writes ``test/output_figures/gallery/clock_coverage_across_the_genome.png``
(and the ``.txt`` beside it): one
genome track per DNA methylation clock in the catalogue, coloured where the
EPIC v1 array carries the CpG. The home page shows it right after the
introduction, and the figure gallery stages it from there. Rerun when the
catalogue changes (needs the EPIC v1 manifest, fetched once and cached):

    python python/tools/build_coverage_figure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))

import falconage as fa  # noqa: E402
OUT_DIR = ROOT / "test" / "output_figures" / "gallery"


def main() -> int:
    fig, table = fa.plot.clock_coverage_manhattan(platform="EPICv1")
    OUT = fa.plot.save_figure(fig, OUT_DIR)
    print(f"wrote {OUT.relative_to(ROOT)}: {len(table)} clocks, "
          f"mean EPIC v1 coverage {table['coverage'].mean():.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
