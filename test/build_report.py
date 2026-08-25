#!/usr/bin/env python3
"""Build the single-file HTML report from what ``run_all.py`` produced.

Run ``run_all.py`` first; this reads its outputs rather than re-scoring, so the
report and the tables in ``test/README.md`` are guaranteed to be the same run.

    python test/run_all.py
    python test/build_report.py
    # -> test/output/falconage-test-report.html

The report is self-contained: figures are embedded as data URIs and there is no
external stylesheet, so it survives being emailed or dropped on a share. That
also makes it large, around 24 MB, which is the trade and is stated here rather
than discovered.

Rendering needs Quarto. Without it this still writes the ``.qmd`` and says what
to run, because the source is the reproducible artefact and the HTML is a
convenience built from it.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

import falconage as fa

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
FIGS = HERE / "output_figures"
GALLERY = FIGS / "gallery"
LOGO = HERE.parent / "logo" / "logo.png"

#: The dataset the report leads with. Widest age range in the corpus on EPIC,
#: so the calibration and Bland-Altman panels have a trend to show; a cohort
#: spanning three years would make every one of them a blob.
LEAD = "GSE182991"


def _read(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, index_col=0)
    except Exception as exc:  # pragma: no cover - a malformed table is not fatal
        print(f"  skipped {path.name}: {exc}")
        return None


def collect_tables() -> dict[str, pd.DataFrame]:
    """Every table worth showing, in the order a reader should meet them."""
    wanted = [
        ("Scores, all clocks", OUT / "bench" / LEAD / "scores_wide.csv"),
        ("Coverage and what was imputed", OUT / "bench" / LEAD / "coverage.csv"),
        ("Age acceleration", OUT / "bench" / LEAD / "acceleration.csv"),
        ("AA1 and AA2 benchmark", OUT / "bench" / "_combined" / "benchmark.csv"),
        ("Registry catalogue", OUT / "registry" / "catalogue" / "clocks.csv"),
        ("Bundled clocks", OUT / "registry" / "catalogue" / "bundled.csv"),
        ("Licensed scaffolds", OUT / "registry" / "catalogue" / "licensed.csv"),
        ("Entropy and drift, per sample",
         OUT / "disorder" / LEAD / "entropy_drift.csv"),
        ("Sites whose variance rises with age",
         OUT / "disorder" / LEAD / "variable_sites_nominal.csv"),
        ("Noise barometer", OUT / "disorder" / LEAD / "noise_barometer.csv"),
        ("Clock sensitivity to clone structure",
         OUT / "disorder" / "clonality_simulation" / "clonality_slopes.csv"),
        ("Coefficient mass shared between clocks",
         OUT / "disorder" / "coefficient_mass" / "shared_mass.csv"),
        ("Clinical chemistry", OUT / "clinical" / "synthetic" / "scores_wide.csv"),
    ]
    tables = {}
    for label, path in wanted:
        df = _read(path)
        if df is not None and not df.empty:
            tables[label] = df
    return tables


def collect_figures() -> list[Path]:
    """The gallery, with the conclusion figure held back for the hero slot."""
    if not GALLERY.exists():
        return []
    return sorted(p for p in GALLERY.glob("*.png")
                  if p.name != "clock_atlas.png")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-render", action="store_true",
                    help="write the .qmd and stop")
    args = ap.parse_args(argv)

    if not OUT.exists():
        sys.exit("test/output is empty. Run: python test/run_all.py")

    tables = collect_tables()
    figures = collect_figures()
    print(f"  {len(tables)} table(s), {len(figures)} figure(s)")
    if not tables:
        sys.exit("no tables found under test/output; run test/run_all.py first")

    # clock_atlas is the conclusion figure: it is the one panel that answers
    # "which of these algorithms actually work", every clock against every
    # study on one pair of axes, so it leads rather than appearing 14 figures in.
    hero = GALLERY / "clock_atlas.png"
    verdict = None
    bench = _read(OUT / "bench" / "_combined" / "benchmark.csv")
    if bench is not None and "total" in bench.columns:
        best = bench["total"].idxmax()
        verdict = (f"Across {len(bench)} clocks and the ten benchmark studies, "
                   f"{best} carries the highest total score. The atlas above is "
                   "the honest version of that sentence: the spread within any "
                   "one clock across studies is wider than the gap between "
                   "most pairs of clocks.")

    qmd = OUT / "falconage-test-report.qmd"
    fa.report.write_quarto_report(
        None, qmd,
        figures=figures,
        tables=tables,
        logo=LOGO if LOGO.exists() else None,
        title="FALCONAge test corpus results",
        verdict=verdict,
        conclusion_figure=hero if hero.exists() else None,
    )
    print(f"  wrote {qmd}")

    if args.no_render:
        print(f"  render with: quarto render {qmd}")
        return 0

    if shutil.which("quarto") is None:
        print("  quarto not on PATH; the .qmd is written and can be rendered "
              "anywhere:\n    quarto render " + str(qmd))
        return 0

    proc = subprocess.run(["quarto", "render", str(qmd), "--to", "html"],
                          cwd=qmd.parent)
    if proc.returncode != 0:
        return proc.returncode
    html = qmd.with_suffix(".html")
    if html.exists():
        print(f"  {html}  ({html.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
