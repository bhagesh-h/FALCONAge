"""One call that checks, scores, quantifies, interprets and writes a run.

What ``falconage report`` does, as a library function, so that R and scripts
can run the same sequence on a dataset they already hold and get the same
directory: the tables and figures of every step, the one-page report, and, on
request, the step-ordered Quarto report of :mod:`falconage.report.quarto`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd

__all__ = ["run_report"]


def run_report(data: Any, outdir: str | Path, *, clocks: Any = "compatible",
               group_col: str | None = None, reference: Any = None,
               level: float = 0.90, min_coverage: float = 0.8,
               figures: bool = True, quarto: bool = False, render: bool = True,
               title: str = "FALCONAge report",
               log: Callable[[str], None] = print) -> dict[str, Path]:
    """Write every deliverable of one run into ``outdir``.

    Steps, in order: quality control (methylation), scoring, technical standard
    error, conformal intervals, age acceleration where age is known, a consensus
    test across clocks when ``group_col`` names a column, interpretation and
    evidence tables, figures, the one-page ``report.html`` and, with
    ``quarto=True``, ``falconage_report.qmd`` rendered to a self-contained
    ``falconage_report.html`` when Quarto is available (``render=False`` writes
    the source only). A step the data cannot support is reported through
    ``log`` and skipped, not raised.

    Returns the paths of the main files, by name.
    """
    import falconage as fa

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    if data.modality == "dna_methylation":
        qc = fa.qc(data)
        qc.per_sample.to_csv(out / "qc_per_sample.csv")
        for w in qc.warnings:
            log(f"  QC: {w}")

    res = fa.score(data, clocks=clocks, min_coverage=min_coverage)
    log(f"scored {res.scores.shape[1]} clock(s); {len(res.skipped)} skipped")
    pd.DataFrame(sorted(res.skipped.items()), columns=["clock", "reason"]).to_csv(
        out / "skipped_clocks.csv", index=False)

    se = conf = cons = acc = None
    try:
        se = fa.technical_se(res, data)
        se.se.to_csv(out / "technical_se.csv")
        se.diagnostics.to_csv(out / "reliability_diagnostics.csv")
    except Exception as exc:                      # noqa: BLE001
        log(f"  technical_se unavailable: {exc}")
    try:
        conf = fa.conformal_interval(res, level=level)
        conf.to_csv(out / "conformal_interval.csv", index=False)
    except Exception as exc:                      # noqa: BLE001
        log(f"  conformal interval unavailable: {exc}")
    if "age" in res.obs.columns:
        try:
            acc = fa.acceleration(res, method="residual")
            acc.to_csv(out / "acceleration.csv", index_label="sample_id")
        except Exception as exc:                  # noqa: BLE001
            log(f"  acceleration unavailable: {exc}")
    if group_col and group_col in res.obs.columns:
        try:
            cons = fa.consensus(res, group_col, reference=reference)
            cons.table.to_csv(out / "consensus.csv", index_label="clock")
            (out / "consensus_verdict.txt").write_text(
                f"{cons.verdict} -- {cons.why}\n", encoding="utf-8", newline="\n")
            log(f"  consensus: {cons.verdict}")
        except Exception as exc:                  # noqa: BLE001
            log(f"  consensus unavailable: {exc}")

    res.write(out)
    res.interpretation().to_csv(out / "interpretation.csv")
    res.evidence().to_csv(out / "evidence.csv", index=False)

    if figures:
        from .. import plot as fplot

        w = fplot.save_all(res, out / "figures", data=data, acc=acc, group=group_col,
                           se=se, conformal=conf, consensus=cons)
        log(f"  {len(w)} figure(s)")

    from .html import write_report

    written["report"] = write_report(res, out / "report.html", group=group_col,
                                     title=title)
    if quarto:
        from .quarto import write_quarto_report

        try:
            written["quarto"] = write_quarto_report(out, res, title=title, render=render)
        except RuntimeError as exc:
            # The source is written before rendering is attempted, so a missing
            # Quarto costs the HTML, not the report.
            written["quarto"] = out / "falconage_report.qmd"
            log(f"  {exc}")
    return written
