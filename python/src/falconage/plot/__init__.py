"""The standard figure set for aging-clock analysis.

Twenty figures in five families -- per-clock accuracy, age acceleration, cross-
clock agreement, quality control, and benchmarking -- chosen from what the
literature to 2026 actually publishes: the predicted-against-chronological
scatter every clock paper opens with, the Bland-Altman and residual pair the
2024 Nature Reviews Genetics methods review argues for, the clock correlation
heatmap and UMAP overlays from the pyaging paper, forest plots from the survival
and meta-analytic literature, spaghetti trajectories from the longitudinal
cohorts, and the beta-density and detection QC that every array pipeline runs.

Three rules hold across all of them:

**Every figure returns ``(figure, data)``.** The data half is not a courtesy: R
renders the same numbers with ggplot2 rather than shipping a second matplotlib,
so anything a figure knows has to be available as a frame. It also means a user
who dislikes these defaults can plot the frame without reverse-engineering it.

**Every figure carries its own title, subtitle and one-line description**, and
all three come from ``colorscheme.yaml`` rather than from code. A figure ends up
pasted into a slide with no caption sooner or later, and at that point the
description is the only thing telling a reader what to conclude.

**Nothing here picks a colour.** Every colour is a lookup in the same file, so
restyling a whole report is one file and both languages change together.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np  # noqa: F401
import pandas as pd  # noqa: F401

from . import spec
from .spec import (  # noqa: F401
    group_colours, palette, platform_colour, semantic, theme_value,
)
from ._common import (  # noqa: F401
    NothingToPlot, PALETTE, _age, _dress, _figure_text, _mpl, _new,
    _radial_labels, _ref, _require_signal, _titles_outside, _unit,
    _wrap_caption,
)
from .accuracy import ba_vs_ca, bland_altman, calibration
from .acceleration import (
    acceleration_by_group, acceleration_density, acceleration_heatmap,
    forest, timecourse, trajectory,
)
from .agreement import (
    clock_chord, clock_corr, clock_embedding, clock_pca, clock_radar,
    clock_scatter_matrix,
)
from .qc import (
    beta_density, coverage_bar, missingness, platform_comparison, sex_check,
    study_comparison,
)
from .benchmark import benchmark_bars, benchmark_error_bias, benchmark_heatmap
from .outcomes import kaplan_meier, volcano
from .uncertainty import (
    consensus_plot, platform_bias, reliability_forest, score_interval,
)
from .atlas import clock_atlas

__all__ = [
    "NothingToPlot", "acceleration_by_group", "acceleration_density", "acceleration_heatmap",
    "ba_vs_ca", "benchmark_bars", "benchmark_error_bias", "benchmark_heatmap",
    "beta_density", "bland_altman", "calibration", "clock_corr", "clock_embedding",
    "clock_chord", "clock_pca", "clock_radar", "clock_scatter_matrix",
    "consensus_plot", "coverage_bar",
    "forest", "kaplan_meier", "missingness", "platform_bias",
    "platform_comparison", "reliability_forest", "save_all", "score_interval",
    "sex_check", "spec", "study_comparison",
    "timecourse", "trajectory", "clock_atlas", "volcano",
]


# ===========================================================================
def save_all(result, outdir, *, data=None, bench=None, acc=None, age_col: str = "age",
             group: str | None = None, platform_col: str | None = None,
             dataset_col: str | None = None, max_per_clock: int = 4,
             se=None, conformal=None, consensus=None) -> dict[str, Path]:
    """Render every figure this result can support and write it as PNG.

    Skips anything the data cannot support rather than raising -- a result with
    no age column should still get its coverage and agreement figures.
    """
    plt = _mpl()
    d = Path(outdir)
    d.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    skipped: dict[str, str] = {}

    def emit(name, fn):
        try:
            fig, _ = fn()
        except NothingToPlot as exc:
            # An empty figure is worse than a missing one. Recorded so the
            # absence is legible rather than mysterious.
            skipped[name] = str(exc)
            return
        except Exception as exc:
            skipped[name] = f"{type(exc).__name__}: {exc}"
            return
        p = d / f"{name}.png"
        fig.savefig(p, bbox_inches="tight", dpi=theme_value("dpi"),
                    facecolor="white")
        plt.close(fig)
        written[name] = p

    age_clocks = [c for c in result.scores.columns
                  if result.registry.get(c).scale_type == "age_years"][:max_per_clock]

    emit("coverage_bar", lambda: coverage_bar(result))
    emit("clock_corr", lambda: clock_corr(result))
    emit("clock_pca", lambda: clock_pca(result, colour_by=group or age_col))
    # clock_scatter_matrix and per-clock clock_embedding are deliberately not
    # emitted: the first restates the correlation heatmap pair by pair, the
    # second recolours the layout clock_pca already draws. Both stay callable
    # for anyone who wants them; neither earns a slot in every run's output.
    emit("clock_chord", lambda: clock_chord(result))
    emit("clock_chord", lambda: clock_chord(result))
    emit("clock_radar", lambda: clock_radar(result, group=group))
    if data is not None:
        emit("beta_density", lambda: beta_density(data))
        emit("missingness", lambda: missingness(data))
        emit("sex_check", lambda: sex_check(data))

    for c in age_clocks:
        emit(f"ba_vs_ca_{c}", lambda c=c: ba_vs_ca(result, c, age_col=age_col, group=group))
        emit(f"bland_altman_{c}", lambda c=c: bland_altman(result, c, age_col=age_col))
        emit(f"calibration_{c}", lambda c=c: calibration(result, c, age_col=age_col))
        if platform_col:
            emit(f"platform_{c}", lambda c=c: platform_comparison(
                result, c, platform_col=platform_col))
        if dataset_col:
            emit(f"study_{c}", lambda c=c: study_comparison(result, c,
                                                            dataset_col=dataset_col))

    if acc is not None:
        emit("acceleration_heatmap", lambda: acceleration_heatmap(acc, result.obs, group))
        for c in [x for x in age_clocks if x in acc.columns]:
            if group:
                emit(f"acceleration_group_{c}",
                     lambda c=c: acceleration_by_group(acc, c, result.obs, group))
            emit(f"acceleration_density_{c}",
                 lambda c=c: acceleration_density(acc, c, result.obs, group))

    # The uncertainty set. Emitted whenever the caller has computed them, which
    # is the point at which a reader can be told how much of the table above is
    # the assay rather than the biology.
    if se is not None:
        emit("reliability_forest", lambda: reliability_forest(se))
        for c in age_clocks[:1]:
            emit(f"score_interval_{c}",
                 lambda c=c: score_interval(result, c, se=se, conformal=conformal,
                                            age_col=age_col))
    emit("platform_bias", lambda: platform_bias())
    if consensus is not None:
        emit("consensus_plot", lambda: consensus_plot(consensus))

    if bench is not None:
        emit("benchmark_bars", lambda: benchmark_bars(bench))
        emit("benchmark_error_bias", lambda: benchmark_error_bias(bench))
        emit("benchmark_heatmap", lambda: benchmark_heatmap(bench))
        emit("forest", lambda: forest(bench, top=40))
        # The one figure that needs several studies. It raises NothingToPlot on
        # a single cohort, so no condition is needed here.
        emit("clock_atlas", lambda: clock_atlas(result, bench))

    if skipped:
        (d / "SKIPPED.txt").write_text(
            "Figures not drawn, and why. An empty figure reads as a measurement\n"
            "of nothing; an absent one with a reason reads as an absence.\n\n"
            + "\n".join(f"{k}: {v}" for k, v in sorted(skipped.items())) + "\n",
            encoding="utf-8", newline="\n")
    return written

