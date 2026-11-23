"""The AA1/AA2 benchmark, drawn."""

from __future__ import annotations

from typing import Sequence  # noqa: F401

import numpy as np

from ..core.errors import AnalysisError  # noqa: F401
from . import spec
from .spec import (  # noqa: F401
    group_colours, palette, platform_colour, semantic, theme_value,
)
from ._common import (  # noqa: F401
    NothingToPlot, PALETTE, _age, _dress, _figure_text, _mpl, _new,
    _radial_labels, _ref, _require_signal, _titles_outside, _unit, _legend_outside,
    _wrap_caption,
)


def benchmark_bars(bench, *, alpha: float = 0.05):
    """AA2 and AA1 counts per clock, ordered by the combined total."""
    s = bench.summary_table
    if s.empty or (s[["AA2", "AA1"]].to_numpy() == 0).all():
        raise NothingToPlot("benchmark_bars: no clock detected anything, so "
                            "every bar would be zero")
    fig, ax = _new(height=max(3.0, 0.3 * len(s) + 1.6))
    y = np.arange(len(s))
    ax.barh(y - 0.2, s["AA2"], height=0.38, color=palette()[0], label="AA2 (vs own controls)")
    ax.barh(y + 0.2, s["AA1"], height=0.38, color=palette()[1], label="AA1 (vs zero)")
    ax.set_yticks(y, s.index, fontsize=theme_value("caption_size"))
    ax.invert_yaxis()
    _legend_outside(fig, ax, widen=1.3)
    _dress(fig, ax, spec.text("benchmark_bars", n_clocks=len(s),
                              n_datasets=bench.per_dataset["dataset"].nunique(),
                              alpha=alpha))
    return fig, s


def benchmark_error_bias(bench):
    """MedAE against MedE, sized by the benchmark total.

    Neither axis is a ranking, and that is the point of drawing them together.
    A clock at the origin predicts chronological age well, which is the one
    thing a useful clock does not have to do.
    """
    s = bench.summary_table.reset_index()
    _require_signal(s["MedAE"], what="benchmark_error_bias", min_n=1,
                    allow_constant=True)
    fig, ax = _new()
    sizes = 40 + 90 * s["total"] / max(s["total"].max(), 1e-9)
    ax.scatter(s["MedAE"], s["MedE"], s=sizes, color=palette()[0],
               alpha=theme_value("point_alpha"), edgecolor="none")
    for _, r in s.iterrows():
        ax.annotate(r["clock"], (r["MedAE"], r["MedE"]), fontsize=6.5,
                    xytext=(3, 3), textcoords="offset points", color="#555555")
    # Room on the right for the last point's name, which would otherwise run
    # past the axes into the legend.
    ax.margins(x=0.14, y=0.08)
    _ref(ax, "h", 0.0, label="no bias")
    # Three reference sizes spanning the benchmark totals, so the point size
    # can be read without the description.
    top = float(max(s["total"].max(), 1e-9))
    marks = sorted({round(top * f, 1) for f in (0.25, 0.5, 1.0)} - {0.0})
    handles = [ax.scatter([], [], s=40 + 90 * m / top, color=palette()[0],
                          alpha=theme_value("point_alpha"), edgecolor="none",
                          label=f"{m:g}") for m in marks]
    from matplotlib.lines import Line2D

    handles.append(Line2D([], [], color=semantic("reference"), lw=theme_value("line_width"),
                          ls=theme_value("reference_line"), label="no bias"))
    _legend_outside(fig, ax, title="benchmark total", widen=1.25, handles=handles)
    _dress(fig, ax, spec.text("benchmark_error_bias", n_clocks=len(s)))
    return fig, s


def benchmark_heatmap(bench):
    """Effect size per clock and dataset, with significant cells outlined."""
    plt = _mpl()
    d = bench.per_dataset
    if d.empty:
        raise NothingToPlot("benchmark_heatmap: no comparisons")
    m = d.pivot_table(index="clock", columns="dataset", values="delta", aggfunc="mean")
    _require_signal(m.to_numpy(), what="benchmark_heatmap deltas", min_n=1)
    sig = d.pivot_table(index="clock", columns="dataset", values="significant",
                        aggfunc="max").reindex_like(m).fillna(False)

    p = palette("diverging")
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("falcon_div", [p["low"], p["mid"], p["high"]])

    fig, ax = _new(width=0.7 * m.shape[1] + 3.4, height=0.28 * m.shape[0] + 2.2)
    lim = float(np.nanpercentile(np.abs(m.to_numpy()), 95)) or 1.0
    im = ax.imshow(m.to_numpy(), aspect="auto", cmap=cmap, vmin=-lim, vmax=lim)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if bool(sig.iat[i, j]):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor="black", lw=1.4))
    ax.set_xticks(range(m.shape[1]), m.columns, rotation=90,
                  fontsize=theme_value("caption_size"))
    ax.set_yticks(range(m.shape[0]), m.index, fontsize=theme_value("caption_size"))
    fig.colorbar(im, ax=ax, shrink=0.7, label="delta (years)")
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(facecolor="white", edgecolor="black", lw=1.4,
                             label="significant (BH)")],
              frameon=False, fontsize=theme_value("caption_size"), loc="lower left",
              bbox_to_anchor=(0.0, 1.01), borderaxespad=0.0)
    ax.grid(False)
    _dress(fig, ax, spec.text("benchmark_heatmap", n_clocks=m.shape[0],
                              n_datasets=m.shape[1]))
    return fig, m

