"""The pooled clock atlas: every algorithm, every study, every output, one figure.

WHAT THIS IS FOR. When several studies are scored and combined, the question a
reader actually has is not "how did clock X do" but "of these twenty-odd
algorithms, which ones are measuring aging at all". Answering that from the
other figures means holding a benchmark bar chart, an error-versus-bias scatter,
a coverage plot and ten per-study panels in your head at once. This puts them on
one shared vertical axis so the answer is a column you read down.

WHAT IT SHOWS, left to right, rows aligned on the clock:

``A  identity``     generation and output scale, as a coloured badge. Generation
                    is the single best predictor of whether a clock will detect
                    a condition, and the scale says which downstream operations
                    are even defined for its output.
``B  accuracy``     median absolute error against chronological age, on healthy
                    controls. Reported, never ranked on -- see below.
``C  bias``         median signed error, diverging from zero. This is what
                    discounts the AA1 credit in the benchmark total, because a
                    clock that over-predicts everybody makes every group look
                    accelerated.
``D  detection``    one dot per study: the case-minus-control difference in age
                    acceleration, filled when it survives BH correction, hollow
                    when it does not. The panel that carries the figure.
``E  benchmark``    AA2 and AA1 counts, stacked.
``F  coverage``     mean feature coverage across the pooled studies, with the
                    floor drawn. A clock that scored nothing because it could
                    not see its own features is a different failure from one
                    that saw everything and found nothing, and panel D alone
                    cannot tell you which happened.

WHY PANEL D IS THE POINT. A row of hollow circles clustered on zero is an
algorithm that detected nothing in any cohort. A row with filled dots pushed
right is one that did. Reading down that column ranks the catalogue by the only
criterion that matters biologically; reading across a row says whether a clock
is consistent or got lucky in one study.

WHY MedAE IS NOT A RANKING. A hypothetical clock returning chronological age
exactly would have zero error in panel B and an empty panel D. It would be the
best clock in the figure by accuracy and useless for every purpose anyone scores
a clock for. Panels B and C are diagnostics for reading D, not scores.
"""

from __future__ import annotations

import textwrap

import numpy as np
import pandas as pd

from ..core.errors import AnalysisError

__all__ = ["clock_atlas"]

#: Generation badge order, following how the field itself groups the clocks, so
#: a reader who knows the literature finds a family without the legend.
_GEN_ORDER = ["first", "second", "causal", "pace", "mitotic", "system", "other"]


def clock_atlas(result, bench, *, dataset_col: str = "dataset",
                min_datasets: int = 2, coverage_floor: float = 0.8,
                max_clocks: int = 40):
    """One figure comparing every clock across every pooled study.

    Optional by design: it needs several studies to say anything, so it is not
    part of the default figure set and :func:`falconage.plot.save_all` emits it
    only when ``dataset_col`` names at least ``min_datasets`` studies.

    Parameters
    ----------
    result
        A combined :class:`~falconage.score.FalconResult`, normally from
        :func:`falconage.score.combine`.
    bench
        The matching :class:`~falconage.analysis.BenchmarkResult`.
    min_datasets
        Refuse below this many studies. Two is the floor at which "consistent
        across cohorts" means anything.
    coverage_floor
        Drawn on panel F; use the same value the run was scored with.
    max_clocks
        Keep the highest-scoring this many, so a full catalogue still fits a
        page. Truncation is stated in the subtitle rather than silent.

    Returns
    -------
    (figure, data)
        The frame is one row per clock and carries every number drawn.
    """
    from . import (
        NothingToPlot,
        _mpl,
        _require_signal,
        group_colours,
        palette,
        semantic,
        spec,
        theme_value,
    )

    plt = _mpl()

    if dataset_col not in result.obs.columns:
        raise AnalysisError(
            f"the atlas pools studies and needs a {dataset_col!r} column naming "
            "them. falconage.score.combine() adds one.")
    datasets = sorted(result.obs[dataset_col].astype(str).unique())
    if len(datasets) < min_datasets:
        raise NothingToPlot(
            f"clock_atlas: {len(datasets)} study(ies) pooled, need at least "
            f"{min_datasets}. This figure compares clocks ACROSS cohorts; with "
            "one cohort it is the per-study panels drawn twice.")

    per = bench.per_dataset
    summary = bench.summary_table
    if per.empty:
        raise NothingToPlot("clock_atlas: the benchmark produced no comparisons")

    # ---- one row per clock -------------------------------------------------
    reg = result.registry
    clocks = [c for c in summary.index if c in result.scores.columns]
    if not clocks:
        raise NothingToPlot("clock_atlas: no scored clock appears in the benchmark")

    # Ordered by what the benchmark found, then accuracy as a tie-break, so the
    # reading order is most-informative-first rather than alphabetical. Row 0 is
    # drawn at the bottom, so ascending here puts the best clock at the top.
    ranked = summary.loc[clocks].sort_values(
        ["total", "AA2", "MedAE"], ascending=[True, True, False])
    order = ranked.index.tolist()
    truncated = max(0, len(order) - max_clocks)
    if truncated:
        order = order[truncated:]

    # Combined results key coverage as "<dataset>:<clock>"; a single-study run
    # keys it plainly. Averaging across studies is the honest summary here
    # because panel F exists to explain a missing dot in panel D.
    cov = {}
    for c in order:
        vals = [v.get("coverage") for k, v in result.coverage.items()
                if k.split(":")[-1] == c and v.get("coverage") is not None]
        cov[c] = float(np.mean(vals)) if vals else np.nan

    rows = []
    for c in order:
        e = reg.get(c)
        s = summary.loc[c]
        rows.append({
            "clock": c, "generation": e.generation, "scale_type": e.scale_type,
            "n_features": e.n_features, "availability": e.availability,
            "AA2": int(s["AA2"]), "AA1": int(s["AA1"]),
            "MedAE": float(s["MedAE"]), "MedE": float(s["MedE"]),
            "total": float(s["total"]), "coverage": cov.get(c, np.nan),
            "n_studies": int((per["clock"] == c).sum()),
            "n_significant": int(((per["clock"] == c) & per["significant"]).sum()),
        })
    d = pd.DataFrame(rows).set_index("clock").loc[order]
    _require_signal(d["MedAE"], what="clock_atlas MedAE", min_n=1,
                    allow_constant=True)

    # ---- geometry ----------------------------------------------------------
    n = len(order)
    # Inches per clock row: floored so a three-clock atlas is not a stripe, and
    # bounded above by max_clocks so a full catalogue still fits a page.
    row_h = 0.32
    height = max(5.0, n * row_h + 3.1)
    width = 13.6
    # Panel D is widest because it carries the answer; A is a badge strip.
    widths = [0.50, 1.00, 1.00, 3.25, 0.92, 0.92]

    fig = plt.figure(figsize=(width, height), dpi=theme_value("dpi"))
    # Explicit margins, not tight_layout: the clock names on the far left and
    # the caption at the bottom are drawn with fig.text(), which tight_layout
    # cannot see and would therefore crop.
    gs = fig.add_gridspec(
        1, 6, width_ratios=widths, wspace=0.17,
        left=0.145, right=0.985,
        top=1 - 1.30 / height, bottom=1.05 / height)
    axes = [fig.add_subplot(gs[0, i]) for i in range(6)]
    y = np.arange(n)

    base = theme_value("base_size")
    tick = base - 2.0
    small = theme_value("caption_size") - 0.5

    def _panel(ax, title, *, grid_axis="x"):
        ax.set_ylim(-0.7, n - 0.3)
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(labelsize=tick, length=2.5, pad=1.5)
        if grid_axis:
            ax.grid(axis=grid_axis, alpha=theme_value("grid_alpha"), linewidth=0.45)
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=base - 0.5, loc="left", pad=7, color="#333333")

    # ---- A: identity badge, and the clock names ----------------------------
    ax = axes[0]
    _panel(ax, "A  type", grid_axis="")
    gens = [g for g in _GEN_ORDER if g in set(d["generation"])]
    gcol = dict(zip(gens, palette()))
    for i, (_, r) in enumerate(d.iterrows()):
        ax.add_patch(plt.Rectangle((0.06, i - 0.33), 0.88, 0.66,
                                   facecolor=gcol.get(r["generation"], "#B0B0B0"),
                                   edgecolor="none", alpha=0.92))
        ax.text(0.5, i, r["scale_type"].replace("_", " ").replace(" years", "y"),
                ha="center", va="center", fontsize=small - 1.5, color="white")
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)

    # Names once, on the leftmost panel, rather than repeated across six
    # columns. Tier is marked only when it is not A, so the common case stays
    # quiet and a user-supplied-coefficient row is obvious.
    for i, (c, r) in enumerate(d.iterrows()):
        tier = "" if r["availability"] == "A" else f"  [{r['availability']}]"
        ax.text(-0.12, i, f"{c}{tier}", transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=base - 1.5, color="#222222")

    # ---- B: accuracy -------------------------------------------------------
    ax = axes[1]
    _panel(ax, "B  MedAE (years)")
    ax.barh(y, d["MedAE"], height=0.62, color=semantic("neutral"), alpha=0.55)
    ax.set_xlim(0, max(float(np.nanmax(d["MedAE"])) * 1.14, 1.0))
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=3))

    # ---- C: bias -----------------------------------------------------------
    ax = axes[2]
    _panel(ax, "C  MedE, signed")
    ax.barh(y, d["MedE"], height=0.62, alpha=0.78,
            color=[semantic("accelerated") if v > 0 else semantic("decelerated")
                   for v in d["MedE"]])
    ax.axvline(0, color=semantic("reference"), lw=0.9,
               ls=theme_value("reference_line"))
    lim = max(float(np.nanmax(np.abs(d["MedE"]))) * 1.16, 1.0)
    ax.set_xlim(-lim, lim)
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=3))

    # ---- D: detection, one dot per study -----------------------------------
    ax = axes[3]
    _panel(ax, "D  case - control acceleration, one dot per study   "
               "(filled = BH q < 0.05)")
    conds = sorted(per["condition"].astype(str).unique())
    ccol = group_colours(conds)
    # Alternating bands: across a wide panel the eye loses which dot belongs to
    # which row without them.
    for i in range(n):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#000000", alpha=0.03, lw=0)
    ax.axvline(0, color=semantic("reference"), lw=1.0,
               ls=theme_value("reference_line"), zorder=2)

    for i, c in enumerate(order):
        sub_rows = per[per["clock"] == c]
        if sub_rows.empty:
            continue
        # Deterministic vertical offsets, so two studies with the same delta
        # stay legible and the figure reproduces byte for byte.
        offs = (np.linspace(-0.23, 0.23, len(sub_rows)) if len(sub_rows) > 1
                else np.zeros(1))
        for off, (_, r) in zip(offs, sub_rows.iterrows()):
            sig = bool(r["significant"])
            col = ccol.get(str(r["condition"]), semantic("neutral"))
            ax.scatter(r["delta"], i + off, s=48 if sig else 30,
                       facecolor=col if sig else "none", edgecolor=col,
                       linewidths=1.15, alpha=0.95 if sig else 0.6, zorder=3)
    dmax = float(np.nanmax(np.abs(per["delta"]))) * 1.10 or 1.0
    ax.set_xlim(-dmax, dmax)
    ax.set_xlabel("difference in median acceleration (years)", fontsize=tick)

    handles = [plt.Line2D([], [], marker="o", ls="", markerfacecolor=ccol[c],
                          markeredgecolor=ccol[c], markersize=6, label=c)
               for c in conds]
    ax.legend(handles=handles, frameon=False, fontsize=small,
              ncol=min(len(conds), 7), loc="lower center",
              bbox_to_anchor=(0.5, 1.055), handletextpad=0.3, columnspacing=1.0)

    # ---- E: benchmark counts ----------------------------------------------
    ax = axes[4]
    _panel(ax, "E  datasets hit")
    ax.barh(y, d["AA2"], height=0.62, color=palette()[0], label="AA2")
    ax.barh(y, d["AA1"], height=0.62, left=d["AA2"], color=palette()[1], label="AA1")
    ax.set_xlim(0, max(float((d["AA2"] + d["AA1"]).max()) + 0.6, 1.0))
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=4))
    ax.legend(frameon=False, fontsize=small, loc="lower center",
              bbox_to_anchor=(0.5, 1.03), ncol=2, handletextpad=0.3,
              columnspacing=0.8)

    # ---- F: coverage -------------------------------------------------------
    ax = axes[5]
    _panel(ax, "F  coverage")
    cv = d["coverage"].fillna(0.0)
    ax.barh(y, cv, height=0.62, alpha=0.85,
            color=[semantic("fail") if v < coverage_floor else
                   semantic("warn") if v < 0.95 else semantic("pass") for v in cv])
    ax.axvline(coverage_floor, color=semantic("reference"), lw=0.9,
               ls=theme_value("reference_line"))
    ax.set_xlim(0, 1.03)
    ax.set_xticks([0, 0.5, 1.0])

    # Generation legend beneath the badges it explains, in the left margin the
    # clock names already occupy, so it costs no panel width.
    ghandles = [plt.Rectangle((0, 0), 1, 1, facecolor=gcol[g], edgecolor="none",
                              label=g) for g in gens]
    axes[0].legend(handles=ghandles, frameon=False, fontsize=small - 0.5,
                   loc="upper left", bbox_to_anchor=(-1.30, -0.012),
                   ncol=min(len(gens), 4), handlelength=1.0, handletextpad=0.3,
                   columnspacing=0.8)

    # ---- figure text -------------------------------------------------------
    t = spec.text("clock_atlas", n_clocks=n, n_datasets=len(datasets),
                  n_samples=result.scores.shape[0],
                  n_significant=int(per["significant"].sum()),
                  n_comparisons=len(per))
    sub = t["subtitle"]
    if truncated:
        sub += f" · {truncated} lower-scoring clock(s) not shown"
    cap = "\n".join(textwrap.wrap(" ".join(t["description"].split()), width=168))

    fig.text(0.012, 1 - 0.20 / height, t["title"],
             fontsize=theme_value("title_size") + 2, va="top", ha="left")
    fig.text(0.012, 1 - 0.52 / height, sub,
             fontsize=theme_value("subtitle_size"), color="#555555",
             va="top", ha="left")
    fig.text(0.012, 0.16 / height, cap, fontsize=theme_value("caption_size"),
             color="#666666", va="bottom", ha="left")

    return fig, d.reset_index()
