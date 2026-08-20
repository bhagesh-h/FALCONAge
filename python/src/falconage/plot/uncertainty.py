"""How much a score can be trusted. Added in this release."""

from __future__ import annotations

from typing import Sequence  # noqa: F401

import numpy as np
import pandas as pd

from ..core.errors import AnalysisError  # noqa: F401
from . import spec
from .spec import (  # noqa: F401
    group_colours, palette, platform_colour, semantic, theme_value,
)
from ._common import (  # noqa: F401
    NothingToPlot, PALETTE, _age, _dress, _figure_text, _mpl, _new,
    _radial_labels, _ref, _require_signal, _titles_outside, _unit,
    _wrap_caption,
)


#: Where the ratio stops being worth drawing to scale. One is already the
#: point at which the assay noise equals the spread between these samples, so
#: a clock at two cannot tell them apart twice over; past that the exact figure
#: is a diagnostic rather than something to compare by eye.
_BREAK_AT = 2.0

#: And how far past the rest of the field one bar has to reach before breaking
#: the axis is worth the discontinuity.
_BREAK_RATIO = 3.0


def reliability_forest(se, *, top: int | None = None):
    """Technical standard error per clock, with the ICC that cohort implies.

    The figure that says which of the numbers beside it can be trusted to move.
    A clock whose measurement error is a large share of the spread between these
    samples cannot detect a small difference between two of them, however
    confident the point estimate looks.

    Takes the :class:`~falconage.uncertainty.SEResult` from
    :func:`falconage.technical_se`.
    """
    d = se.diagnostics.copy()
    if d.empty:
        raise NothingToPlot("reliability_forest: no clock had a usable reliability source")
    if "se_over_sd" not in d.columns:
        raise NothingToPlot("reliability_forest: needs a cohort to scale against")
    # Noise as a fraction of the cohort's own spread, not the raw SE. The clocks
    # here report years, kilobases, pace ratios and unitless scores; putting
    # their standard errors on one axis would be the units error the rest of
    # this package refuses to commit, drawn instead of computed. The raw SE and
    # its unit go on the label, where they cannot be compared by eye.
    d = d.dropna(subset=["se_over_sd"]).sort_values("se_over_sd")
    if top:
        d = d.tail(top)
    _require_signal(d["se_over_sd"], what="reliability_forest se_over_sd", min_n=1,
                    allow_constant=True)

    pal = palette("semantic")
    icc = d.get("implied_cohort_icc", pd.Series(np.nan, index=d.index))
    colours = [pal["fail"] if not np.isfinite(v) or v <= 0
               else pal["warn"] if v < 0.8 else pal["pass"] for v in icc]
    vals = d["se_over_sd"].to_numpy(dtype=float)
    y = np.arange(len(d))

    # ---- one bar can flatten every other -----------------------------------
    # The axis is a ratio and it is unbounded above: a clock whose samples
    # barely differ divides a real standard error by almost nothing. HypoClock
    # reaches 55 on the shipped corpus and epiTOC2 reaches four figures, while
    # every clock that can tell two samples apart sits under 1. Drawn on one
    # scale the useful bars are slivers a millimetre wide, which is the
    # opposite of what this figure is for.
    #
    # So the long bars are cut at the break and marked with an ellipsis, and
    # their real value is on the label. The scale stays linear and honest for
    # the bars a reader is comparing; the ones that run off it say how far.
    readable = vals[vals <= _BREAK_AT]
    over = vals > _BREAK_AT
    breaking = bool(over.any() and (~over).any()
                    and vals.max() > _BREAK_RATIO * (readable.max()
                                                     if readable.size else 1.0))
    # Never below 1.05: the reference line at 1.0 is where the assay noise
    # equals the spread between these samples, and a scale that cuts it off
    # removes the one mark that says which side of useful a bar is on.
    limit = (max(float(readable.max()) * 1.10, 1.05) if breaking and readable.size
             else float(vals.max()))
    drawn = np.minimum(vals, limit) if breaking else vals

    fig, ax = _new(height=max(3.0, 0.34 * len(d) + 1.6))
    ax.barh(y, drawn, color=colours, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(d.index)
    _ref(ax, "v", 1.0)

    base = theme_value("base_size")
    for i, (v, k, raw, cut) in enumerate(zip(vals, icc, d["median_se"], over)):
        lab = f"{raw:.3g}" + ("" if not np.isfinite(k) else f"  ICC {k:.2f}")
        if breaking and cut:
            # The ellipsis sits where the bar stops, in the bar's own colour,
            # so it reads as a continuation rather than as punctuation in the
            # label. The ratio follows, because that is the number the bar can
            # no longer show.
            ax.text(limit, i, "...", va="center", ha="left", fontsize=base + 1,
                    color=colours[i], fontweight="bold")
            lab = f"{v:.3g}x off scale  ·  {lab}"
            ax.text(limit, i, "      " + lab, va="center", fontsize=base - 2,
                    color="#444444")
        else:
            ax.text(v, i, "  " + lab, va="center", fontsize=base - 2,
                    color="#444444")

    if breaking:
        ax.set_xlim(0, limit)
        # Room for the label, which now runs past the bar it belongs to.
        ax.margins(x=0.55)
    else:
        ax.margins(x=0.30)

    t = spec.text("reliability_forest", n=len(d),
             method=", ".join(sorted(set(d["method"]))))
    if breaking:
        t = dict(t)
        t["subtitle"] += f"  ·  {int(over.sum())} cut at {limit:.2g}, marked ..."
    _dress(fig, ax, t)
    return fig, d


def score_interval(result, clock: str, *, se=None, conformal=None,
                   age_col: str = "age", max_samples: int = 40):
    """One clock's scores with their intervals, sample by sample.

    The plot that changes what a score means. Each sample is a point with a bar;
    two samples whose bars overlap are not distinguishable by this clock on this
    assay, whatever their point estimates say.

    Pass ``se`` from :func:`falconage.technical_se` for measurement error, or
    ``conformal`` from :func:`falconage.conformal_interval` for prediction
    error, or both -- they are drawn as an inner and an outer bar, because they
    answer different questions and conflating them is the mistake the pair
    exists to prevent.
    """
    if se is None and conformal is None:
        raise NothingToPlot("score_interval: pass se= or conformal=")
    v = result.scores[clock].astype(float)
    order = _age(result, age_col).sort_values().index if age_col in result.obs.columns \
        else v.sort_values().index
    order = [i for i in order if i in v.index][:max_samples]
    v = v.loc[order]
    x = np.arange(len(v))

    pal = palette("semantic")
    fig, ax = _new(height=4.4)
    if conformal is not None:
        c = conformal[conformal["clock"] == clock].set_index("sample_id").reindex(order)
        ax.vlines(x, c["lo"], c["hi"], color=pal["neutral"], linewidth=4, alpha=0.35,
                  label="prediction interval")
    if se is not None:
        s = (se.se[clock] if hasattr(se, "se") else se[clock]).reindex(order).astype(float)
        ax.vlines(x, v - 1.96 * s, v + 1.96 * s, color=pal["case"], linewidth=1.6,
                  label="technical 95%")
    ax.plot(x, v, "o", markersize=theme_value("point_size") + 0.6,
            color=palette("categorical")[0], zorder=3, label="score")
    if age_col in result.obs.columns:
        ax.plot(x, _age(result, age_col).reindex(order), color=pal["reference"],
                linewidth=1.0, linestyle="dashed", label="chronological age")
    ax.set_xticks([])
    ax.legend(frameon=False, fontsize=theme_value("base_size") - 2, ncol=2)

    t = spec.text("score_interval", clock=clock, n=len(v), unit=_unit(result, clock))
    _dress(fig, ax, t)
    return fig, pd.DataFrame({"score": v})


def platform_bias(*, clocks: Sequence[str] | None = None, top: int = 14):
    """What probe loss costs each clock, in years, on each platform.

    Read straight off ``registry/data/platform_bias.csv``, which is measured
    rather than assumed: full 450K matrices masked down to each platform's probe
    set and re-scored. A bar at zero is a clock that lost nothing; a long bar is
    a clock whose missing probes were the ones it leans on.
    """
    from ..preprocess import _load_platform_bias

    tab = _load_platform_bias()
    if not tab:
        raise NothingToPlot(
            "platform_bias: no measurement table; run python/tools/build_platform_bias.py")
    rows = [{"clock": c, "platform": p, **v} for (c, p), v in tab.items()]
    d = pd.DataFrame(rows)
    d = d[d["unit"].str.contains("year", case=False, na=False)]
    if clocks:
        d = d[d["clock"].isin(list(clocks))]
    if d.empty:
        raise NothingToPlot("platform_bias: nothing on an age scale to draw")

    keep = (d.groupby("clock")["median_shift"].apply(lambda s: s.abs().max())
            .sort_values(ascending=False).head(top).index)
    d = d[d["clock"].isin(keep)]
    plats = sorted(d["platform"].unique())
    order = (d.groupby("clock")["median_shift"].apply(lambda s: s.abs().max())
             .sort_values().index)

    cat = palette("categorical")
    fig, ax = _new(height=max(3.2, 0.42 * len(order) + 1.6))
    h = 0.8 / len(plats)
    for k, plat in enumerate(plats):
        sub = d[d["platform"] == plat].set_index("clock").reindex(order)
        y = np.arange(len(order)) + (k - (len(plats) - 1) / 2) * h
        ax.barh(y, sub["median_shift"].fillna(0), height=h * 0.92,
                color=cat[k % len(cat)], label=plat)
        ax.hlines(y, sub["ci_lo"], sub["ci_hi"], color="#333333", linewidth=0.9)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order)
    _ref(ax, "v", 0.0)
    ax.legend(frameon=False, fontsize=theme_value("base_size") - 2, ncol=len(plats))

    t = spec.text("platform_bias", n=len(order), platforms=", ".join(plats))
    _dress(fig, ax, t)
    return fig, d


def consensus_plot(report, *, alpha: float = 0.05):
    """The multi-clock verdict, drawn.

    Effect size per clock, coloured by generation, with the two correction
    thresholds marked. The point of the figure is the shape rather than any one
    bar: a real change lights up across generations, and a single lit bar among
    twenty dark ones is the published signature of a false positive.
    """
    d = report.table.copy()
    if d.empty:
        raise NothingToPlot("consensus_plot: no clock was testable")
    d = d.sort_values("cohens_d")
    gens = sorted(set(d["generation"]))
    cat = palette("categorical")
    cmap = {g: cat[i % len(cat)] for i, g in enumerate(gens)}

    fig, ax = _new(height=max(3.2, 0.34 * len(d) + 1.8))
    y = np.arange(len(d))
    ax.barh(y, d["cohens_d"], color=[cmap[g] for g in d["generation"]], height=0.62)
    for i, (sig_b, sig_h) in enumerate(zip(d["sig_bonferroni"], d["sig_bh"])):
        mark = "**" if sig_b else ("*" if sig_h else "")
        if mark:
            ax.text(d["cohens_d"].iloc[i], i, "  " + mark, va="center",
                    fontsize=theme_value("base_size"), color="#222222")
    ax.set_yticks(y)
    ax.set_yticklabels(d.index)
    _ref(ax, "v", 0.0)
    handles = [__import__("matplotlib").patches.Patch(color=cmap[g], label=g)
               for g in gens]
    # "best" rather than a fixed corner: which corner is free depends entirely
    # on the effect sizes, and a legend over the bars defeats the figure.
    ax.legend(handles=handles, frameon=False, loc="best",
              fontsize=theme_value("base_size") - 2, ncol=min(len(gens), 4))

    t = spec.text("consensus_plot", verdict=report.verdict, n=report.n_tests,
             alpha=f"{alpha:g}")
    _dress(fig, ax, t)
    return fig, d
