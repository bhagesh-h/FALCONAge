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
from typing import Sequence

import numpy as np
import pandas as pd

from ..core.errors import AnalysisError
from . import spec
from .spec import group_colours, palette, platform_colour, semantic, theme_value

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

# Backwards-compatible alias: the correlation heatmap used to be the only
# cross-clock figure.
PALETTE = palette("categorical")


def _mpl():
    try:
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise AnalysisError(
            "plotting needs matplotlib: pip install matplotlib\n"
            "  Every plot function also returns the data frame behind the figure, "
            "so you can render it yourself without the dependency."
        ) from exc
    return plt


def _new(width: float | None = None, height: float | None = None):
    plt = _mpl()
    return plt.subplots(figsize=(width or theme_value("width"),
                                 height or theme_value("height")),
                        dpi=theme_value("dpi"))


def _dress(fig, ax, t: dict[str, str]) -> None:
    """Apply the shared theme and the figure's own text.

    The description goes on the figure, not the axes, so it survives being
    cropped to the plotting area -- which is what a screenshot does.
    """
    ax.set_title(t["title"], fontsize=theme_value("title_size"), loc="left", pad=18)
    if t["subtitle"]:
        ax.text(0, 1.02, t["subtitle"], transform=ax.transAxes,
                fontsize=theme_value("subtitle_size"), color="#555555", va="bottom")
    ax.set_xlabel(t["xlab"], fontsize=theme_value("base_size"))
    ax.set_ylabel(t["ylab"], fontsize=theme_value("base_size"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=theme_value("grid_alpha"), linewidth=0.5)
    ax.tick_params(labelsize=theme_value("base_size") - 1.5)
    if t["description"]:
        # Wrapped by hand and placed inside the figure, not below it. A negative
        # y coordinate puts the caption outside the canvas, where
        # bbox_inches="tight" rescues it but every other consumer -- a PDF page,
        # a slide, anything honouring the declared figure size -- crops it away.
        import textwrap

        cap = "\n".join(textwrap.wrap(" ".join(t["description"].split()), width=96))
        n_lines = cap.count("\n") + 1
        fig.tight_layout(rect=(0, 0.035 + 0.028 * n_lines, 1, 1))
        fig.text(0.012, 0.012, cap, fontsize=theme_value("caption_size"),
                 color="#666666", va="bottom", ha="left")
    else:
        fig.tight_layout()


class NothingToPlot(AnalysisError):
    """Raised when a figure would be empty, flat, or all zeros.

    Drawn anyway, these are worse than absent: an axis-free rectangle in a
    report reads as a measurement of nothing rather than as an absence of
    measurement, and a heatmap of a constant is a solid block of whatever colour
    sits at the midpoint of the scale. Every entry point raises this and
    :func:`save_all` records it as a skip with its reason, so the gap in the
    output is legible instead of mysterious.
    """


def _require_signal(values, *, what: str, min_n: int = 2,
                    allow_constant: bool = False):
    """Refuse to draw when there is nothing in the data to see."""
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size < min_n:
        raise NothingToPlot(f"{what}: {v.size} finite value(s), need at least {min_n}")
    if not allow_constant:
        if np.allclose(v, 0.0):
            raise NothingToPlot(f"{what}: every value is zero")
        if np.ptp(v) == 0:
            raise NothingToPlot(f"{what}: every value is {v[0]:g}; nothing varies")
    return v


def _wrap_caption(text: str, width: int = 92) -> str:
    """One place that decides where a caption breaks.

    Figures that size their own canvas need the line count *before* they can
    create the figure, so the wrap cannot live inside the call that draws it.
    """
    import textwrap

    return "\n".join(textwrap.wrap(" ".join((text or "").split()), width=width))


def _figure_text(fig, t, cap: str | None = None) -> None:
    """Title, subtitle and caption, offset in inches rather than in fractions.

    Fractions move when a figure grows: the gap that separates title from
    subtitle on a 7-inch canvas opens up by half again on a 10-inch one, so a
    figure that sizes itself to its contents would also resize its header.
    """
    h = fig.get_figheight()
    fig.text(0.012, 1 - 0.22 / h, t["title"], fontsize=theme_value("title_size"),
             va="top", ha="left")
    fig.text(0.012, 1 - 0.56 / h, t["subtitle"], fontsize=theme_value("subtitle_size"),
             color="#555555", va="top", ha="left")
    fig.text(0.012, 0.012, cap if cap is not None else _wrap_caption(t["description"]),
             fontsize=theme_value("caption_size"), color="#666666",
             va="bottom", ha="left")


def _titles_outside(fig, t, *, top: float = 0.86):
    """Title, subtitle and caption for figures whose axes fill the canvas.

    The polar and circos panels have no margin to hang text in, so the layout
    reserves it explicitly: the axes shrink to a middle band and the three text
    blocks take the strips above and below. Placing them at negative coordinates
    instead pushes them off the declared figure size.
    """
    cap = _wrap_caption(t["description"])
    bottom = 0.045 + 0.026 * (cap.count("\n") + 1)
    fig.subplots_adjust(top=top, bottom=bottom, left=0.04, right=0.96)
    _figure_text(fig, t, cap)


def _radial_labels(ax, angles, labels, radius, *, polar=False, size_delta=-0.5):
    """Labels that stick outward like spikes, none of them upside down.

    Tangential labels collide past about eight axes, and matplotlib's polar tick
    labels are tangential. So they are drawn one at a time along their own
    radius, anchored at the inner end so the text starts clear of the ring
    rather than crossing it, and flipped 180 degrees on the left half so every
    one reads left to right.
    """
    for a, lab in zip(angles, labels):
        deg = np.degrees(a) % 360
        flip = 90 < deg < 270
        x, y = (a, radius) if polar else (radius * np.cos(a), radius * np.sin(a))
        ax.text(x, y, lab,
                rotation=deg + 180 if flip else deg, rotation_mode="anchor",
                ha="right" if flip else "left", va="center",
                fontsize=theme_value("caption_size") + size_delta, color="#444444")


def _ref(ax, kind: str = "h", at: float = 0.0):
    fn = ax.axhline if kind == "h" else ax.axvline
    fn(at, color=semantic("reference"), lw=theme_value("line_width"),
       ls=theme_value("reference_line"), zorder=0)


def _age(result, age_col: str) -> pd.Series:
    if age_col not in result.obs.columns:
        raise AnalysisError(f"no {age_col!r} column in the sample annotation")
    return pd.to_numeric(result.obs[age_col], errors="coerce")


def _unit(result, clock: str) -> str:
    return ", ".join(result.registry.get(clock).unit) or "score"


# ===========================================================================
# per-clock accuracy and calibration
# ===========================================================================
def ba_vs_ca(result, clock: str, *, age_col: str = "age", group: str | None = None):
    """Predicted against chronological age, with the identity line.

    The identity line, not a regression line. A regression line always looks
    like a good fit; the identity line is what exposes a clock running five
    years high on everybody, which is what MedE measures and what quietly wins
    AA1 in a benchmark that does not discount for it.
    """
    _mpl()          # import, set the Agg backend, or raise
    age, y = _age(result, age_col), result.scores[clock]
    df = pd.DataFrame({"chronological": age, "predicted": y})
    if group and group in result.obs.columns:
        df["group"] = result.obs[group].astype(str)
    d = df.dropna(subset=["chronological", "predicted"])
    _require_signal(d["predicted"], what="ba_vs_ca predictions")

    fig, ax = _new()
    if "group" in d:
        cols = group_colours(sorted(d["group"].unique()))
        for g, sub in d.groupby("group"):
            ax.scatter(sub["chronological"], sub["predicted"], s=theme_value("point_size") * 8,
                       alpha=theme_value("point_alpha"), color=cols[g], label=g,
                       edgecolor="none")
        ax.legend(frameon=False, fontsize=theme_value("caption_size"))
    else:
        ax.scatter(d["chronological"], d["predicted"], s=theme_value("point_size") * 8,
                   alpha=theme_value("point_alpha"), color=palette()[0], edgecolor="none")

    lim = [float(np.nanmin(d[["chronological", "predicted"]].to_numpy())),
           float(np.nanmax(d[["chronological", "predicted"]].to_numpy()))]
    ax.plot(lim, lim, color=semantic("reference"), lw=theme_value("line_width"),
            ls=theme_value("reference_line"), zorder=0)

    r = float(d["chronological"].corr(d["predicted"])) if len(d) > 2 else np.nan
    medae = float(np.median(np.abs(d["predicted"] - d["chronological"])))
    _dress(fig, ax, spec.text("ba_vs_ca", clock=clock, n=len(d), r=f"{r:.2f}",
                              medae=f"{medae:.2f}", unit=_unit(result, clock)))
    return fig, d


def bland_altman(result, clock: str, *, age_col: str = "age"):
    """Difference against mean, with bias and 95% limits of agreement.

    A correlation coefficient cannot show that a clock's error depends on age.
    This can, and age-dependent error is the failure mode that makes a single
    MedAE meaningless: a clock accurate at 40 and eight years out at 80 reports
    the same summary statistic as one uniformly four years out.
    """
    age, y = _age(result, age_col), result.scores[clock]
    d = pd.DataFrame({"mean": (age + y) / 2.0, "diff": y - age}).dropna()
    _require_signal(d["diff"], what="bland_altman differences")

    bias = float(d["diff"].mean())
    sd = float(d["diff"].std(ddof=1))
    lo, hi = bias - 1.96 * sd, bias + 1.96 * sd

    fig, ax = _new()
    ax.scatter(d["mean"], d["diff"], s=theme_value("point_size") * 8,
               alpha=theme_value("point_alpha"), color=palette()[0], edgecolor="none")
    _ref(ax, "h", 0.0)
    for v, style in ((bias, "-"), (lo, ":"), (hi, ":")):
        ax.axhline(v, color=semantic("case"), lw=theme_value("line_width"), ls=style)
    _dress(fig, ax, spec.text("bland_altman", clock=clock, bias=f"{bias:.2f}",
                              lo=f"{lo:.1f}", hi=f"{hi:.1f}"))
    return fig, d


def calibration(result, clock: str, *, age_col: str = "age"):
    """Residual against chronological age.

    The slope is the diagnostic. A negative one means the clock over-ages the
    young and under-ages the old -- regression to the mean, the most common
    artefact in this field and the one most often reported as a finding.
    """
    age, y = _age(result, age_col), result.scores[clock]
    ok = age.notna() & y.notna()
    slope, intercept = np.polyfit(age[ok].to_numpy(float), y[ok].to_numpy(float), 1)
    d = pd.DataFrame({"chronological": age[ok],
                      "residual": y[ok] - (slope * age[ok] + intercept)})
    rslope = np.polyfit(d["chronological"].to_numpy(float),
                        d["residual"].to_numpy(float), 1)[0]

    _require_signal(d["residual"], what="calibration residuals")
    fig, ax = _new()
    ax.scatter(d["chronological"], d["residual"], s=theme_value("point_size") * 8,
               alpha=theme_value("point_alpha"), color=palette()[0], edgecolor="none")
    _ref(ax, "h", 0.0)
    xs = np.linspace(d["chronological"].min(), d["chronological"].max(), 50)
    ax.plot(xs, rslope * xs + np.mean(d["residual"]) - rslope * np.mean(d["chronological"]),
            color=semantic("case"), lw=theme_value("line_width"))
    _dress(fig, ax, spec.text("calibration", clock=clock, slope=f"{rslope:+.3f}"))
    return fig, d


# ===========================================================================
# age acceleration
# ===========================================================================
def acceleration_by_group(acc: pd.DataFrame, clock: str, obs: pd.DataFrame,
                          group: str, *, method: str | None = None):
    """Box and strip of acceleration by group -- the case/control workhorse."""
    _mpl()          # import, set the Agg backend, or raise
    d = pd.DataFrame({"acceleration": acc[clock], "group": obs[group].astype(str)}).dropna()
    _require_signal(d["acceleration"], what="acceleration")
    levels = sorted(d["group"].unique())
    if len(levels) < 2:
        raise NothingToPlot("acceleration_by_group: only one group present, so "
                            "there is no comparison to draw")
    cols = group_colours(levels)

    fig, ax = _new()
    data = [d.loc[d["group"] == g, "acceleration"].to_numpy() for g in levels]
    bp = ax.boxplot(data, positions=range(len(levels)), widths=0.55,
                    patch_artist=True, showfliers=False,
                    medianprops={"color": semantic("reference"), "lw": 1.4})
    for patch, g in zip(bp["boxes"], levels):
        patch.set_facecolor(cols[g])
        patch.set_alpha(0.35)
        patch.set_edgecolor(cols[g])
    rng = np.random.default_rng(0)
    for i, g in enumerate(levels):
        v = data[i]
        ax.scatter(i + rng.uniform(-0.13, 0.13, v.size), v, s=theme_value("point_size") * 7,
                   color=cols[g], alpha=theme_value("point_alpha"), edgecolor="none", zorder=3)
    _ref(ax, "h", 0.0)
    ax.set_xticks(range(len(levels)), [f"{g}\nn={len(data[i])}" for i, g in enumerate(levels)])
    _dress(fig, ax, spec.text("acceleration_by_group", clock=clock,
                              method=method or acc.attrs.get("method", "residual"),
                              groups=" vs ".join(levels)))
    return fig, d


def acceleration_density(acc: pd.DataFrame, clock: str, obs: pd.DataFrame | None = None,
                         group: str | None = None):
    """Histogram of acceleration, optionally split by group.

    Counted, not smoothed. A kernel density on twelve progeria cases invents a
    bimodality that is not in the data, and the figure is usually read before
    anyone checks the sample size.
    """
    d = pd.DataFrame({"acceleration": acc[clock]})
    if group and obs is not None and group in obs.columns:
        d["group"] = obs[group].astype(str)
    d = d.dropna()
    _require_signal(d["acceleration"], what="acceleration")

    fig, ax = _new(height=theme_value("height") * 0.8)
    groups = list(d.groupby("group")) if "group" in d else [("all", d)]
    cols = group_colours([g for g, _ in groups])
    for g, sub in groups:
        v = sub["acceleration"].to_numpy()
        if v.size < 2:
            continue
        ax.hist(v, bins=min(24, max(5, v.size // 2)), alpha=0.55,
                color=cols[g], label=f"{g} (n={v.size})")
    _ref(ax, "v", 0.0)
    if "group" in d:
        ax.legend(frameon=False, fontsize=theme_value("caption_size"))
    _dress(fig, ax, spec.text("acceleration_density", clock=clock, n=len(d),
                              method=acc.attrs.get("method", "residual")))
    return fig, d


def acceleration_heatmap(acc: pd.DataFrame, obs: pd.DataFrame | None = None,
                         group: str | None = None):
    """Clocks by samples, z-scored within clock.

    Z-scored per row because the clocks are on wildly different scales; without
    it one clock's variance dominates the whole colour map and the figure shows
    only that clock.
    """
    _mpl()          # import, set the Agg backend, or raise
    _require_signal(acc.to_numpy(), what="acceleration heatmap", min_n=4)
    z = acc.apply(lambda c: (c - c.mean()) / (c.std(ddof=0) or 1.0)).T
    order = z.index[np.argsort(-z.var(axis=1).to_numpy())]
    z = z.loc[order]
    if group is not None and obs is not None and group in obs.columns:
        z = z[obs[group].astype(str).sort_values().index]

    p = palette("diverging")
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("falcon_div", [p["low"], p["mid"], p["high"]])

    fig, ax = _new(width=max(6.0, 0.06 * z.shape[1] + 3),
                   height=max(3.0, 0.22 * z.shape[0] + 1.6))
    lim = float(np.nanpercentile(np.abs(z.to_numpy()), 98)) or 1.0
    im = ax.imshow(z.to_numpy(), aspect="auto", cmap=cmap, vmin=-lim, vmax=lim)
    ax.set_yticks(range(len(z)), z.index, fontsize=theme_value("caption_size"))
    ax.set_xticks([])
    fig.colorbar(im, ax=ax, shrink=0.6, label="z within clock")
    ax.grid(False)
    _dress(fig, ax, spec.text("acceleration_heatmap", n_clocks=z.shape[0],
                              n_samples=z.shape[1]))
    return fig, z


def forest(bench, *, top: int | None = None):
    """Effect size and 95% interval per clock/dataset comparison.

    The interval is what makes this honest. An eight-versus-eight comparison
    with a twenty-year point estimate has an interval wide enough to say so,
    and a bar chart of the point estimates alone would not.
    """
    d = bench.per_dataset.copy()
    if d.empty:
        raise NothingToPlot("forest: no comparisons in this benchmark")
    _require_signal(d["delta"], what="forest effect sizes", min_n=1,
                    allow_constant=True)
    # A normal-approximation interval from the rank statistic: crude, and
    # labelled as such, but it scales with n the way the point estimate does not.
    n_eff = np.sqrt(1.0 / d["n_case"].clip(lower=1) + 1.0 / d["n_control"].clip(lower=1))
    spread = d.groupby("clock")["delta"].transform(lambda s: s.abs().max() or 1.0)
    d["se"] = n_eff * spread * 0.8
    d["lo"], d["hi"] = d["delta"] - 1.96 * d["se"], d["delta"] + 1.96 * d["se"]
    d["label"] = d["clock"] + " · " + d["dataset"] + " · " + d["condition"]
    d = d.sort_values("delta")
    if top:
        d = pd.concat([d.head(top // 2), d.tail(top - top // 2)])

    fig, ax = _new(height=max(3.0, 0.24 * len(d) + 1.6))
    ys = np.arange(len(d))
    for y, (_, r) in zip(ys, d.iterrows()):
        c = semantic("case") if r["significant"] else semantic("neutral")
        ax.plot([r["lo"], r["hi"]], [y, y], color=c, lw=theme_value("line_width"))
        ax.scatter(r["delta"], y, s=theme_value("point_size") * 12, color=c, zorder=3)
    _ref(ax, "v", 0.0)
    ax.set_yticks(ys, d["label"], fontsize=theme_value("caption_size"))
    _dress(fig, ax, spec.text("forest", n=len(d)))
    return fig, d


def trajectory(result, clock: str, subject_col: str, *, age_col: str = "age"):
    """Spaghetti plot of repeated measures, one line per subject."""
    d = pd.DataFrame({"subject": result.obs[subject_col].astype(str),
                      "age": _age(result, age_col),
                      "predicted": result.scores[clock]}).dropna()
    fig, ax = _new()
    pal = palette()
    for i, (s, sub) in enumerate(d.sort_values("age").groupby("subject")):
        ax.plot(sub["age"], sub["predicted"], marker="o", ms=3.5,
                lw=theme_value("line_width"), alpha=0.75, color=pal[i % len(pal)])
    lim = [d["age"].min(), d["age"].max()]
    ax.plot(lim, lim, color=semantic("reference"), lw=1, ls=theme_value("reference_line"),
            zorder=0)
    _dress(fig, ax, spec.text("trajectory", clock=clock, n=d["subject"].nunique(),
                              unit=_unit(result, clock)))
    return fig, d


def timecourse(result, time_col: str, *, clocks: Sequence[str] | None = None,
               n_boot: int = 1000, time_label: str = "time"):
    """Scaled prediction against time, with a bootstrap band per clock.

    Scaled within clock so curves on different scales are comparable, which is
    the only way to put a mortality log-hazard and an age in years on one panel.
    Read the shape and where curves cross, never the height.
    """
    cols = list(clocks) if clocks else list(result.scores.columns)
    t = pd.to_numeric(result.obs[time_col], errors="coerce")
    rng = np.random.default_rng(0)
    rows = []
    for c in cols:
        z = (result.scores[c] - result.scores[c].mean()) / (result.scores[c].std(ddof=0) or 1)
        for tv, idx in z.groupby(t).groups.items():
            v = z.loc[idx].dropna().to_numpy()
            if v.size == 0:
                continue
            boot = rng.choice(v, size=(min(n_boot, 1000), v.size), replace=True).mean(axis=1)
            rows.append({"clock": c, "time": tv, "mean": float(v.mean()),
                         "lo": float(np.percentile(boot, 2.5)),
                         "hi": float(np.percentile(boot, 97.5))})
    d = pd.DataFrame(rows)

    fig, ax = _new()
    pal = palette()
    for i, (c, sub) in enumerate(d.groupby("clock")):
        sub = sub.sort_values("time")
        ax.plot(sub["time"], sub["mean"], lw=theme_value("line_width"),
                color=pal[i % len(pal)], label=c)
        ax.fill_between(sub["time"], sub["lo"], sub["hi"], alpha=0.16,
                        color=pal[i % len(pal)], linewidth=0)
    ax.legend(frameon=False, fontsize=theme_value("caption_size"), ncol=2)
    _dress(fig, ax, spec.text("timecourse", n_clocks=d["clock"].nunique(),
                              time_label=time_label))
    return fig, d


# ===========================================================================
# cross-clock
# ===========================================================================
def _corr(result, method: str = "spearman") -> pd.DataFrame:
    return result.scores.corr(method=method)


def clock_corr(result, *, method: str = "spearman", cluster: bool = True):
    """Correlation heatmap, hierarchically ordered.

    Rank correlation, not Pearson: clocks on different scales have no meaningful
    linear correlation, and mixing scales is the normal case. Clustering the
    rows is what makes the blocks visible, and the blocks are usually shared
    training cohorts rather than shared biology.
    """
    _mpl()          # import, set the Agg backend, or raise
    if result.scores.shape[1] < 2:
        raise NothingToPlot("clock_corr: needs at least two clocks")
    m = _corr(result, method)
    _require_signal(m.to_numpy()[np.triu_indices(len(m), 1)],
                    what="clock_corr off-diagonal", min_n=1, allow_constant=True)
    if cluster and m.shape[0] > 2:
        try:
            from scipy.cluster.hierarchy import leaves_list, linkage
            from scipy.spatial.distance import squareform
            dist = squareform(np.clip(1 - m.to_numpy(), 0, 2), checks=False)
            order = leaves_list(linkage(dist, method="average"))
            m = m.iloc[order, order]
        except Exception:
            pass

    p = palette("diverging")
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("falcon_div", [p["low"], p["mid"], p["high"]])

    n = len(m)
    fig, ax = _new(width=0.34 * n + 3.2, height=0.34 * n + 2.4)
    im = ax.imshow(m.to_numpy(), vmin=-1, vmax=1, cmap=cmap)
    ax.set_xticks(range(n), m.columns, rotation=90, fontsize=theme_value("caption_size"))
    ax.set_yticks(range(n), m.index, fontsize=theme_value("caption_size"))
    fig.colorbar(im, ax=ax, shrink=0.65, label=method)
    ax.grid(False)
    _dress(fig, ax, spec.text("clock_corr", n_clocks=n, n=result.scores.shape[0]))
    return fig, m


def clock_scatter_matrix(result, clocks: Sequence[str] | None = None):
    """Every pair of clocks against each other, on the rank scale."""
    plt = _mpl()
    cols = list(clocks) if clocks else list(result.scores.columns)[:6]
    r = result.scores[cols].rank(pct=True)
    n = len(cols)
    fig, axes = plt.subplots(n, n, figsize=(1.5 * n + 1.2, 1.5 * n + 1.4),
                             dpi=theme_value("dpi"), squeeze=False)
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            ax = axes[i][j]
            if i == j:
                ax.hist(r[a].dropna(), bins=14, color=palette()[0], alpha=0.7)
            else:
                ax.scatter(r[b], r[a], s=5, alpha=0.6, color=palette()[0], edgecolor="none")
                ax.plot([0, 1], [0, 1], color=semantic("reference"), lw=0.7,
                        ls=theme_value("reference_line"))
            ax.set_xticks([]), ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(a, fontsize=theme_value("caption_size") - 1, rotation=0,
                              ha="right", va="center")
            if i == n - 1:
                ax.set_xlabel(b, fontsize=theme_value("caption_size") - 1, rotation=45,
                              ha="right")
    t = spec.text("clock_scatter_matrix", n_clocks=n)
    import textwrap

    cap = "\n".join(textwrap.wrap(" ".join(t["description"].split()), width=92))
    fig.tight_layout(rect=(0.02, 0.055 + 0.022 * (cap.count("\n") + 1), 1, 0.90))
    fig.text(0.012, 0.975, t["title"], fontsize=theme_value("title_size"),
             va="top", ha="left")
    fig.text(0.012, 0.935, t["subtitle"], fontsize=theme_value("subtitle_size"),
             color="#555555", va="top", ha="left")
    fig.text(0.012, 0.012, cap, fontsize=theme_value("caption_size"),
             color="#666666", va="bottom", ha="left")
    return fig, r


def _embed(result, method: str = "pca"):
    X = result.scores.apply(lambda c: (c - c.mean()) / (c.std(ddof=0) or 1.0)).fillna(0.0)
    if method == "umap":
        try:
            import umap
            emb = umap.UMAP(n_neighbors=min(15, max(2, len(X) - 1)),
                            random_state=0).fit_transform(X.to_numpy())
            return pd.DataFrame(emb, index=X.index, columns=["d1", "d2"]), (np.nan, np.nan)
        except ImportError:
            method = "pca"
    Xc = X.to_numpy() - X.to_numpy().mean(axis=0)
    u, s, _ = np.linalg.svd(Xc, full_matrices=False)
    var = (s**2) / max((s**2).sum(), 1e-12) * 100
    return (pd.DataFrame(u[:, :2] * s[:2], index=X.index, columns=["d1", "d2"]),
            (float(var[0]), float(var[1])))


def clock_pca(result, *, colour_by: str | None = None, method: str = "pca"):
    """Samples embedded in clock space.

    PC1 is almost always chronological age; that is expected and is not a
    finding. Structure on PC2 that tracks a plate or a scan date and not a
    phenotype is a batch effect, and this is where it shows up.
    """
    d, (pc1, pc2) = _embed(result, method)
    fig, ax = _new()
    if colour_by and colour_by in result.obs.columns:
        v = result.obs[colour_by]
        if pd.api.types.is_numeric_dtype(v):
            sc = ax.scatter(d["d1"], d["d2"], c=pd.to_numeric(v, errors="coerce"),
                            s=theme_value("point_size") * 9, cmap="viridis",
                            edgecolor="none")
            fig.colorbar(sc, ax=ax, shrink=0.7, label=colour_by)
        else:
            cols = group_colours(sorted(v.astype(str).unique()))
            for g, idx in v.astype(str).groupby(v.astype(str)).groups.items():
                ax.scatter(d.loc[idx, "d1"], d.loc[idx, "d2"], label=g, color=cols[g],
                           s=theme_value("point_size") * 9, alpha=theme_value("point_alpha"),
                           edgecolor="none")
            ax.legend(frameon=False, fontsize=theme_value("caption_size"))
    else:
        ax.scatter(d["d1"], d["d2"], s=theme_value("point_size") * 9,
                   color=palette()[0], edgecolor="none")
    _dress(fig, ax, spec.text("clock_pca", pc1=f"{pc1:.0f}", pc2=f"{pc2:.0f}"))
    return fig, d


def clock_embedding(result, clock: str, *, method: str = "pca"):
    """One embedding, recoloured by a single clock's prediction.

    The pyaging figure: fix the sample layout, then recolour it per clock.
    Clocks whose gradients point in different directions are ordering the same
    samples differently, which a correlation matrix reports as a number and this
    shows as a picture.
    """
    d, _ = _embed(result, method)
    d = d.assign(value=result.scores[clock])
    _require_signal(d["value"], what=f"clock_embedding {clock}")
    fig, ax = _new()
    sc = ax.scatter(d["d1"], d["d2"], c=d["value"], s=theme_value("point_size") * 9,
                    cmap="viridis", edgecolor="none")
    fig.colorbar(sc, ax=ax, shrink=0.7, label=_unit(result, clock))
    _dress(fig, ax, spec.text("clock_embedding", clock=clock, method=method.upper(),
                              n=len(d)))
    return fig, d


# ===========================================================================
# quality control
# ===========================================================================
def beta_density(data, *, max_samples: int = 60, n_grid: int = 200):
    """Per-sample beta distribution.

    Two peaks near 0 and 1 is what a working array looks like. A sample with a
    filled middle failed, and no amount of downstream normalisation recovers it
    -- which is why this is the first figure to look at and not the last.
    """
    X = data.X
    if X.shape[0] > max_samples:
        X = X.sample(max_samples, random_state=0)
    rows = {}
    for sid, row in X.iterrows():
        v = row.to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if v.size < 50:
            continue
        # Histogram-based density: a Gaussian KDE on a bimodal beta distribution
        # smears the two modes into one and hides exactly the failure this plot
        # exists to catch.
        h, edges = np.histogram(v, bins=n_grid, range=(0, 1), density=True)
        rows[sid] = h
    if not rows:
        raise NothingToPlot("beta_density: no sample has 50 finite values")
    d = pd.DataFrame(rows, index=(edges[:-1] + edges[1:]) / 2)

    fig, ax = _new()
    for i, c in enumerate(d.columns):
        ax.plot(d.index, d[c], lw=0.8, alpha=0.6,
                color=platform_colour(data.platform))
    _dress(fig, ax, spec.text("beta_density", n=d.shape[1],
                              platform=data.platform or "unknown platform"))
    return fig, d


def coverage_bar(result, *, floor: float = 0.8, platform: str | None = None):
    """Per-clock feature coverage. The plot to read before the scores."""
    d = pd.DataFrame([{"clock": k, "coverage": v.get("coverage"),
                       "n_imputed": v.get("n_imputed", 0)}
                      for k, v in result.coverage.items()])
    d = d.dropna(subset=["coverage"]).sort_values("coverage")
    if d.empty:
        raise AnalysisError("no coverage recorded for this result")

    colours = [semantic("fail") if c < floor else
               semantic("warn") if c < 0.95 else semantic("pass") for c in d["coverage"]]
    fig, ax = _new(height=max(2.6, 0.24 * len(d) + 1.5))
    ax.barh(d["clock"], d["coverage"], color=colours)
    _ref(ax, "v", floor)
    ax.set_xlim(0, 1)
    ax.tick_params(axis="y", labelsize=theme_value("caption_size"))
    _dress(fig, ax, spec.text("coverage_bar", n_clocks=len(d), floor=int(floor * 100),
                              platform=platform or "mixed"))
    return fig, d


def missingness(data):
    """Distribution of per-sample missingness."""
    d = pd.DataFrame({"missing_fraction": data.X.isna().mean(axis=1)})
    _require_signal(d["missing_fraction"], what="missingness")
    fig, ax = _new(height=theme_value("height") * 0.8)
    ax.hist(d["missing_fraction"], bins=30, color=palette()[0], alpha=0.8)
    _dress(fig, ax, spec.text("missingness", n=len(d),
                              median=f"{100 * d['missing_fraction'].median():.2f}"))
    return fig, d


_X_PROBES = ["cg12653510", "cg05533223", "cg03691818", "cg26355737", "cg09516963"]


def sex_check(data, *, sex_col: str = "sex"):
    """Declared sex against X-linked probe methylation.

    A point in the wrong cluster is a sample sheet error, not biology. Worth
    running before any sex-stratified analysis, because mislabelled sex is the
    single most common metadata error in public series.
    """
    probes = [p for p in _X_PROBES if p in data.X.columns]
    if len(probes) < 3:
        raise AnalysisError(
            f"only {len(probes)} of the {len(_X_PROBES)} X-linked check probes are "
            "present; the sex check needs at least three")
    d = pd.DataFrame({"x_beta": data.X[probes].mean(axis=1)})
    d["declared"] = (data.obs[sex_col].astype(str).str.upper().str[0]
                     if sex_col in data.obs.columns else "U")

    cols = {"M": palette()[0], "F": palette()[1], "U": semantic("neutral")}
    fig, ax = _new(height=theme_value("height") * 0.85)
    rng = np.random.default_rng(0)
    for g, sub in d.groupby("declared"):
        ax.scatter(sub["x_beta"], rng.uniform(-0.4, 0.4, len(sub)),
                   color=cols.get(g, semantic("neutral")), label=g,
                   s=theme_value("point_size") * 9, alpha=theme_value("point_alpha"),
                   edgecolor="none")
    ax.set_yticks([])
    ax.legend(frameon=False, fontsize=theme_value("caption_size"))

    known = d[d["declared"].isin(["M", "F"])]
    mismatch = 0
    if known["declared"].nunique() == 2:
        med = known.groupby("declared")["x_beta"].median()
        cut = float(med.mean())
        pred = np.where((known["x_beta"] < cut) == (med["M"] < med["F"]), "M", "F")
        mismatch = int((pred != known["declared"].to_numpy()).sum())
        _ref(ax, "v", cut)
    _dress(fig, ax, spec.text("sex_check", n=len(d), mismatch=mismatch))
    return fig, d


def platform_comparison(result, clock: str, *, platform_col: str = "platform"):
    """Score distribution split by array platform."""
    return _split_box(result, clock, platform_col, "platform_comparison",
                      n_platforms_key="n_platforms", colour_map=platform_colour)


def study_comparison(result, clock: str, *, dataset_col: str = "dataset"):
    """Score distribution split by study."""
    return _split_box(result, clock, dataset_col, "study_comparison",
                      n_platforms_key="n_studies")


def _split_box(result, clock, col, plot_name, *, n_platforms_key, colour_map=None):
    if col not in result.obs.columns:
        raise AnalysisError(f"no {col!r} column in the sample annotation")
    d = pd.DataFrame({"value": result.scores[clock],
                      "split": result.obs[col].astype(str)}).dropna()
    _require_signal(d["value"], what=f"{plot_name} values")
    levels = sorted(d["split"].unique())
    if len(levels) < 2:
        raise NothingToPlot(f"{plot_name}: only one level of {col!r}, so there "
                            "is nothing to compare across")
    cols = ({lv: colour_map(lv) for lv in levels} if colour_map
            else group_colours(levels))

    fig, ax = _new(width=max(5.0, 0.8 * len(levels) + 2.6))
    data = [d.loc[d["split"] == lv, "value"].to_numpy() for lv in levels]
    bp = ax.boxplot(data, positions=range(len(levels)), widths=0.6, patch_artist=True,
                    showfliers=False, medianprops={"color": semantic("reference")})
    for patch, lv in zip(bp["boxes"], levels):
        patch.set_facecolor(cols[lv]), patch.set_alpha(0.4), patch.set_edgecolor(cols[lv])
    rng = np.random.default_rng(0)
    for i, lv in enumerate(levels):
        ax.scatter(i + rng.uniform(-0.14, 0.14, data[i].size), data[i],
                   s=theme_value("point_size") * 6, color=cols[lv], alpha=0.8,
                   edgecolor="none", zorder=3)
    ax.set_xticks(range(len(levels)),
                  [f"{lv}\nn={len(data[i])}" for i, lv in enumerate(levels)],
                  fontsize=theme_value("caption_size"))
    fields = {"clock": clock, "n": len(d), "unit": _unit(result, clock),
              n_platforms_key: len(levels)}
    _dress(fig, ax, spec.text(plot_name, **fields))
    return fig, d


# ===========================================================================
# benchmark
# ===========================================================================
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
    ax.legend(frameon=False, fontsize=theme_value("caption_size"))
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
    _ref(ax, "h", 0.0)
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
    ax.grid(False)
    _dress(fig, ax, spec.text("benchmark_heatmap", n_clocks=m.shape[0],
                              n_datasets=m.shape[1]))
    return fig, m


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


# ===========================================================================
# circos and radar
# ===========================================================================
def clock_chord(result, *, min_shared: int = 5, max_clocks: int = 24):
    """Circos-style chord diagram of CpG sharing between clocks.

    The correlation heatmap says two clocks agree. This says how much of that
    agreement is built in: chord width is the number of CpGs the two clocks have
    literally in common. A pair with a thick chord and a high correlation has
    told you much less than a pair with a high correlation and no chord at all.

    Following the circos convention of the single-cell aging literature -- an
    outer ring of colour-coded entities, chords in the interior weighted by the
    strength of the relationship.
    """
    _mpl()          # import, set the Agg backend, or raise
    reg = result.registry
    clocks = [c for c in result.scores.columns if reg.has_coefficients(c)][:max_clocks]
    feats = {c: set(reg.feature_ids(c)) for c in clocks}

    pairs = []
    for i, a in enumerate(clocks):
        for b in clocks[i + 1:]:
            n = len(feats[a] & feats[b])
            if n >= min_shared:
                pairs.append({"a": a, "b": b, "shared": n,
                              "jaccard": n / max(len(feats[a] | feats[b]), 1)})
    d = pd.DataFrame(pairs)

    if len(d) == 0:
        raise NothingToPlot(
            f"clock_chord: no clock pair shares {min_shared} or more CpGs, so "
            "there are no chords to draw")
    n = len(clocks)
    ang = {c: 2 * np.pi * i / n for i, c in enumerate(clocks)}
    pal = palette()
    fig, ax = _new(width=7.0, height=7.0)
    ax.set_aspect("equal")
    ax.axis("off")

    # outer ring, then the labels as one call so their geometry lives in a
    # single place shared with the radar.
    for i, c in enumerate(clocks):
        a0, a1 = ang[c] - np.pi / n * 0.86, ang[c] + np.pi / n * 0.86
        t = np.linspace(a0, a1, 24)
        ax.plot(np.cos(t), np.sin(t), lw=7, color=pal[i % len(pal)],
                solid_capstyle="butt")
    _radial_labels(ax, [ang[c] for c in clocks], clocks, 1.06)

    if len(d):
        wmax = d["shared"].max()
        for _, r in d.sort_values("shared").iterrows():
            a, b = ang[r["a"]], ang[r["b"]]
            p0, p1 = np.array([np.cos(a), np.sin(a)]), np.array([np.cos(b), np.sin(b)])
            # Quadratic Bezier through a control point pulled toward the centre
            # by the angular distance, which is what gives circos its
            # characteristic inward-bowing chords.
            sep = abs(((a - b + np.pi) % (2 * np.pi)) - np.pi) / np.pi
            ctrl = (p0 + p1) / 2 * (1 - 0.85 * sep)
            t = np.linspace(0, 1, 60)[:, None]
            curve = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * ctrl + t**2 * p1
            ax.plot(curve[:, 0], curve[:, 1],
                    lw=0.4 + 3.6 * r["shared"] / wmax,
                    color=pal[clocks.index(r["a"]) % len(pal)],
                    alpha=0.18 + 0.5 * r["shared"] / wmax, solid_capstyle="round")

    # Room for the longest label at its outward extent, estimated from the
    # name rather than fixed: a registry with longer ids would otherwise crop
    # silently.
    pad = 1.10 + 0.030 * max(len(c) for c in clocks)
    ax.set_xlim(-pad, pad), ax.set_ylim(-pad, pad)
    t = spec.text("clock_chord", n_clocks=n, n_chords=len(d), min_shared=min_shared)
    _titles_outside(fig, t)
    return fig, d


def clock_radar(result, *, group: str | None = None, clocks: Sequence[str] | None = None,
                max_clocks: int = 12):
    """Closed-polygon profile across clocks, one polygon per group.

    Z-scored within clock against the cohort, because the axes are otherwise in
    incompatible units and the polygon would be a picture of the scales rather
    than of the samples.
    """
    plt = _mpl()
    cols = list(clocks) if clocks else list(result.scores.columns)[:max_clocks]
    if len(cols) < 3:
        raise NothingToPlot("clock_radar: a polygon needs at least three axes")
    _require_signal(result.scores[cols].to_numpy(), what="clock_radar scores", min_n=3)
    z = result.scores[cols].apply(lambda c: (c - c.mean()) / (c.std(ddof=0) or 1.0))

    if group and group in result.obs.columns:
        prof = z.groupby(result.obs[group].astype(str)).median()
        label = f"{prof.shape[0]} groups by {group}"
    else:
        prof = pd.DataFrame([z.median()], index=["all samples"])
        label = "cohort median"

    n = len(cols)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    closed = np.concatenate([ang, ang[:1]])
    t = spec.text("clock_radar", n_clocks=n, label=label)
    names = [str(g) for g in prof.index]

    # ---- layout, computed in inches ---------------------------------------
    # Four things want the same space and each needs an amount only the data
    # knows: the spike labels (longest clock id), the polygon, the legend
    # (how many levels, and the longest level name) and the caption (how it
    # wraps). A fixed canvas divided into fixed fractions has to put two of
    # them in the same place, which is how the legend came to sit on top of
    # the left-hand spikes. So the canvas is sized to its contents and each
    # band gets a strip of its own: header, then the square that holds the
    # circle and its spikes, then the legend, then the caption.
    fs = theme_value("caption_size")
    char = 0.60 * (fs - 0.5) / 72.0            # width of one character, inches
    r_in = 2.05                                # radius of the polygon itself

    # A spike label runs along its own radius, so how far it reaches to the
    # left is its length times |cos|, and upward its length times |sin|. Each
    # side of the circle therefore needs only what the labels pointing that way
    # actually use. Reserving the longest label on all four sides instead --
    # the easy version -- leaves the bottom of a twelve-axis figure three
    # quarters of an inch emptier than it has to be.
    reach = np.array([char * len(c) + 0.12 for c in cols])
    bulk = 0.5 * (fs + 2.0) / 72.0             # the text's own line height
    pad = [float(np.max(np.maximum(reach * v, 0.0))) + bulk
           for v in (-np.cos(ang), np.cos(ang), -np.sin(ang), np.sin(ang))]
    pad_l, pad_r, pad_b, pad_t = pad
    box_w, box_h = 2 * r_in + pad_l + pad_r, 2 * r_in + pad_b + pad_t

    entry = char * max(len(g) for g in names) + 0.55   # swatch, gap, text
    ncol = max(1, min(len(names), int((box_w - 0.3) // entry)))
    nrow = int(np.ceil(len(names) / ncol))
    legend_in = 0.12 + nrow * (fs + 5.0) / 72.0

    cap = _wrap_caption(t["description"])
    cap_in = 0.16 + (cap.count("\n") + 1) * (fs + 3.0) / 72.0
    head_in = 0.92
    width = max(box_w, 6.4)
    height = head_in + box_h + legend_in + cap_in

    fig = plt.figure(figsize=(width, height), dpi=theme_value("dpi"))
    ax = fig.add_subplot(111, projection="polar")
    cmap = group_colours(list(prof.index))

    for g, row in prof.iterrows():
        v = np.concatenate([row.to_numpy(), row.to_numpy()[:1]])
        ax.plot(closed, v, lw=theme_value("line_width") * 1.4, color=cmap[g], label=g)
        ax.fill(closed, v, color=cmap[g], alpha=0.14)

    ax.plot(closed, np.zeros_like(closed), color=semantic("reference"), lw=1,
            ls=theme_value("reference_line"))

    # matplotlib's polar tick labels sit tangentially and collide past about
    # eight axes, so they are suppressed and redrawn as radial spikes.
    ax.set_xticks(ang)
    ax.set_xticklabels([])
    rmax = float(np.nanmax(np.abs(prof.to_numpy()))) * 1.15 or 1.0
    ax.set_ylim(-rmax, rmax)
    _radial_labels(ax, ang, cols, rmax * 1.06, polar=True)
    # The r-axis numbers default to theta=0, which is exactly where a spoke and
    # its spike label already are. Halfway between the first two spokes is free
    # at every n.
    ax.set_rlabel_position(180.0 / n)
    ax.tick_params(axis="y", labelsize=fs - 1, pad=0)
    # The radial axis is only r_in inches long however wide the z range is, so
    # the default locator crowds its labels into each other on a tight cohort.
    # Ask for as many ticks as fit at a third of an inch apart, no more.
    from matplotlib.ticker import MaxNLocator

    ax.yaxis.set_major_locator(
        MaxNLocator(nbins=max(3, int(r_in / 0.34)), steps=[1, 2, 2.5, 5, 10]))

    # The axes box is the diameter exactly -- a polar axes inscribes its circle
    # in its own box -- so the spike allowance is what lies outside it.
    ax.set_position([((width - box_w) / 2 + pad_l) / width,
                     (cap_in + legend_in + pad_b) / height,
                     2 * r_in / width, 2 * r_in / height])
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center",
               bbox_to_anchor=(0.5, (cap_in + 0.04) / height), ncol=ncol,
               frameon=False, fontsize=fs, columnspacing=1.4,
               handlelength=1.5, handletextpad=0.5, borderaxespad=0.0)
    _figure_text(fig, t, cap)
    return fig, prof


# ---------------------------------------------------------------------------
# The pooled atlas is large enough to deserve its own module, and is imported
# last because it imports the helpers defined above.
# ---------------------------------------------------------------------------
from .atlas import clock_atlas  # noqa: E402


# ---------------------------------------------------------------------------
# survival and association
# ---------------------------------------------------------------------------
def _km_curve(time: np.ndarray, event: np.ndarray):
    """Kaplan-Meier estimator. Returns step points and the number at risk.

    Twenty lines and no new dependency. The product-limit form is
    ``S(t) = prod over event times t_i <= t of (1 - d_i / n_i)`` where ``n_i``
    is the number still at risk just before ``t_i`` and ``d_i`` the number of
    events at it. Censored observations leave the risk set without an event,
    which is the whole reason this is not one minus a cumulative proportion.
    """
    order = np.argsort(time, kind="stable")
    t, e = np.asarray(time)[order], np.asarray(event)[order].astype(bool)

    times, surv, at_risk = [0.0], [1.0], [len(t)]
    s, n = 1.0, len(t)
    for ti in np.unique(t):
        at = t == ti
        d = int(e[at].sum())
        if d:
            s *= 1.0 - d / n
            times.append(float(ti))
            surv.append(s)
            at_risk.append(n)
        n -= int(at.sum())
    return np.array(times), np.array(surv), np.array(at_risk)


def _logrank(t1, e1, t2, e2) -> float:
    """Two-sample log-rank p-value, from the standard O-E statistic."""
    t = np.concatenate([t1, t2])
    e = np.concatenate([e1, e2]).astype(bool)
    g = np.concatenate([np.zeros(len(t1), bool), np.ones(len(t2), bool)])

    o_minus_e, var = 0.0, 0.0
    for ti in np.unique(t[e]):
        at_risk = t >= ti
        n, n1 = int(at_risk.sum()), int((at_risk & ~g).sum())
        d = int((e & (t == ti)).sum())
        d1 = int((e & (t == ti) & ~g).sum())
        if n < 2 or d == 0:
            continue
        exp1 = d * n1 / n
        o_minus_e += d1 - exp1
        var += (d * (n1 / n) * (1 - n1 / n) * (n - d)) / (n - 1)
    if var <= 0:
        return float("nan")
    from scipy import stats as _st
    return float(_st.chi2.sf(o_minus_e ** 2 / var, df=1))


def kaplan_meier(result, clock: str, *, time_col: str, event_col: str,
                 age_col: str = "age", quantile: float = 0.1):
    """Survival of the fastest-ageing tail against the slowest.

    The convention in the literature is the extreme deciles rather than a
    median split -- the middle of the acceleration distribution is where a
    clock discriminates least, and pooling it into two halves dilutes whatever
    signal the tails carry.

    Parameters
    ----------
    quantile
        Size of each tail. 0.1 gives top and bottom 10%, the published default.
    """
    from ..analysis import acceleration

    aa = acceleration(result, age_col=age_col, clocks=[clock])[clock]
    d = pd.DataFrame({
        "aa": aa,
        "time": pd.to_numeric(result.obs[time_col], errors="coerce"),
        "event": pd.to_numeric(result.obs[event_col], errors="coerce"),
    }).dropna()

    if len(d) < 8:
        raise NothingToPlot(
            f"kaplan_meier: {len(d)} subjects with acceleration, time and "
            "event; need at least 8 to split into tails")
    if not d["event"].astype(bool).any():
        raise NothingToPlot(
            f"kaplan_meier: no events in {event_col!r}; every subject is "
            "censored and there is no survival curve to draw")

    lo_cut, hi_cut = d["aa"].quantile([quantile, 1 - quantile])
    slow, fast = d[d["aa"] <= lo_cut], d[d["aa"] >= hi_cut]

    fig, ax = _new()
    for sub, key, label in ((slow, "control", f"slowest {quantile:.0%}"),
                            (fast, "case", f"fastest {quantile:.0%}")):
        ts, ss, _ = _km_curve(sub["time"].to_numpy(), sub["event"].to_numpy())
        ax.step(ts, ss, where="post", color=semantic(key),
                lw=theme_value("line_width"), label=f"{label} (n={len(sub)})")

    p = _logrank(slow["time"].to_numpy(), slow["event"].to_numpy(),
                 fast["time"].to_numpy(), fast["event"].to_numpy())

    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=theme_value("caption_size"), loc="lower left")
    _dress(fig, ax, spec.text("kaplan_meier", clock=clock, n=len(d),
                              events=int(d["event"].sum()),
                              p="< 0.001" if p < 1e-3 else f"{p:.3f}"))
    return fig, pd.DataFrame({"group": ["slow", "fast"],
                              "n": [len(slow), len(fast)],
                              "events": [int(slow["event"].sum()),
                                         int(fast["event"].sum())],
                              "logrank_p": [p, p]})


def volcano(assoc, *, effect: str = "beta", p: str = "p", fdr: float = 0.05,
            label_top: int = 10):
    """Effect size against evidence, for a table from :func:`~falconage.associate`.

    The dashed line is the Benjamini-Hochberg threshold at ``fdr``, taken from
    the ``q`` column when the table has one rather than recomputed. Drawing a
    raw p-value cut instead is the common error: across thousands of tests the
    two thresholds differ by orders of magnitude, and the raw one calls noise
    significant.
    """
    d = assoc.copy()
    for col in (effect, p):
        if col not in d.columns:
            raise AnalysisError(
                f"volcano: no {col!r} column. associate() returns "
                f"{', '.join(assoc.columns)}")
    d = d[np.isfinite(d[effect]) & np.isfinite(d[p])]
    if d.empty:
        raise NothingToPlot("volcano: nothing with a finite effect and p-value")

    d["_y"] = -np.log10(d[p].clip(lower=np.finfo(float).tiny))
    passing = d["q"] <= fdr if "q" in d.columns else d[p] <= fdr
    d["_hit"] = passing

    # The threshold to draw is the largest p that still passes: with BH that is
    # a property of the whole set, so it cannot be computed from one row.
    y_cut = float(d.loc[passing, "_y"].min()) if passing.any() else None

    fig, ax = _new()
    ax.scatter(d.loc[~d["_hit"], effect], d.loc[~d["_hit"], "_y"],
               s=theme_value("point_size") * 8, color=semantic("neutral"),
               alpha=0.65, edgecolors="none")
    ax.scatter(d.loc[d["_hit"], effect], d.loc[d["_hit"], "_y"],
               s=theme_value("point_size") * 12, color=semantic("case"),
               edgecolors="none")
    if y_cut is not None:
        _ref(ax, "h", y_cut)
    _ref(ax, "v", 0.0)

    for name, r in d.nlargest(label_top, "_y").iterrows():
        ax.annotate(str(name), (r[effect], r["_y"]),
                    fontsize=theme_value("caption_size") - 1,
                    xytext=(3, 3), textcoords="offset points")

    _dress(fig, ax, spec.text("volcano", n=len(d), hits=int(passing.sum()),
                              fdr=fdr))
    return fig, d.drop(columns=["_y", "_hit"])


# ---------------------------------------------------------------------------
# uncertainty
# ---------------------------------------------------------------------------
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
    fig, ax = _new(height=max(3.0, 0.34 * len(d) + 1.6))
    y = np.arange(len(d))
    icc = d.get("implied_cohort_icc", pd.Series(np.nan, index=d.index))
    colours = [pal["fail"] if not np.isfinite(v) or v <= 0
               else pal["warn"] if v < 0.8 else pal["pass"] for v in icc]
    ax.barh(y, d["se_over_sd"], color=colours, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(d.index)
    for i, (v, k, raw) in enumerate(zip(d["se_over_sd"], icc, d["median_se"])):
        lab = f"{raw:.3g}" + ("" if not np.isfinite(k) else f"  ICC {k:.2f}")
        ax.text(v, i, "  " + lab, va="center", fontsize=theme_value("base_size") - 2,
                color="#444444")
    _ref(ax, "v", 1.0)
    ax.margins(x=0.30)

    t = spec.text("reliability_forest", n=len(d),
             method=", ".join(sorted(set(d["method"]))))
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
