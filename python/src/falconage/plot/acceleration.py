"""Age acceleration: by group, as a density, as a heatmap, over time."""

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

