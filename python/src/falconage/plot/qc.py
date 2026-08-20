"""Quality control, read before the scores."""

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
    # Outside the panel: a key drawn over a scatter hides exactly
    # the points a reader is looking for. See clock_pca.
    fig.set_size_inches(fig.get_figwidth() * 1.18, fig.get_figheight())
    ax.legend(frameon=False, fontsize=theme_value("caption_size"),
              loc="upper left", bbox_to_anchor=(1.015, 1.0),
              borderaxespad=0.0, handletextpad=0.4, labelspacing=0.55)

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

