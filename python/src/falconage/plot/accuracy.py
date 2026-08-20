"""Per-clock accuracy against chronological age."""

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
        # Outside the panel: a key drawn over a scatter hides exactly
        # the points a reader is looking for. See clock_pca.
        fig.set_size_inches(fig.get_figwidth() * 1.18, fig.get_figheight())
        ax.legend(frameon=False, fontsize=theme_value("caption_size"),
                  loc="upper left", bbox_to_anchor=(1.015, 1.0),
                  borderaxespad=0.0, handletextpad=0.4, labelspacing=0.55)
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
