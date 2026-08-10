"""Shared plumbing for every figure: canvas, theming, captions, guards.

Private to the plot package. This was the first two hundred lines of
``plot/__init__.py``. It lives here so that file can be what a package
initialiser should be --- the public surface --- rather than the implementation.
Nothing here draws a figure and nothing here picks a colour; both come from
:mod:`falconage.plot.spec`.
"""

from __future__ import annotations

from typing import Sequence  # noqa: F401

import numpy as np
import pandas as pd

from ..core.errors import AnalysisError
from .spec import (  # noqa: F401
    group_colours, palette, platform_colour, semantic, theme_value,
)

#: Backwards-compatible alias: the correlation heatmap used to be the only
#: cross-clock figure.
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
    title_size = theme_value("title_size")
    sub_size = theme_value("subtitle_size")
    base = theme_value("base_size")

    # Title and subtitle are both offset from the top of the axes in POINTS.
    #
    # The subtitle used to sit at y=1.02 in axes coordinates, which is 2% of the
    # axes *height* -- a constant only if every figure is the same height. The
    # forest plot's axes are 11 inches tall, so 2% came to 16pt while the
    # title's pad was 18pt, and the two printed on top of each other. Anything
    # positioned as a fraction of the data area will collide on some figure
    # eventually; points will not.
    gap = 5.0
    if t["subtitle"]:
        ax.annotate(t["subtitle"], xy=(0, 1), xycoords="axes fraction",
                    xytext=(0, gap), textcoords="offset points",
                    fontsize=sub_size, color="#555555", va="bottom", ha="left")
        pad = gap + sub_size * 1.35 + gap
    else:
        pad = gap
    ax.set_title(t["title"], fontsize=title_size, loc="left", pad=pad)

    ax.set_xlabel(t["xlab"], fontsize=base)
    ax.set_ylabel(t["ylab"], fontsize=base)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=theme_value("grid_alpha"), linewidth=0.5)
    ax.tick_params(labelsize=base - 1.5)
    if t["description"]:
        # Wrapped by hand and placed inside the figure, not below it. A negative
        # y coordinate puts the caption outside the canvas, where
        # bbox_inches="tight" rescues it but every other consumer -- a PDF page,
        # a slide, anything honouring the declared figure size -- crops it away.
        #
        # The space reserved for it is computed from the line height in inches
        # rather than from a fixed fraction: 0.028 of a 4-inch figure is a line
        # of caption, and 0.028 of an 11-inch forest plot is three of them, so
        # the fraction over-reserved on tall figures and under-reserved on short
        # ones.
        import textwrap

        cap = "\n".join(textwrap.wrap(" ".join(t["description"].split()), width=96))
        n_lines = cap.count("\n") + 1
        line_in = theme_value("caption_size") * 1.45 / 72.0
        reserve = (0.14 + line_in * n_lines) / fig.get_figheight()
        fig.tight_layout(rect=(0, reserve, 1, 1))
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
