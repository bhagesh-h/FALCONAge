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
    """Apply the shared theme and attach the figure's text without drawing it.

    The figure carries axis labels and its legend, nothing else. Title,
    subtitle and description go to :func:`save_figure`, which names the PNG
    after the title and writes the rest to a ``.txt`` beside it: a journal
    figure has its caption in the manuscript, not burned into the pixels, and
    a figure without a header can be placed in a panel layout as it is.
    """
    base = theme_value("base_size")
    _attach_text(fig, t)
    ax.set_xlabel(t["xlab"], fontsize=base)
    ax.set_ylabel(t["ylab"], fontsize=base)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=theme_value("grid_alpha"), linewidth=0.5)
    ax.tick_params(labelsize=base - 1.5)
    fig.tight_layout()


def _attach_text(fig, t: dict[str, str]) -> None:
    """Keep a figure's title, subtitle and description on the Figure object."""
    fig._falconage_text = {k: t.get(k, "") for k in ("title", "subtitle", "description")}


def figure_text(fig) -> dict[str, str]:
    """The title, subtitle and description a FALCONAge figure carries.

    They are not drawn on the canvas. :func:`save_figure` writes them to the
    ``.txt`` that accompanies the PNG; call this to put them in a caption of
    your own.
    """
    return dict(getattr(fig, "_falconage_text", None)
                or {"title": "", "subtitle": "", "description": ""})


def _slug(text: str) -> str:
    import re

    s = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
    return s or "figure"


def figure_filename(fig, suffix: str | None = None) -> str:
    """The file stem :func:`save_figure` uses: the title, lower case, underscores.

    ``suffix`` (a clock id, usually) is appended for figures drawn once per
    clock, so ``Predicted against chronological age`` for Horvath is
    ``predicted_against_chronological_age_horvath2013``.
    """
    stem = _slug(figure_text(fig)["title"])
    return f"{stem}_{_slug(suffix)}" if suffix else stem


def caption_text(fig) -> str:
    """The ``.txt`` written beside a figure: title, then subtitle, then description."""
    import textwrap

    t = figure_text(fig)
    parts = [t["title"].strip()]
    if t["subtitle"].strip():
        parts.append(t["subtitle"].strip())
    if t["description"].strip():
        parts.append("\n".join(textwrap.wrap(" ".join(t["description"].split()), width=78)))
    return "\n\n".join(p for p in parts if p) + "\n"


def read_caption(path) -> dict[str, str]:
    """Parse a figure's ``.txt`` back into title, subtitle and description.

    ``path`` may be the PNG or the ``.txt``. One block is a title; two are a
    title and a description; three or more, a title, a subtitle and the rest.
    """
    from pathlib import Path

    p = Path(path).with_suffix(".txt")
    out = {"title": "", "subtitle": "", "description": ""}
    if not p.is_file():
        return out
    blocks = [" ".join(b.split()) for b in p.read_text(encoding="utf-8").split("\n\n")
              if b.strip()]
    if blocks:
        out["title"] = blocks[0]
    if len(blocks) == 2:
        out["description"] = blocks[1]
    elif len(blocks) > 2:
        out["subtitle"] = blocks[1]
        out["description"] = " ".join(blocks[2:])
    return out


def save_figure(fig, outdir, *, suffix: str | None = None, close: bool = True):
    """Write ``<title>.png`` and ``<title>.txt`` into ``outdir``; return the PNG path.

    The PNG holds the axes and the legend only. The text file holds the title,
    the figure's own numbers (the subtitle) and the description, ready to paste
    into a legend.
    """
    from pathlib import Path

    d = Path(outdir)
    d.mkdir(parents=True, exist_ok=True)
    stem = figure_filename(fig, suffix)
    png = d / f"{stem}.png"
    fig.savefig(png, bbox_inches="tight", dpi=theme_value("dpi"), facecolor="white")
    (d / f"{stem}.txt").write_text(caption_text(fig), encoding="utf-8", newline="\n")
    if close:
        _mpl().close(fig)
    return png


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
    """Attach the figure's text; nothing is drawn (see :func:`save_figure`)."""
    _attach_text(fig, t)


def _titles_outside(fig, t, *, top: float = 0.97):
    """Margins for figures whose axes fill the canvas (polar, circos), and the text."""
    fig.subplots_adjust(top=top, bottom=0.04, left=0.04, right=0.96)
    _attach_text(fig, t)


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


def _ref(ax, kind: str = "h", at: float = 0.0, label: str | None = None):
    fn = ax.axhline if kind == "h" else ax.axvline
    fn(at, color=semantic("reference"), lw=theme_value("line_width"),
       ls=theme_value("reference_line"), zorder=0, label=label)


def _legend_outside(fig, ax, *, handles=None, title: str | None = None,
                    widen: float = 1.22, **kw):
    """One legend style for every figure: frameless, right of the axes.

    Outside the panel because a key drawn over the data hides exactly the
    points a reader is looking for, and the figure is widened to make room
    rather than shrinking the axes.
    """
    fig.set_size_inches(fig.get_figwidth() * widen, fig.get_figheight())
    args = dict(frameon=False, fontsize=theme_value("caption_size"), loc="upper left",
                bbox_to_anchor=(1.015, 1.0), borderaxespad=0.0, handletextpad=0.5,
                labelspacing=0.55, handlelength=1.8, alignment="left")
    if title:
        args.update(title=title, title_fontsize=theme_value("caption_size"))
    args.update(kw)
    return ax.legend(handles=handles, **args) if handles is not None else ax.legend(**args)


def _legend_above(ax, *, handles=None, ncol: int = 4, title: str | None = None, **kw):
    """A legend in a row above the axes, for tall figures with no room on the right."""
    args = dict(frameon=False, fontsize=theme_value("caption_size"), loc="lower left",
                bbox_to_anchor=(0.0, 1.01), borderaxespad=0.0, ncol=ncol,
                handletextpad=0.5, columnspacing=1.2, handlelength=1.4)
    if title:
        args.update(title=title, title_fontsize=theme_value("caption_size"),
                    alignment="left")
    args.update(kw)
    return ax.legend(handles=handles, **args) if handles is not None else ax.legend(**args)


def _age(result, age_col: str) -> pd.Series:
    if age_col not in result.obs.columns:
        raise AnalysisError(f"no {age_col!r} column in the sample annotation")
    return pd.to_numeric(result.obs[age_col], errors="coerce")


def _unit(result, clock: str) -> str:
    return ", ".join(result.registry.get(clock).unit) or "score"
