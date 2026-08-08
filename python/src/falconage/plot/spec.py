"""Loads ``colorscheme.yaml`` -- the one input that decides how figures look.

Both languages read this file. Colours, typography, and the title / subtitle /
description text of every figure live there rather than in code, so a change
lands in the Python and the R rendering at once and neither can drift from the
other.

The file is searched for in three places, in order: an explicit path, the
``FALCONAGE_COLORSCHEME`` environment variable, then the copy packaged inside
the wheel. That order lets a user restyle every figure in a report without
touching the installation.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any

import yaml

PACKAGED = Path(__file__).with_name("colorscheme.yaml")


@functools.lru_cache(maxsize=4)
def load(path: str | None = None) -> dict[str, Any]:
    p = Path(path) if path else Path(os.environ.get("FALCONAGE_COLORSCHEME", PACKAGED))
    if not p.exists():
        raise FileNotFoundError(
            f"colour scheme not found at {p}. Set FALCONAGE_COLORSCHEME to a "
            "copy of colorscheme.yaml, or reinstall the package."
        )
    doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    if doc.get("schema_version") != 1:
        raise ValueError(f"{p}: schema_version is not 1")
    doc["_path"] = str(p)
    return doc


def palette(name: str = "categorical") -> Any:
    return load()["palette"][name]


def semantic(role: str) -> str:
    return load()["palette"]["semantic"][role]


def platform_colour(name: str | None) -> str:
    return load()["palette"]["platform"].get(name or "unknown",
                                             load()["palette"]["platform"]["unknown"])


def theme_value(key: str) -> Any:
    return load()["theme"][key]


def text(plot: str, **fields: Any) -> dict[str, str]:
    """Title, subtitle, description and axis labels for one figure.

    Missing substitution fields become ``?`` rather than raising. A figure that
    cannot compute its own r-squared should still render with everything else
    intact -- losing the whole plot to a formatting error is a worse outcome
    than a subtitle with a gap in it.
    """
    spec = load()["plots"].get(plot)
    if spec is None:
        raise KeyError(f"no text defined for plot {plot!r} in colorscheme.yaml")

    class _Safe(dict):
        def __missing__(self, k):  # noqa: D105
            return "?"

    out = {}
    for k in ("title", "subtitle", "description", "xlab", "ylab"):
        v = spec.get(k, "")
        out[k] = str(v).format_map(_Safe(fields)) if v else ""
    return out


def group_colours(levels) -> dict[str, str]:
    """Assign categorical colours to group levels, with the semantic ones fixed.

    A control arm is the same blue in every figure and every study, because a
    reader comparing two panels should not have to re-read the legend. Anything
    that looks like a control -- ``HC``, ``control``, ``healthy`` -- gets the
    control colour; the rest are assigned in order.
    """
    pal = palette("categorical")
    ctrl_names = {"hc", "control", "healthy", "ctrl", "normal", "none"}
    out: dict[str, str] = {}
    i = 0
    for lv in levels:
        s = str(lv)
        if s.lower() in ctrl_names:
            out[s] = semantic("control")
        else:
            out[s] = pal[i % len(pal)]
            i += 1 if pal[i % len(pal)] != semantic("control") else 2
    return out
