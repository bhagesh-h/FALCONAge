"""Survival and association: what a score predicts, not what it is."""

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

