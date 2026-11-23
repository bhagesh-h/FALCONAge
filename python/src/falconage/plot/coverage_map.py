"""Where each clock's CpGs sit on the genome, and which of them the data carries.

One horizontal track per DNA methylation clock, every track on the same
genome axis (chromosomes 1 to 22, X and Y, GRCh37/hg19 coordinates from
Illumina's manifests). A CpG the data carries is drawn in the clock's colour; a
CpG it lacks, in grey. Because the tracks share one axis, CpGs that several
clocks use line up vertically, so the overlap between clocks, including any
clock added with :func:`falconage.add_clock`, is visible at a glance. The
legend gives each clock's colour and the share of its CpGs present.

It is drawn before any clock is scored, so the coverage that decides which
clocks can run is the first thing a run shows.
"""

from __future__ import annotations

import colorsys
import functools
import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.errors import AnalysisError
from . import spec
from ._common import _dress, _mpl

__all__ = ["clock_coverage_manhattan", "clock_coverage_points", "clock_positions"]

#: GRCh37/hg19 chromosome lengths, the build of the manifest coordinates.
CHROM_LENGTHS = {
    "1": 249250621, "2": 243199373, "3": 198022430, "4": 191154276, "5": 180915260,
    "6": 171115067, "7": 159138663, "8": 146364022, "9": 141213431, "10": 135534747,
    "11": 135006516, "12": 133851895, "13": 115169878, "14": 107349540,
    "15": 102531392, "16": 90354753, "17": 81195210, "18": 78077248, "19": 59128983,
    "20": 63025520, "21": 48129895, "22": 51304566, "X": 155270560, "Y": 59373566,
}
_OFFSET = dict(zip(CHROM_LENGTHS, np.concatenate([[0], np.cumsum(list(CHROM_LENGTHS.values()))[:-1]])))
_GENOME = float(sum(CHROM_LENGTHS.values()))


@functools.lru_cache(maxsize=1)
def clock_positions() -> pd.DataFrame:
    """Chromosome and position of every catalogue clock CpG (``feature_id`` index).

    Built by ``python/tools/build_cpg_positions.py`` from Illumina's EPIC v1,
    450K and 27K manifests (GRCh37/hg19) and shipped with the package, so the
    plot needs no network. CpGs without a position are left out.
    """
    from ..registry.registry import DATA_DIR

    p = DATA_DIR / "cpg_positions.csv.gz"
    with gzip.open(p, "rt", encoding="utf-8") as fh:
        t = pd.read_csv(fh, dtype={"chr": str}, index_col="feature_id",
                        usecols=["feature_id", "chr", "pos"])
    t = t.dropna(subset=["pos"])
    t["pos"] = t["pos"].astype("int64")
    return t


def _colours(n: int) -> list[str]:
    """``n`` distinguishable colours: evenly spaced hues, alternating lightness."""
    out = []
    for i in range(n):
        h = (i * 0.618033988749895) % 1.0
        light = 0.42 if i % 2 == 0 else 0.55
        r, g, b = colorsys.hls_to_rgb(h, light, 0.75)
        out.append(f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}")
    return out


def _present_set(data, platform: str | None) -> tuple[set[str], str]:
    if data is not None:
        X = data.X
        cols = X.columns[X.notna().any(axis=0).to_numpy()]
        label = f"{data.n_samples} samples"
        if getattr(data, "platform", None):
            label += f" on {data.platform}"
        return set(map(str, cols)), label
    if platform:
        from ..preprocess import platform_probes

        probes = platform_probes(platform)
        if probes is None:
            from ..preprocess import ARRAY_BITS

            raise AnalysisError(f"unknown array {platform!r}; expected one of "
                                + ", ".join(ARRAY_BITS))
        return set(probes), f"the {platform} array"
    raise AnalysisError("pass data, or a platform to measure coverage against")


def _compute(data, platform, registry, clocks):
    from ..registry import load

    reg = registry or load()
    present, label = _present_set(data, platform)
    if clocks is None:
        clocks = [c.id for c in reg if c.data_type == "dna_methylation"]
    feats: dict[str, list[str]] = {}
    for cid in clocks:
        try:
            f = [x for x in reg.feature_ids(cid) if not x.lower().startswith("(intercept")]
        except Exception:  # noqa: BLE001 - a clock with no per-CpG feature list
            continue
        if f:
            feats[cid] = f
    if not feats:
        raise AnalysisError("no DNA methylation clock with a CpG list to draw")

    pos = clock_positions()
    if reg is not None:
        # CpGs of a user-added clock that the shipped table does not place
        missing = sorted({f for fs in feats.values() for f in fs} - set(pos.index))
        if missing and data is not None and getattr(data, "var", None) is not None:
            v = data.var
            if {"chr", "pos"} <= set(v.columns):
                extra = v.loc[v.index.intersection(missing), ["chr", "pos"]].dropna()
                pos = pd.concat([pos, extra.astype({"chr": str})])

    counts: dict[str, int] = {}
    for fs in feats.values():
        for f in set(fs):
            counts[f] = counts.get(f, 0) + 1

    order = sorted(feats, key=lambda c: (reg.get(c).origin != "registry", c))
    colours = dict(zip(order, _colours(len(order))))
    rows = []
    for cid in order:
        fs = feats[cid]
        n_present = sum(f in present for f in fs)
        placed = pos.index.intersection(fs)
        rows.append({"clock": cid, "n_features": len(fs), "n_present": n_present,
                     "coverage": round(100.0 * n_present / len(fs), 1),
                     "n_placed": len(placed),
                     "n_shared": sum(counts[f] > 1 for f in set(fs)),
                     "colour": colours[cid], "origin": reg.get(cid).origin})
    table = pd.DataFrame(rows).set_index("clock")
    return order, feats, pos, present, table, label, colours


def clock_coverage_points(data=None, *, platform: str | None = None, registry=None,
                          clocks=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The numbers behind :func:`clock_coverage_manhattan`, for drawing elsewhere.

    Returns ``(points, table)``: one row per placed CpG per clock with
    ``clock``, ``track`` (0 at the top), ``x`` (position on the concatenated
    hg19 genome) and ``present``; and the per-clock table the figure returns.
    """
    order, feats, pos, present, table, label, colours = _compute(data, platform, registry, clocks)
    parts = []
    for row, cid in enumerate(order):
        fs = [f for f in feats[cid] if f in pos.index]
        if not fs:
            continue
        q = pos.loc[fs]
        parts.append(pd.DataFrame({
            "clock": cid, "track": row,
            "x": q["pos"].to_numpy(float) + q["chr"].map(_OFFSET).to_numpy(float),
            "present": [f in present for f in fs]}))
    pts = (pd.concat(parts, ignore_index=True) if parts
           else pd.DataFrame(columns=["clock", "track", "x", "present"]))
    return pts, table


def clock_coverage_manhattan(data=None, *, platform: str | None = None, registry=None,
                             clocks=None, width: float = 13.0):
    """One genome track per clock: its CpGs, coloured where the data carries them.

    Parameters
    ----------
    data
        A :class:`~falconage.FalconData` with DNA methylation features. A CpG
        counts as present when at least one sample has a value for it.
    platform
        Without ``data``: measure coverage against an array's probe list
        (``"EPICv1"``, ``"EPICv2"``, ``"450K"``, ``"27K"``) instead.
    registry
        Defaults to the loaded registry, so clocks added with
        :func:`falconage.add_clock` get a track too.
    clocks
        Clock ids to draw; default every DNA methylation clock with per-CpG
        features, catalogue and user-added alike.

    Returns
    -------
    (Figure, DataFrame)
        The figure, and one row per clock: ``n_features``, ``n_present``,
        ``coverage`` (percent), ``n_placed`` (CpGs with a known position),
        ``n_shared`` (CpGs another drawn clock also uses), ``colour`` and
        ``origin`` (``registry`` or ``user``).
    """
    order, feats, pos, present, table, label, colours = _compute(data, platform, registry, clocks)

    plt = _mpl()
    from matplotlib.patches import Patch

    n = len(order)
    fig, ax = plt.subplots(figsize=(width, max(3.0, 0.2 * n + 1.8)),
                           dpi=spec.theme_value("dpi"))
    for i, ch in enumerate(CHROM_LENGTHS):
        if i % 2 == 0:
            ax.axvspan(_OFFSET[ch], _OFFSET[ch] + CHROM_LENGTHS[ch],
                       color="#f0f2f5", zorder=0, lw=0)
    for row, cid in enumerate(order):
        fs = [f for f in feats[cid] if f in pos.index]
        if not fs:
            continue
        p = pos.loc[fs]
        x = p["pos"].to_numpy(float) + p["chr"].map(_OFFSET).to_numpy(float)
        have = np.fromiter((f in present for f in fs), bool, len(fs))
        y = np.full(len(fs), n - 1 - row, float)
        ax.scatter(x[~have], y[~have], marker="|", s=40, linewidths=0.5,
                   color="#c4c9d2", rasterized=True, zorder=1)
        ax.scatter(x[have], y[have], marker="|", s=40, linewidths=0.7,
                   color=colours[cid], rasterized=True, zorder=2)

    ax.set_xlim(0, _GENOME)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_yticks(range(n))
    ax.set_yticklabels(list(reversed(order)), fontsize=spec.theme_value("caption_size"))
    ax.set_xticks([_OFFSET[c] + CHROM_LENGTHS[c] / 2 for c in CHROM_LENGTHS])
    ax.set_xticklabels(list(CHROM_LENGTHS), fontsize=spec.theme_value("caption_size"))
    ax.tick_params(axis="both", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    handles = [Patch(color=colours[c], label=f"{c}  {table.loc[c, 'coverage']:.1f}%"
                     + ("  (yours)" if table.loc[c, "origin"] == "user" else ""))
               for c in order]
    handles.append(Patch(color="#c4c9d2", label="CpG absent from the data"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.005, 1.0),
              frameon=False, fontsize=spec.theme_value("caption_size") - 1,
              handlelength=1.0, handleheight=0.8, borderaxespad=0.0,
              title="clock, % of CpGs present", title_fontsize=spec.theme_value("caption_size"))
    mean = float(table["coverage"].mean())
    _dress(fig, ax, spec.text("clock_coverage_manhattan", n_clocks=n, what=label,
                              mean=f"{mean:.1f}",
                              n_full=int((table["coverage"] >= 99.95).sum())))
    return fig, table


def save_coverage_manhattan(data, outdir: str | Path, *, registry=None) -> tuple[Path, pd.DataFrame]:
    """Draw the coverage map for ``data``; write ``clock_coverage_across_the_genome.png`` and ``.txt``."""
    from ._common import save_figure

    fig, table = clock_coverage_manhattan(data, registry=registry)
    return save_figure(fig, outdir), table
