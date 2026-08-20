"""Do the clocks agree with each other, and how much of that was built in."""

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
            # Outside the panel, not inside it. matplotlib's "best" location
            # scores candidate corners by how few points they cover, and on a
            # cloud that fills the panel every corner covers some: the legend
            # landed on the HGPS and IHD samples, hiding the phenotype a
            # reader is looking for. A key that can obscure the data is worth
            # a little width instead.
            fig.set_size_inches(fig.get_figwidth() * 1.18, fig.get_figheight())
            ax.legend(frameon=False, fontsize=theme_value("caption_size"),
                      loc="upper left", bbox_to_anchor=(1.015, 1.0),
                      borderaxespad=0.0, handletextpad=0.4,
                      labelspacing=0.55)
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

