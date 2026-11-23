"""A single self-contained HTML report.

Self-contained means one file: inlined CSS, base64 figures, the scores table
embedded. A report that references ``figures/ba_vs_ca.png`` stops working the
moment somebody emails it, which is the only thing anybody does with a report.
"""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path

import pandas as pd

# Rules only. The colours and fonts are the interface tokens in
# colorscheme.yaml, shared with the Quarto report and the documentation site,
# so this file names roles (var(--accent)) and never spells a hex value.
RULES = """
body { font:15px/1.6 var(--font-sans); color:var(--ink); background:var(--bg);
       max-width:60rem; margin:2rem auto; padding:0 1.25rem; }
h1 { font-size:1.6rem; margin:0 0 .25rem; }
h2 { font-size:1.15rem; margin:2rem 0 .5rem; border-bottom:1px solid var(--line);
     padding-bottom:.3rem; }
a { color:var(--accent); }
.sub { color:var(--muted); margin:0 0 1.5rem; }
table { border-collapse:collapse; width:100%; font-size:.85rem; }
th,td { text-align:left; padding:.35rem .6rem; border-bottom:1px solid var(--line); }
th { font-weight:600; color:var(--muted); font-size:.78rem; text-transform:uppercase;
     letter-spacing:.03em; }
td.num { text-align:right; font-variant-numeric:tabular-nums; }
.scroll { overflow-x:auto; }
.warn { color:var(--warn); }
.kv { display:grid; grid-template-columns:12rem 1fr; gap:.2rem 1rem; font-size:.88rem; }
.kv dt { color:var(--muted); }
.kv dd { margin:0; font-variant-numeric:tabular-nums; }
img { max-width:100%; height:auto; }
figure { margin:1rem 0 1.5rem; }
figcaption { font-size:.88rem; color:var(--muted); max-width:60rem; line-height:1.45; margin-top:.35rem; }
code { font-family:var(--font-mono); background:var(--panel); padding:.1rem .3rem;
       border-radius:3px; font-size:.85em; }
footer { margin-top:3rem; color:var(--muted); font-size:.8rem; }
"""


def css() -> str:
    """The report stylesheet: interface tokens, then the rules that use them."""
    from ..plot.spec import ui_css_variables

    return ui_css_variables() + RULES


def _table(df: pd.DataFrame, max_rows: int = 60) -> str:
    d = df.head(max_rows)
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in [d.index.name or ""] + list(d.columns))
    rows = []
    for idx, r in d.iterrows():
        cells = "".join(
            f'<td class="num">{v:.4g}</td>' if isinstance(v, (int, float)) and pd.notna(v)
            else f"<td>{html.escape(str(v))}</td>" for v in r)
        rows.append(f"<tr><td><code>{html.escape(str(idx))}</code></td>{cells}</tr>")
    more = (f'<p class="sub">showing {max_rows} of {len(df)} rows</p>'
            if len(df) > max_rows else "")
    return f'<div class="scroll"><table><tr>{head}</tr>{"".join(rows)}</table></div>{more}'


def _caption(title: str, subtitle: str, description: str) -> str:
    out = f"<figcaption><b>{html.escape(title)}.</b>"
    if subtitle:
        out += f" {html.escape(subtitle)}."
    if description:
        out += f" {html.escape(' '.join(description.split()))}"
    return out + "</figcaption>"


def _fig(fig) -> str:
    """The figure as an inline PNG, its title and description as an HTML caption.

    The PNG carries no title or caption of its own (see
    :func:`falconage.plot.save_figure`), so the report supplies them here.
    """
    from ..plot import figure_text

    t = figure_text(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=140, facecolor="white")
    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:  # noqa: BLE001
        pass
    return ('<figure><img alt="' + html.escape(t["title"]) + '" src="data:image/png;base64,'
            + base64.b64encode(buf.getvalue()).decode() + '">'
            + _caption(t["title"], t["subtitle"], t["description"]) + "</figure>")


def _png_with_txt(png: Path) -> str:
    """An already-written figure, captioned from the ``.txt`` beside it."""
    img = base64.b64encode(png.read_bytes()).decode()
    from ..plot._common import read_caption

    t = read_caption(png)
    title = t["title"] or png.stem.replace("_", " ").capitalize()
    sub, desc = t["subtitle"], t["description"]
    return (f'<figure><img alt="{html.escape(title)}" src="data:image/png;base64,{img}">'
            + _caption(title, sub, desc) + "</figure>")


def write_report(result, path: str | Path, *, age_col: str = "age",
                 group: str | None = None, title: str = "FALCONAge report",
                 coverage_map: str | Path | None = None) -> Path:
    """Render one self-contained HTML file for a scoring run.

    ``coverage_map`` is the PNG of :func:`falconage.plot.clock_coverage_manhattan`,
    shown first, before the scores, as it was drawn before them.
    """
    m = result.manifest
    parts: list[str] = []

    parts.append(f"<h1>{html.escape(title)}</h1>")
    parts.append(f'<p class="sub">{result.scores.shape[0]} samples &times; '
                 f'{result.scores.shape[1]} clocks &middot; '
                 f'FALCONAge {m.falconage_version} &middot; registry {m.registry_version}</p>')

    if coverage_map and Path(coverage_map).is_file():
        parts.append("<h2>Clock coverage across the genome</h2>"
                     '<p class="sub">Drawn before any clock was scored.</p>'
                     + _png_with_txt(Path(coverage_map)))

    parts.append("<h2>Run</h2><dl class='kv'>")
    for k, v in (("started", m.started_utc), ("finished", m.finished_utc or ""),
                 ("precision", m.compute_summary()),
                 ("caller", m.caller), ("python", m.python),
                 ("clocks scored", len(result.scores.columns)),
                 ("clocks skipped", len(result.skipped)),
                 ("warnings", len(m.warnings))):
        parts.append(f"<dt>{k}</dt><dd>{html.escape(str(v))}</dd>")
    parts.append("</dl>")

    parts.append("<h2>Scores</h2>" + _table(result.summary().round(4)))

    if m.warnings:
        parts.append("<h2>Warnings</h2><ul>")
        for w in m.warnings[:40]:
            who = f"<code>{html.escape(w['clock'])}</code> " if w.get("clock") else ""
            parts.append(f'<li class="warn">{who}{html.escape(w["message"])}</li>')
        parts.append("</ul>")

    if result.skipped:
        parts.append("<h2>Skipped</h2>" + _table(
            pd.DataFrame({"reason": result.skipped}).rename_axis("clock")))

    # One try per figure, not one around all of them. Sharing a block means the
    # first figure that cannot be drawn silently removes every figure after it,
    # and the report still renders -- so the loss is invisible.
    def add_figure(heading: str, make):
        try:
            parts.append((f"<h2>{heading}</h2>" if heading else "") + _fig(make()))
        except Exception as exc:
            parts.append(f'<p class="sub">{html.escape(heading or "figure")} '
                         f'unavailable: {html.escape(str(exc).splitlines()[0])}</p>')

    try:
        from ..plot import ba_vs_ca, coverage_bar
    except Exception as exc:
        parts.append(f'<p class="sub">figures unavailable: {html.escape(str(exc))}</p>')
    else:
        add_figure("Coverage", lambda: coverage_bar(result)[0])
        if age_col in result.obs.columns:
            first = True
            for cid in list(result.scores.columns)[:4]:
                if "acceleration" not in result.registry.get(cid).legal_operations:
                    continue
                add_figure("Predicted against chronological age" if first else "",
                           lambda cid=cid: ba_vs_ca(result, cid, age_col=age_col,
                                                    group=group)[0])
                first = False

    parts.append("<h2>Coefficient provenance</h2>" + _table(
        pd.DataFrame(m.weights).T.rename_axis("clock")))

    parts.append(
        "<footer>Every score above carries the SHA-256 of the coefficient file it "
        "was computed from, in <code>run_manifest.json</code>. Two runs reporting "
        "the same number either used the same coefficients or the manifest says "
        "they did not.</footer>")

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        f"<!doctype html><meta charset='utf-8'><title>{html.escape(title)}</title>"
        f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<style>{css()}</style>{''.join(parts)}",
        encoding="utf-8", newline="\n")
    return p
