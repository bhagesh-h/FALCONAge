"""One Quarto HTML report: every score, every table, every figure, interpreted.

WHY A SECOND REPORT WRITER. ``report.write_report`` produces a compact
self-contained page that survives being emailed. This one is the opposite
document: everything the run produced, with the clocks grouped the way the
responsiveness literature groups them, and each figure carrying the sentence a
reader needs in order to know what would count as a bad one.

WHY THE GROUPING IS NOT ALPHABETICAL. A results page listing forty clocks in
name order invites the reader to compare numbers that are not comparable. The
categories here are the ones Figure 1c of the TranslAGE paper uses, because they
are the axis along which clocks actually behave differently: what a clock was
trained on decides what its number means and how it responds to an intervention.
The category header says what the output means; the clock rows sit under it.

WHY THE TABLES ARE JAVASCRIPT AND NOT A QUARTO OPTION. `df-print: paged` gives
pagination and no search, and the DataTables route needs a CDN, which breaks the
moment the file is opened offline -- which is the normal way a report is read.
The enhancer below is about eighty lines, is inlined, and does the three things
asked of it: search, choose 10/50/100/all rows, collapse.
"""

from __future__ import annotations

import base64
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

__all__ = ["CATEGORIES", "write_quarto_report"]

# ---------------------------------------------------------------------------
# how the clocks are grouped, and what each group's number means
# ---------------------------------------------------------------------------
#: (key, title, what the output is, what it implies, how it is selected)
#:
#: `predicate` runs against a registry Clock. Order matters: a clock lands in
#: the first category that claims it, so the specific tests come before the
#: general ones.
CATEGORIES: tuple[dict[str, Any], ...] = (
    {
        "key": "first",
        "title": "First generation: chronological age predictors",
        "output": "An age in years, on the same scale as the birth certificate.",
        "means": (
            "These were fitted to calendar age, so they are at their best a very "
            "good estimator of something already known. The useful part is the "
            "**residual**: the difference between the prediction and the person's "
            "actual age. A clock that predicted chronological age perfectly would "
            "have no residual left and would detect nothing."),
        "implication": (
            "Read the residual, not the prediction. In intervention studies these "
            "respond least of any group, which is expected rather than a failure: "
            "two years of anything moves a level slowly."),
        "predicate": lambda c: c.generation == "first",
    },
    {
        "key": "second",
        "title": "Second generation: mortality and morbidity trained",
        "output": "A hazard, rescaled into year-like units.",
        "means": (
            "Trained on a survival-weighted composite rather than on age. The "
            "unit is years and the target never was: the number is a risk wearing "
            "the clothes of an age."),
        "implication": (
            "These respond most strongly to interventions and agree with one "
            "another when they do, which is the main finding of the "
            "responsiveness literature. If one group is going to move, it is "
            "this one."),
        "predicate": lambda c: c.generation == "second",
    },
    {
        "key": "pace",
        "title": "Third generation: pace of aging",
        "output": "A ratio: years of biological change per calendar year.",
        "means": (
            "Fitted to the *rate* of change in organ-system biomarkers tracked "
            "longitudinally. A value of 1.0 is aging at one year per year."),
        "implication": (
            "A rate changes before an accumulated level does, which is why a "
            "two-year trial can move a pace clock while every first-generation "
            "clock stays flat. **Age acceleration is undefined here**: a pace is "
            "already a rate and subtracting an age from it is a units error."),
        "predicate": lambda c: c.scale_type == "pace_ratio" or c.generation == "pace",
    },
    {
        "key": "causal",
        "title": "Causal: damage separated from adaptation",
        "output": "Two scores with no shared origin.",
        "means": (
            "Features restricted to CpGs with Mendelian-randomisation support, "
            "split into changes that damage and changes that compensate."),
        "implication": (
            "The origin does not travel between cohorts, so `predicted minus "
            "chronological` is not a quantity here. Use the residual fitted "
            "inside the dataset at hand, or a group difference."),
        "predicate": lambda c: c.generation == "causal"
        or c.scale_type == "age_years_relative",
    },
    {
        "key": "mitotic",
        "title": "Mitotic: cumulative cell divisions",
        "output": "A count of divisions, not elapsed time.",
        "means": (
            "Estimates how many times a stem-cell pool has divided. Two tissues "
            "from the same donor on the same day share a chronological age and "
            "have very different division counts."),
        "implication": (
            "Not an age and not comparable with one. Acceleration is refused "
            "because the quantity was never elapsed time."),
        "predicate": lambda c: c.scale_type == "divisions" or c.generation == "mitotic",
    },
    {
        "key": "system",
        "title": "Explainable: system and organ subscores",
        "output": "One score per physiological system, plus a composite.",
        "means": (
            "A clock with subscores says *which* component moved rather than "
            "only that something did. SystemsAge splits into eleven system ages; "
            "the GrimAge family into its plasma-protein surrogates."),
        "implication": (
            "The most informative group for mechanism, and the one that needs "
            "the most care in reading: with eleven subscores, one of them looking "
            "extreme is the expected result in a healthy person, not a finding."),
        "predicate": lambda c: c.generation == "system"
        or c.id.startswith(("systemsage", "grimage2", "pcgrimage", "dnamfitage")),
    },
    {
        "key": "composition",
        "title": "Cell composition",
        "output": "Proportions, constrained to sum to one.",
        "means": (
            "Deconvolution estimates of which cells were in the tube, not an age "
            "at all."),
        "implication": (
            "Report these beside any bulk-blood clock. A treatment that shifts "
            "leukocyte proportions shifts the clock, and an apparent "
            "rejuvenation can be a change in cell mix."),
        "predicate": lambda c: c.scale_type == "proportion",
    },
    {
        "key": "other",
        "title": "Other predictors",
        "output": "Varies; read the unit on each row.",
        "means": (
            "Exposure scores, protein surrogates, telomere length and the "
            "clocks that fit no other group."),
        "implication": (
            "Most are `relative_score`, which permits correlation and ranking "
            "and nothing else: there is no external unit to difference or "
            "average."),
        "predicate": lambda c: True,
    },
)


def categorise(clock) -> str:
    for cat in CATEGORIES:
        if cat["predicate"](clock):
            return cat["key"]
    return "other"


# ---------------------------------------------------------------------------
# what each figure is, and how to know it went wrong
# ---------------------------------------------------------------------------
#: A caption says what a figure is. These say what to look for, and what a bad
#: one looks like, which is the part a reader cannot reconstruct from the axes.
FIGURE_NOTES: dict[str, dict[str, str]] = {
    "ba_vs_ca": {
        "caption": "Predicted age against chronological age",
        "read": "Points should sit near the diagonal with a slope near one.",
        "wrong": "A slope well below one is regression to the mean, which is normal "
                 "and means the residual at the extremes is compressed. A vertical "
                 "offset that differs between groups is a batch effect until proven "
                 "otherwise.",
    },
    "bland_altman": {
        "caption": "Agreement across the age range",
        "read": "The difference against the mean, so bias and its dependence on age "
                "are separated.",
        "wrong": "A trend in this plot means the two measures disagree differently "
                 "at different ages, which a correlation coefficient would hide "
                 "entirely.",
    },
    "calibration": {
        "caption": "Residual against chronological age",
        "read": "A flat band centred on zero.",
        "wrong": "A tilt means the clock is miscalibrated on this cohort and every "
                 "age-acceleration value inherits it.",
    },
    "acceleration_group": {
        "caption": "Age acceleration by group",
        "read": "Group separation, with the spread shown rather than only the mean.",
        "wrong": "Overlapping distributions with a significant p-value means the "
                 "effect is small and the sample large. Read the effect size.",
    },
    "acceleration_density": {
        "caption": "Distribution of age acceleration",
        "read": "Roughly symmetric and centred near zero.",
        "wrong": "A shifted centre means the clock's intercept does not suit this "
                 "cohort; a bimodal shape usually means two batches.",
    },
    "acceleration_heatmap": {
        "caption": "Age acceleration across clocks and samples",
        "read": "Columns that agree indicate the signal is in the sample, not the clock.",
        "wrong": "One column disagreeing with all the others is usually a coverage "
                 "problem on that clock rather than biology.",
    },
    "forest": {
        "caption": "Effect of condition on age acceleration, per clock",
        "read": "Intervals crossing zero are not evidence of an effect.",
        "wrong": "Reading only the clocks whose intervals exclude zero, out of "
                 "forty tested, is the multiple-comparison trap this figure exists "
                 "to make visible.",
    },
    "reliability_forest": {
        "caption": "Reliability per clock",
        "read": "Technical and biological reliability side by side.",
        "wrong": "They do not track together. A clock can be excellent on split "
                 "samples and still move with a meal.",
    },
    "score_interval": {
        "caption": "Each score with its uncertainty",
        "read": "The interval, not the point.",
        "wrong": "Comparing two points whose intervals overlap heavily is comparing "
                 "noise.",
    },
    "coverage_bar": {
        "caption": "Feature coverage per clock",
        "read": "Both bars: probes present, and the share of model weight present.",
        "wrong": "High probe coverage with low weight coverage is the dangerous "
                 "case, because it looks fine and is not.",
    },
    "missingness": {
        "caption": "Missing values per sample",
        "read": "A flat low band.",
        "wrong": "A spike on a few samples usually means a failed array, and those "
                 "samples' scores are mostly imputation.",
    },
    "beta_density": {
        "caption": "Beta value distribution per sample",
        "read": "The characteristic bimodal shape, with all samples overlapping.",
        "wrong": "A sample whose curve sits apart from the rest has a normalisation "
                 "or quality problem and should not be scored.",
    },
    "clock_corr": {
        "caption": "Agreement between clocks",
        "read": "Blocks of agreement usually follow generation, not chance.",
        "wrong": "Two clocks of the same generation disagreeing points at a "
                 "coverage difference between them.",
    },
    "clock_pca": {
        "caption": "Samples in clock space",
        "read": "Whether samples separate by group once every clock is considered.",
        "wrong": "Separation along the first component that tracks batch rather "
                 "than biology is the usual disappointment here.",
    },
    "clock_atlas": {
        "caption": "Every algorithm across every pooled study",
        "read": "The whole catalogue at once, for orientation.",
        "wrong": "Not a results figure. Do not read a single cell of it as a finding.",
    },
    "volcano": {
        "caption": "Association with age acceleration",
        "read": "Effect size against significance.",
        "wrong": "Points high on the y-axis and near zero on the x-axis are "
                 "statistically significant and biologically uninteresting.",
    },
    "kaplan_meier": {
        "caption": "Survival by age acceleration",
        "read": "Separation between strata, with the numbers at risk.",
        "wrong": "Curves that separate only where few remain at risk are driven by "
                 "a handful of people.",
    },
    "benchmark_bars": {
        "caption": "Datasets detected per clock",
        "read": "How many studies each clock separated cases from controls in.",
        "wrong": "A high count from a clock with a large bias is the AA1 problem: "
                 "over-predicting everybody looks like detecting everybody.",
    },
    "benchmark_error_bias": {
        "caption": "Error against bias on healthy controls",
        "read": "Both, together: a clock can be accurate and biased.",
        "wrong": "Judging on error alone is what the bias discount exists to "
                 "correct.",
    },
    "benchmark_heatmap": {
        "caption": "Effect size per clock and dataset",
        "read": "Consistency across datasets for a given clock.",
        "wrong": "One strong dataset carrying a clock's reputation is visible here "
                 "and invisible in a summary statistic.",
    },
}


def figure_note(stem: str) -> dict[str, str]:
    return FIGURE_NOTES.get(stem, {
        "caption": stem.replace("_", " ").capitalize(),
        "read": "",
        "wrong": "",
    })


# ---------------------------------------------------------------------------
# the inlined table enhancer
# ---------------------------------------------------------------------------
TABLE_JS = r"""
// Search, row count and collapse for every table in the report, and a filter
// for the sidebar index. Written out rather than pulled from a CDN because a
// report is normally read offline, and a table that loses its search the
// moment the network is gone is worse than one that never had it.
(function () {
  function enhance(wrap) {
    const table = wrap.querySelector('table');
    if (!table) return;
    const rows = Array.from(table.tBodies[0]?.rows || []);
    if (!rows.length) return;

    const bar = document.createElement('div');
    bar.className = 'fa-tablebar';
    const search = document.createElement('input');
    search.type = 'search';
    search.placeholder = 'Search ' + rows.length + ' rows';
    search.setAttribute('aria-label', 'Search table');
    const count = document.createElement('select');
    count.setAttribute('aria-label', 'Rows to show');
    [10, 50, 100, 0].forEach(n => {
      const o = document.createElement('option');
      o.value = n; o.textContent = n === 0 ? 'All' : n;
      count.appendChild(o);
    });
    // 10 by default only when there is enough to hide; a six-row table paged
    // to ten is a control that does nothing.
    count.value = rows.length > 10 ? '10' : '0';
    const status = document.createElement('span');
    status.className = 'fa-tablecount';
    bar.append(search, count, status);
    wrap.insertBefore(bar, wrap.firstChild);

    function apply() {
      const q = search.value.trim().toLowerCase();
      const limit = parseInt(count.value, 10);
      let shown = 0, matched = 0;
      for (const r of rows) {
        const hit = !q || r.textContent.toLowerCase().includes(q);
        if (hit) matched++;
        const show = hit && (limit === 0 || shown < limit);
        if (show) shown++;
        r.style.display = show ? '' : 'none';
      }
      status.textContent = limit === 0 || matched <= shown
        ? matched + ' of ' + rows.length
        : 'showing ' + shown + ' of ' + matched + ' matched (' + rows.length + ' total)';
    }
    search.addEventListener('input', apply);
    count.addEventListener('change', apply);
    apply();
  }

  function toc() {
    const nav = document.querySelector('#TOC, nav[role="doc-toc"], .sidebar');
    if (!nav) return;
    const box = document.createElement('input');
    box.type = 'search';
    box.className = 'fa-tocfilter';
    box.placeholder = 'Filter contents';
    box.setAttribute('aria-label', 'Filter contents');
    nav.insertBefore(box, nav.firstChild);
    const links = Array.from(nav.querySelectorAll('a'));
    box.addEventListener('input', () => {
      const q = box.value.trim().toLowerCase();
      links.forEach(a => {
        const hit = !q || a.textContent.toLowerCase().includes(q);
        const li = a.closest('li') || a;
        li.style.display = hit ? '' : 'none';
      });
    });
  }

  document.addEventListener('DOMContentLoaded', function () {
    document.querySelectorAll('.fa-table').forEach(enhance);
    document.querySelectorAll('.fa-toggle > summary').forEach(s => {
      s.addEventListener('click', () => {
        const d = s.parentElement;
        // Enhance on first open: a table inside a closed <details> has no
        // layout, and measuring it there gets every column width wrong.
        if (!d.open && !d.dataset.done) {
          d.dataset.done = '1';
          setTimeout(() => d.querySelectorAll('.fa-table').forEach(enhance), 0);
        }
      });
    });
    toc();
  });
})();
"""

TABLE_CSS = r"""
.fa-tablebar { display:flex; gap:.5rem; align-items:center; margin:.4rem 0 .5rem; }
.fa-tablebar input[type=search] { flex:1 1 14rem; padding:.3rem .5rem;
  border:1px solid var(--bs-border-color,#ddd); border-radius:4px; font-size:.85rem; }
.fa-tablebar select { padding:.3rem; border:1px solid var(--bs-border-color,#ddd);
  border-radius:4px; font-size:.85rem; }
.fa-tablecount { font-size:.78rem; opacity:.7; white-space:nowrap; }
.fa-tocfilter { width:100%; box-sizing:border-box; margin:0 0 .6rem; padding:.3rem .5rem;
  border:1px solid var(--bs-border-color,#ddd); border-radius:4px; font-size:.85rem; }
.fa-toggle { border:1px solid var(--bs-border-color,#e3e3e3); border-radius:6px;
  padding:.4rem .7rem; margin:.8rem 0; }
.fa-toggle > summary { cursor:pointer; font-weight:600; font-size:.92rem; }
.fa-table { overflow-x:auto; }
.fa-table table { width:100%; font-size:.84rem; border-collapse:collapse; }
.fa-table th, .fa-table td { padding:.3rem .55rem; border-bottom:1px solid
  var(--bs-border-color,#eee); text-align:left; white-space:nowrap; }
.fa-meaning { border-left:3px solid #e06000; padding:.5rem .9rem; margin:.8rem 0;
  background:rgba(224,96,0,.05); font-size:.92rem; }
.fa-figure { margin:1.4rem 0; }
.fa-figure img { width:100%; height:auto; border:1px solid var(--bs-border-color,#eee);
  border-radius:6px; background:#fbfbfa; }
.fa-readit { font-size:.88rem; margin:.4rem 0 0; }
.fa-readit b { color:#a84800; }
"""


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _table(df: pd.DataFrame, *, title: str, note: str = "",
           open_by_default: bool = False) -> str:
    """A collapsible, searchable table. Always collapsible, so a page with
    thirty tables is navigable rather than a scroll."""
    if df is None or len(df) == 0:
        return f'<p class="fa-readit"><em>{html.escape(title)}: no rows.</em></p>\n'
    body = df.to_html(index=False, escape=True, border=0,
                      classes="fa-inner", table_id=None)
    return (
        f'<details class="fa-toggle"{" open" if open_by_default else ""} data-done="1">\n'
        f'<summary>{html.escape(title)} '
        f'<span class="fa-tablecount">({len(df)} rows)</span></summary>\n'
        + (f'<p class="fa-readit">{note}</p>\n' if note else "")
        + f'<div class="fa-table">\n{body}\n</div>\n</details>\n')


def _figure(path: Path) -> str:
    n = figure_note(path.stem)
    read = f'<p class="fa-readit"><b>What to look for.</b> {html.escape(n["read"])}</p>' if n["read"] else ""
    wrong = f'<p class="fa-readit"><b>What a bad one looks like.</b> {html.escape(n["wrong"])}</p>' if n["wrong"] else ""
    return (
        f'<figure class="fa-figure">\n'
        f'<img src="data:image/png;base64,{_b64(path)}" alt="{html.escape(n["caption"])}">\n'
        f'<figcaption><strong>{html.escape(n["caption"])}</strong></figcaption>\n'
        f'{read}\n{wrong}\n</figure>\n')


def write_quarto_report(
    result: Any = None,
    out: str | Path = "falconage-report.qmd",
    *,
    figures: Iterable[str | Path] = (),
    tables: dict[str, pd.DataFrame] | None = None,
    logo: str | Path | None = None,
    title: str = "FALCONAge results",
    registry: Any = None,
) -> Path:
    """Write the report source. Render it with ``quarto render <file>``.

    Returns the path to the ``.qmd``. Rendering is left to the caller so the
    same source can be produced on a machine with no Quarto and rendered
    elsewhere, which is the normal shape of a CI job.
    """
    from .. import registry as _registry_mod

    reg = registry or _registry_mod.load()
    out = Path(out)
    stamp = datetime.now(timezone.utc).strftime("%d %B %Y, %H:%M UTC")

    logo_html = ""
    if logo and Path(logo).exists():
        logo_html = (f'<img src="data:image/png;base64,{_b64(Path(logo))}" '
                     f'alt="FALCONAge" style="height:46px;width:auto">')

    parts: list[str] = []
    parts.append(
        "---\n"
        f'title: "{title}"\n'
        "format:\n"
        "  html:\n"
        "    theme: cosmo\n"
        "    toc: true\n"
        "    toc-location: left\n"
        "    toc-depth: 3\n"
        "    toc-title: Contents\n"
        "    embed-resources: true\n"
        "    page-layout: full\n"
        "    code-tools: false\n"
        "---\n\n")
    parts.append(f"```{{=html}}\n<style>{TABLE_CSS}</style>\n"
                 f"<script>{TABLE_JS}</script>\n```\n\n")
    parts.append(
        "```{=html}\n"
        '<div style="display:flex;align-items:center;gap:.9rem;margin:0 0 1.2rem">\n'
        f"{logo_html}\n"
        f'<div><div style="font-size:.82rem;opacity:.7">Generated {stamp}</div></div>\n'
        "</div>\n```\n\n")

    # ---- how to read this page --------------------------------------------
    parts.append(
        "## How to read this report\n\n"
        "Clocks are grouped by **what they were trained to predict**, not "
        "alphabetically, because that is what decides what a number means. Each "
        "group below states its unit and what follows from it. A score from one "
        "group is not comparable with a score from another, and the arithmetic "
        "that mixes them has no defined meaning.\n\n"
        "Every table collapses, searches, and shows 10, 50, 100 or all rows. "
        "Every figure carries what to look for and what a bad one looks like.\n\n")

    # ---- the clock catalogue, by category ----------------------------------
    parts.append("## Clocks by category\n\n")
    scored = set()
    if result is not None and getattr(result, "scores", None) is not None:
        scored = set(map(str, result.scores.columns))

    for cat in CATEGORIES:
        members = [c for c in reg if categorise(c) == cat["key"]]
        if not members:
            continue
        here = [c for c in members if c.id in scored] if scored else members
        parts.append(f"### {cat['title']}\n\n")
        parts.append(
            '```{=html}\n<div class="fa-meaning">\n'
            f'<p><strong>What the number is.</strong> {html.escape(cat["output"])}</p>\n'
            f'<p>{cat["means"]}</p>\n'
            f'<p><strong>What follows from it.</strong> {cat["implication"]}</p>\n'
            "</div>\n```\n\n")
        df = pd.DataFrame([{
            "clock": c.id,
            "scored here": "yes" if c.id in scored else "",
            "unit": ", ".join(c.unit) or "",
            "scale": c.scale_type,
            "legal operations": ", ".join(sorted(c.legal_operations)),
            "availability": c.availability,
            "features": c.n_features or "",
            "year": c.year or "",
        } for c in sorted(members, key=lambda x: x.id)])
        note = (f"{len(here)} of these {len(members)} were scored in this run."
                if scored else f"{len(members)} catalogued.")
        parts.append("```{=html}\n" + _table(df, title=f"{cat['title']}: catalogue",
                                             note=note) + "```\n\n")

    # ---- results ------------------------------------------------------------
    if result is not None:
        parts.append("## Results\n\n")
        for name, df in (tables or {}).items():
            parts.append("```{=html}\n" + _table(df, title=name) + "```\n\n")

    # ---- figures ------------------------------------------------------------
    figs = [Path(f) for f in figures if Path(f).exists()]
    if figs:
        parts.append("## Figures\n\n")
        for f in sorted(figs, key=lambda p: p.stem):
            parts.append(f"### {figure_note(f.stem)['caption']}\n\n")
            parts.append("```{=html}\n" + _figure(f) + "```\n\n")

    out.write_text("".join(parts), encoding="utf-8")
    return out
