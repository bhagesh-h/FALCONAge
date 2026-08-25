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
        "output": 'An age in years, on the same scale as chronological age.',
        "means": (
            'Fitted by penalised regression against calendar age. Reported accuracy is therefore accuracy against a known quantity, and correlation with age is guaranteed by construction rather than evidence of anything. The estimand of interest is the **residual** of predicted on chronological age within the cohort at hand.'),
        "implication": (
            'Analyse the residual. Test-retest error on split samples reaches nine years for prominent clocks in this group, and median absolute error against chronological age is 3.6 years or worse, so differences below that are not resolvable per individual. Across 51 interventional datasets this group showed the smallest responses, consistent with an accumulated level responding more slowly than a rate.'),
        "predicate": lambda c: c.generation == "first",
    },
    {
        "key": "second",
        "title": "Second generation: mortality and morbidity trained",
        "output": 'A mortality hazard, rescaled to year-like units.',
        "means": (
            'Fitted to a survival-weighted composite of clinical measures rather than to age. The reported unit is years; the training target was not age, so the value is a risk expressed on an age-like scale and is not interchangeable with a first-generation prediction.'),
        "implication": (
            "Highest responsiveness of any group across interventional datasets, and the members agree with one another when they move, which is the pattern expected of a shared underlying signal rather than of independent false positives. Note that only about 63 per cent of PhenoAge's accuracy is reproducible by a purely stochastic model of methylation change, against 66 to 75 per cent for Horvath: the non-stochastic remainder is larger in this group."),
        "predicate": lambda c: c.generation == "second",
    },
    {
        "key": "pace",
        "title": "Third generation: pace of aging",
        "output": 'A dimensionless rate: biological change per unit calendar time.',
        "means": (
            'Fitted to the slope of change in organ-system biomarkers measured longitudinally. A value of 1.0 denotes one year of biological change per chronological year.'),
        "implication": (
            'A rate responds before an accumulated level does, which is why a two-year randomised trial (CALERIE) moved DunedinPACE while first-generation clocks did not. **Age acceleration is undefined on this scale**: the quantity is already a rate, so subtracting chronological age is dimensionally invalid and is refused rather than computed.'),
        "predicate": lambda c: c.scale_type == "pace_ratio" or c.generation == "pace",
    },
    {
        "key": "causal",
        "title": "Causal: damage separated from adaptation",
        "output": 'Two scores in years, with no fixed origin.',
        "means": (
            'Features restricted to CpGs with Mendelian-randomisation support and partitioned into damaging and adaptive components. Slope against chronological age is near unity (0.967 for DamAge pooled), but the intercept is cohort-dependent: measured across three healthy cohorts the median bias against chronological age moves by 162 years, against 15 for Horvath.'),
        "implication": (
            '`predicted - chronological` is not a quantity here because the origin does not transfer between cohorts. The within-dataset residual and between-group differences remain defined, and are what the source publications use.'),
        "predicate": lambda c: c.generation == "causal"
        or c.scale_type == "age_years_relative",
    },
    {
        "key": "mitotic",
        "title": "Mitotic: cumulative cell divisions",
        "output": 'An estimated count of stem-cell divisions.',
        "means": (
            'Derived from methylation at polycomb-target promoters or solo-WCGW sites that accumulate change per replication. epiTOC2 and epiTOC3 invert a per-site transmission model rather than summing weighted features.'),
        "implication": (
            'Not elapsed time and not comparable with an age. Tissue turnover rate dominates: two tissues sampled from one donor on one day share a chronological age and differ substantially in division count. Acceleration is refused on this scale.'),
        "predicate": lambda c: c.scale_type == "divisions" or c.generation == "mitotic",
    },
    {
        "key": "system",
        "title": "Explainable: system and organ subscores",
        "output": 'One score per physiological system, plus a composite.',
        "means": (
            'Multi-output models whose subscores are reported alongside the composite. SystemsAge resolves eleven physiological systems; the GrimAge family reports DNAm surrogates of plasma proteins and smoking pack-years, concatenated in a fixed order before a Cox layer.'),
        "implication": (
            'Subscores localise which component contributes to a change, which single-output clocks cannot do, and this is the property the responsiveness literature identifies as most informative for mechanism. Multiplicity applies: with eleven subscores, one extreme value is the expected result under the null and is not a finding on its own.'),
        "predicate": lambda c: c.generation == "system"
        or c.id.startswith(("systemsage", "grimage2", "pcgrimage", "dnamfitage")),
    },
    {
        "key": "composition",
        "title": "Cell composition",
        "output": 'Cell-type proportions, constrained to the simplex.',
        "means": (
            'Reference-based deconvolution against a matrix of cell-type-specific methylation, solved as constrained least squares with non-negativity and a sum-to-one constraint. Not an age estimate.'),
        "implication": (
            'Report alongside any bulk-tissue clock. Bulk methylation reflects cell composition as well as within-cell change, so an intervention that shifts leukocyte proportions shifts the clock; an apparent change in biological age can be a change in the cell mixture sampled.'),
        "predicate": lambda c: c.scale_type == "proportion",
    },
    {
        "key": "other",
        "title": "Other predictors",
        "output": 'Varies by clock. The unit is given per row.',
        "means": (
            'Exposure and lifestyle predictors, DNAm surrogates of individual proteins and clinical measures, telomere-length estimates, and clocks whose training target places them in none of the groups above.'),
        "implication": (
            'Most carry `scale_type: relative_score`, which admits correlation and ranking only: there is no external unit, so differences and means are not defined. Check the unit column before combining any of these.'),
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
    "consensus_plot": {
        "caption": "Consensus across every testable clock",
        "read": "The shape, not any single bar. Effect size per clock, coloured by "
                "generation, with both correction thresholds marked.",
        "wrong": "One lit bar among twenty dark ones is the documented signature of "
                 "a false positive. A real effect lights up across generations, "
                 "because they share the biology and not the feature sets.",
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

#: Click a figure to see it full size.
#:
#: WHY THUMBNAILS AT ALL. Fifty-one figures at full width is a document nobody
#: scrolls to the end of, and the figures are different shapes, so at full width
#: the page also lurches between a square heatmap and a wide forest plot. A
#: uniform tile makes the section scannable; the zoom is there because a
#: thumbnail of a forty-clock heatmap is unreadable by design.
ZOOM_JS = r"""
(function () {
  function overlay() {
    let o = document.getElementById('fa-zoom');
    if (o) return o;
    o = document.createElement('div');
    o.id = 'fa-zoom';
    o.innerHTML = '<img alt=""><button type="button" aria-label="Close">&times;</button>';
    document.body.appendChild(o);
    const close = () => { o.classList.remove('on'); };
    o.addEventListener('click', close);
    // Escape as well as the button: a full-screen overlay with only a small
    // target to dismiss it is a trap on a laptop trackpad.
    document.addEventListener('keydown', e => {
      if (e.key === 'Escape') close();
    });
    return o;
  }
  document.addEventListener('DOMContentLoaded', function () {
    document.querySelectorAll('.fa-figure img').forEach(img => {
      img.tabIndex = 0;
      img.setAttribute('role', 'button');
      img.title = 'Click to enlarge';
      const open = () => {
        const o = overlay();
        o.querySelector('img').src = img.src;
        o.querySelector('img').alt = img.alt || '';
        o.classList.add('on');
      };
      img.addEventListener('click', open);
      img.addEventListener('keydown', e => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); }
      });
    });
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
/* Green, not the brand orange. These blocks state what a quantity is; orange
   is the colour this report uses for cautions, and a definition rendered in it
   read as a warning about the clock rather than a description of its output. */
.fa-meaning { border-left:3px solid #009E73; padding:.55rem .95rem; margin:.8rem 0;
  background:rgba(0,158,115,.06); font-size:.92rem; }
.fa-meaning b, .fa-meaning strong { color:#00674c; }

/* Uniform tiles, three across, so the section is scannable and the page does
   not lurch between a square heatmap and a wide forest plot. `contain` rather
   than `cover`: cropping an axis off a figure to make it fit a grid is worse
   than the letterboxing. */
.fa-figgrid { display:grid; grid-template-columns:repeat(auto-fill,minmax(19rem,1fr));
  gap:1.1rem; margin:1rem 0 1.6rem; }
.fa-figure { margin:0; }
.fa-figure img { width:100%; height:13rem; object-fit:contain; cursor:zoom-in;
  border:1px solid var(--bs-border-color,#e3e3e3); border-radius:6px;
  background:#fbfbfa; padding:.3rem; transition:border-color .12s; }
.fa-figure img:hover, .fa-figure img:focus { border-color:#009E73; outline:none; }
.fa-figure figcaption { font-size:.84rem; margin-top:.35rem; }
.fa-figure .fa-readit { font-size:.8rem; }

/* The conclusion figure is the one worth showing at full size. */
.fa-figure.fa-hero img { height:auto; max-height:none; cursor:zoom-in; }

#fa-zoom { display:none; position:fixed; inset:0; z-index:9999; cursor:zoom-out;
  background:rgba(20,20,20,.88); align-items:center; justify-content:center; padding:2rem; }
#fa-zoom.on { display:flex; }
#fa-zoom img { max-width:96vw; max-height:92vh; width:auto; height:auto;
  background:#fff; border-radius:6px; padding:.5rem; }
#fa-zoom button { position:absolute; top:1rem; right:1.4rem; font-size:2rem;
  line-height:1; color:#fff; background:none; border:none; cursor:pointer; }
.fa-readit { font-size:.88rem; margin:.4rem 0 0; }
.fa-readit b { color:#00674c; }
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


def _figure(path: Path, *, hero: bool = False) -> str:
    n = figure_note(path.stem)
    read = f'<p class="fa-readit"><b>What to look for.</b> {html.escape(n["read"])}</p>' if n["read"] else ""
    wrong = f'<p class="fa-readit"><b>What a bad one looks like.</b> {html.escape(n["wrong"])}</p>' if n["wrong"] else ""
    return (
        f'<figure class="fa-figure{" fa-hero" if hero else ""}">\n'
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
    verdict: str | None = None,
    consensus: pd.DataFrame | None = None,
    conclusion_figure: str | Path | None = None,
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
        # 180px, matching the README. At 46 it read as a favicon beside the
        # title rather than as the mark on the document.
        logo_html = (f'<img src="data:image/png;base64,{_b64(Path(logo))}" '
                     f'alt="FALCONAge" style="width:180px;height:auto">')

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
                 f"<script>{TABLE_JS}</script>\n"
                 f"<script>{ZOOM_JS}</script>\n```\n\n")
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

    # ---- conclusion ---------------------------------------------------------
    #
    # First, not last. A reader who opens a fifty-figure report and has to scroll
    # to the end to find out whether anything was detected will read the figures
    # without knowing what they are looking for. The consensus test is the only
    # statement in the document that is about the run as a whole rather than
    # about one clock.
    if verdict or (consensus is not None and len(consensus)):
        parts.append("## Conclusion\n\n")
        if verdict:
            # `consensus_verdict.txt` is two lines of machine output: a one-word
            # verdict, then the counts. Rendered verbatim it reads as a log line
            # rather than a finding, so the word is capitalised, the counts are
            # made a sentence, and the ASCII dash is not left in prose.
            v = " ".join(str(verdict).split())
            v = v.replace(" -- ", ": ").replace("--", ":")
            for word in ("unsupported", "supported", "equivocal"):
                if v.lower().startswith(word):
                    rest = v[len(word):].lstrip(" .:")
                    v = word.capitalize() + ". " + rest[:1].upper() + rest[1:]
                    break
            else:
                v = v[:1].upper() + v[1:]
            if not v.endswith("."):
                v += "."
            parts.append(
                '```{=html}\n<div class="fa-meaning">\n'
                f"<p><strong>Verdict.</strong> {html.escape(v)}</p>\n"
                "</div>\n```\n\n")
        if consensus is not None and len(consensus):
            n = len(consensus)
            sig_b = int(consensus.get("sig_bonferroni", pd.Series(dtype=bool)).sum())
            sig_h = int(consensus.get("sig_bh", pd.Series(dtype=bool)).sum())
            gens = ", ".join(sorted(set(consensus.get("generation", []))))
            parts.append(
                f"{n} clocks were testable on this contrast. **{sig_b} reached "
                f"significance after Bonferroni correction and {sig_h} after "
                f"Benjamini-Hochberg.** The clocks span {gens}, which matters "
                "for reading the result: a real effect appears across "
                "generations, because they share the underlying biology and not "
                "their feature sets. A single significant clock among twenty is "
                "the documented signature of a false positive rather than of a "
                "narrow effect.\n\n")
        if hero_ok := (conclusion_figure and Path(conclusion_figure).exists()):
            parts.append("```{=html}\n"
                         + _figure(Path(conclusion_figure), hero=True)
                         + "```\n\n")
        if consensus is not None and len(consensus):
            parts.append("```{=html}\n" + _table(
                consensus, title="Consensus test, every clock",
                note="Effect size, uncorrected p, and both corrected thresholds "
                     "per clock. Sorted as tested, not by p, so the table cannot "
                     "be read as a ranking.",
                open_by_default=True) + "```\n\n")

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
    #
    # A grid of uniform tiles rather than fifty-one full-width images stacked.
    # Each tile zooms on click, because a thumbnail of a forty-clock heatmap is
    # unreadable and cropping one to fit the grid would remove an axis.
    figs = [Path(f) for f in figures if Path(f).exists()]
    hero = Path(conclusion_figure) if conclusion_figure else None
    if hero and hero.exists():
        figs = [f for f in figs if f.resolve() != hero.resolve()]
    if figs:
        parts.append("## Figures\n\n")
        parts.append(
            "Every figure is shown at the same size so the section can be "
            "scanned. Click one to see it full size, or press Escape to close.\n\n")
        tiles = "".join(_figure(f) for f in sorted(figs, key=lambda p: p.stem))
        parts.append('```{=html}\n<div class="fa-figgrid">\n' + tiles
                     + "</div>\n```\n\n")

    out.write_text("".join(parts), encoding="utf-8")
    return out
