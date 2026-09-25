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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


#: File stems whose figure is defined in colorscheme.yaml under another name.
_PLOT_ALIASES = {"platform": "platform_comparison", "study": "study_comparison"}


def figure_note(stem: str) -> dict[str, str]:
    """The notes for a figure file's stem.

    Per-clock figures are written as ``<kind>_<clock>`` (``ba_vs_ca_hannum``), so
    the longest known kind the stem starts with is used and the clock is named
    in the caption; an exact lookup missed every one of them. A kind with no
    entry in :data:`FIGURE_NOTES` takes the title and description the figure
    itself prints, from colorscheme.yaml, so its text is not written twice.
    """
    from ..plot.spec import load as _spec

    plots = _spec()["plots"]
    kinds = set(FIGURE_NOTES) | set(plots) | set(_PLOT_ALIASES)
    kind, clock = stem, ""
    if stem not in kinds:
        for k in sorted(kinds, key=len, reverse=True):
            if stem.startswith(k + "_"):
                kind, clock = k, stem[len(k) + 1:]
                break
    if kind in FIGURE_NOTES:
        note = dict(FIGURE_NOTES[kind])
    elif _PLOT_ALIASES.get(kind, kind) in plots:
        spec = plots[_PLOT_ALIASES.get(kind, kind)]
        note = {"caption": spec["title"],
                "read": " ".join(str(spec.get("description", "")).split()),
                "wrong": ""}
    else:
        note = {"caption": stem.replace("_", " ").capitalize(), "read": "", "wrong": ""}
    if clock:
        note["caption"] = f'{note["caption"]}: {clock}'
    return note


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

# Rules only; colours and fonts are the interface tokens in colorscheme.yaml,
# shared with the one-page report and the documentation site.
TABLE_RULES = r"""
body { font-family:var(--font-sans); color:var(--ink); background:var(--bg); }
a { color:var(--accent); }
code, pre { font-family:var(--font-mono); }
.fa-tablebar { display:flex; gap:.5rem; align-items:center; margin:.4rem 0 .5rem; }
.fa-tablebar input[type=search] { flex:1 1 14rem; padding:.3rem .5rem;
  border:1px solid var(--line); border-radius:4px; font-size:.85rem; }
.fa-tablebar select { padding:.3rem; border:1px solid var(--line);
  border-radius:4px; font-size:.85rem; }
.fa-tablecount { font-size:.78rem; color:var(--muted); white-space:nowrap; }
.fa-tocfilter { width:100%; box-sizing:border-box; margin:0 0 .6rem; padding:.3rem .5rem;
  border:1px solid var(--line); border-radius:4px; font-size:.85rem; }
.fa-toggle { border:1px solid var(--line); border-radius:6px;
  padding:.4rem .7rem; margin:.8rem 0; }
.fa-toggle > summary { cursor:pointer; font-weight:600; font-size:.92rem; }
.fa-table { overflow-x:auto; }
.fa-table table { width:100%; font-size:.84rem; border-collapse:collapse;
  font-variant-numeric:tabular-nums; }
.fa-table th, .fa-table td { padding:.3rem .55rem; border-bottom:1px solid var(--line);
  text-align:left; white-space:nowrap; }
/* Neutral, not a status colour. These blocks state what a quantity is; the
   accent and the warning colours mean something else in this report, and a
   definition drawn in either read as a caution about the clock. */
.fa-meaning { border-left:3px solid var(--muted); padding:.55rem .95rem; margin:.8rem 0;
  background:var(--panel); font-size:.92rem; }
.fa-meaning b, .fa-meaning strong { color:var(--ink); }

/* Uniform tiles, three across, so the section is scannable and the page does
   not lurch between a square heatmap and a wide forest plot. `contain` rather
   than `cover`: cropping an axis off a figure to make it fit a grid is worse
   than the letterboxing. */
.fa-figgrid { display:grid; grid-template-columns:repeat(auto-fill,minmax(19rem,1fr));
  gap:1.1rem; margin:1rem 0 1.6rem; }
.fa-figure { margin:0; }
.fa-figure img { width:100%; height:13rem; object-fit:contain; cursor:zoom-in;
  border:1px solid var(--line); border-radius:6px;
  background:#ffffff; padding:.3rem; transition:border-color .12s; }
.fa-figure img:hover, .fa-figure img:focus { border-color:var(--accent); outline:none; }
.fa-figure figcaption { font-size:.84rem; margin-top:.35rem; }
.fa-figure .fa-readit { font-size:.8rem; }

/* The conclusion figure is the one worth showing at full size. */
.fa-figure.fa-hero img { height:auto; max-height:none; cursor:zoom-in; }

#fa-zoom { display:none; position:fixed; inset:0; z-index:9999; cursor:zoom-out;
  background:rgba(20,20,20,.88); align-items:center; justify-content:center; padding:2rem; }
#fa-zoom.on { display:flex; }
#fa-zoom img { max-width:96vw; max-height:92vh; width:auto; height:auto;
  background:#ffffff; border-radius:6px; padding:.5rem; }
#fa-zoom button { position:absolute; top:1rem; right:1.4rem; font-size:2rem;
  line-height:1; color:#ffffff; background:none; border:none; cursor:pointer; }
.fa-readit { font-size:.88rem; margin:.4rem 0 0; }
.fa-readit b { color:var(--ink); }
.fa-dl { display:inline-block; margin-left:.5rem; font-size:.76rem; color:var(--accent);
  text-decoration:none; border:1px solid var(--accent); border-radius:4px;
  padding:.05rem .45rem; white-space:nowrap; }
.fa-dl:hover, .fa-dl:focus { background:var(--accent); color:var(--bg); }
pre { background:var(--panel); border:1px solid var(--line); border-radius:6px;
  padding:.6rem .8rem; font-size:.82rem; }
"""


def report_css() -> str:
    """Interface tokens, then the report's rules. Figures sit on white in both
    themes because they are drawn on white; the frame around them follows the
    theme."""
    from ..plot.spec import ui_css_variables

    return ui_css_variables() + TABLE_RULES


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


#: One download link per figure, built from the image already on the page, so
#: the full-resolution file costs no second copy of its bytes (as in cyRAVEN).
DOWNLOAD_JS = r"""
document.addEventListener('DOMContentLoaded', function () {
  document.querySelectorAll('.fa-figure[data-file]').forEach(fig => {
    const img = fig.querySelector('img');
    const a = document.createElement('a');
    a.className = 'fa-dl'; a.textContent = 'Download PNG';
    a.download = fig.dataset.file.split('/').pop();
    a.href = img.src;
    fig.querySelector('figcaption').appendChild(a);
  });
});
"""

#: A table above this size on disk is named in the last step instead of being
#: embedded: a report that runs to hundreds of megabytes will not be opened.
MAX_TABLE_BYTES = 2_000_000


def _human(n: int) -> str:
    for unit in ("B", "kB", "MB", "GB"):
        if n < 1000 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000
    return f"{n} B"


def _csv_link(path: Path) -> str:
    return (f'<a class="fa-dl" download="{html.escape(path.name)}" '
            f'href="data:text/csv;base64,{_b64(path)}">Download CSV</a>')


def _manifest_html(path: Path) -> str:
    """The run manifest as three readable tables rather than raw JSON."""
    import json

    m = json.loads(path.read_text(encoding="utf-8"))
    keys = ("falconage_version", "registry_version", "started_utc", "finished_utc",
            "caller", "python", "platform", "device_requested", "device", "dtype")
    run = pd.DataFrame([{"field": k.replace("_", " "), "value": str(m.get(k, ""))}
                        for k in keys if k in m])
    weights = pd.DataFrame([
        {"clock": c, "source": w.get("source", ""), "file": w.get("path", ""),
         "sha256": w.get("sha256", ""),
         "primary source traced": w.get("primary_source_traced", "")}
        for c, w in sorted((m.get("weights") or {}).items())])
    warns = pd.DataFrame(m.get("warnings") or [])
    return (_table(run, title="Run", open_by_default=True)
            + _table(weights, title="Coefficients used",
                     note="The file and SHA-256 behind every clock scored. A different "
                          "digest is a different clock, whatever its name.")
            + _table(warns, title="Warnings raised",
                     note="Every warning the run raised, with the clock it concerns."))


def _category_notes(result: Any, reg: Any) -> str:
    """What the number means, for each category of clock that was scored."""
    if result is None or getattr(result, "scores", None) is None:
        return ""
    scored = [reg.get(c) for c in map(str, result.scores.columns) if c in reg]
    out = []
    for cat in CATEGORIES:
        ids = sorted(c.id for c in scored if categorise(c) == cat["key"])
        if not ids:
            continue
        out.append(
            '<div class="fa-meaning">\n'
            f'<p><strong>{html.escape(cat["title"])}</strong> '
            f'({html.escape(", ".join(ids))})</p>\n'
            f'<p><strong>What the number is.</strong> {html.escape(cat["output"])}</p>\n'
            f'<p>{cat["means"]}</p>\n'
            f'<p><strong>What follows from it.</strong> {cat["implication"]}</p>\n'
            "</div>\n")
    return "".join(out)


def _verdict_html(path: Path) -> str:
    """``consensus_verdict.txt`` is machine output; written here as a sentence."""
    v = " ".join(path.read_text(encoding="utf-8").split())
    v = v.replace(" -- ", ": ").replace("--", ":")
    for word in ("unsupported", "supported", "equivocal"):
        if v.lower().startswith(word):
            rest = v[len(word):].lstrip(" .:")
            v = word.capitalize() + ". " + rest[:1].upper() + rest[1:]
            break
    if v and not v.endswith("."):
        v += "."
    return (f'<div class="fa-meaning"><p><strong>Verdict.</strong> '
            f"{html.escape(v)}</p></div>\n")


def write_quarto_report(
    outdir: str | Path,
    result: Any = None,
    *,
    title: str = "FALCONAge report",
    logo: str | Path | None = None,
    registry: Any = None,
    render: bool = False,
) -> Path:
    """Write ``falconage_report.qmd`` into a run's output directory; render it if asked.

    Every file in ``outdir`` is placed under the step of the analysis that
    produced it (:mod:`falconage.report.outputs`), in step order, with its
    description: tables collapsible, searchable and downloadable as CSV,
    figures zoomable and downloadable at full resolution. A file the output
    table does not name is listed in the last step, so nothing in the directory
    is left out. Rendering embeds every figure and table, so the HTML is one
    file that references nothing.

    ``result``, when given, adds what each scored category of clock means to the
    scores step. ``render=True`` runs ``quarto render`` and returns the HTML
    path; it raises if Quarto is not on the path, naming the source to render
    elsewhere. Otherwise the ``.qmd`` path is returned.
    """
    import shutil
    import subprocess

    from .. import __version__
    from .. import registry as _registry_mod
    from .outputs import STEPS, collect

    reg = registry or _registry_mod.load()
    outdir = Path(outdir)
    qmd = outdir / "falconage_report.qmd"
    stamp = datetime.now(timezone.utc).strftime("%d %B %Y, %H:%M UTC")
    placed = collect(outdir)

    logo_html = ""
    if logo and Path(logo).exists():
        logo_html = (f'<img src="data:image/png;base64,{_b64(Path(logo))}" '
                     f'alt="FALCONAge" style="width:120px;height:auto">')

    parts: list[str] = [
        "---\n"
        f'title: "{title}"\n'
        "lang: en\n"
        "format:\n"
        "  html:\n"
        "    theme: default\n"
        "    toc: true\n"
        "    toc-location: left\n"
        "    toc-depth: 2\n"
        "    toc-title: Steps\n"
        "    embed-resources: true\n"
        "    page-layout: full\n"
        "---\n\n",
        f"```{{=html}}\n<style>{report_css()}</style>\n"
        f"<script>{TABLE_JS}</script>\n<script>{ZOOM_JS}</script>\n"
        f"<script>{DOWNLOAD_JS}</script>\n```\n\n",
        "```{=html}\n"
        '<div style="display:flex;align-items:center;gap:.9rem;margin:0 0 1.2rem">\n'
        f"{logo_html}\n"
        f'<div class="fa-tablecount">Generated {stamp} by FALCONAge {__version__}, '
        f"from {html.escape(str(outdir.resolve().name))}/</div>\n</div>\n```\n\n",
        "The sections follow the order of the analysis, because each step is only "
        "as reliable as the ones before it: coverage decides whether a score is a "
        "measurement, and uncertainty decides whether a difference is one. Every "
        "file in the output directory appears once, under the step that wrote it. "
        "Tables expand, search, page and download; figures enlarge on click and "
        "download at full resolution.\n\n",
    ]

    further: list[tuple[str, int, str]] = []
    for step in STEPS:
        items = placed.get(step.number, [])
        parts.append(f"## {step.number}. {step.title} {{#step-{step.number}}}\n\n")
        parts.append(f"{step.purpose}\n\n")
        block: list[str] = []
        figs: list[str] = []
        if step.number == 3:
            block.append(_category_notes(result, reg))
        for path, o in items:
            rel = path.relative_to(outdir).as_posix()
            size = path.stat().st_size
            if o.kind == "figure":
                figs.append(_figure(path).replace(
                    '<figure class="fa-figure">',
                    f'<figure class="fa-figure" data-file="{html.escape(rel)}">', 1))
            elif o.kind == "table" and size <= MAX_TABLE_BYTES:
                df = pd.read_csv(path)
                note = html.escape(o.description) + " " + _csv_link(path)
                block.append(_table(df, title=f"{o.title} ({rel})", note=note,
                                    open_by_default=(step.number in (3, 7))))
            elif o.kind == "json" and rel == "run_manifest.json":
                block.append(_manifest_html(path))
            elif o.kind == "text" and rel == "consensus_verdict.txt":
                block.append(_verdict_html(path))
            elif o.kind == "text":
                block.append(f"<p><strong>{html.escape(o.title or rel)}</strong> "
                             f"({html.escape(rel)}). {html.escape(o.description)}</p>\n<pre>"
                             f"{html.escape(path.read_text(encoding='utf-8'))}</pre>\n")
            else:
                why = o.description or "Not described by the report's output table."
                if o.kind == "table":
                    why = (f"{o.description} Not embedded: {_human(size)} is above the "
                           f"{_human(MAX_TABLE_BYTES)} limit for an embedded table.")
                further.append((rel, size, why))
        if figs:
            block.append('<div class="fa-figgrid">\n' + "".join(figs) + "</div>\n")
        if step.number == STEPS[-1].number and further:
            block.append(_table(
                pd.DataFrame([{"file": f, "size": _human(n), "what it is": w}
                              for f, n, w in further]),
                title="Files in the output directory not embedded above",
                open_by_default=True))
        block = [b for b in block if b]
        if block:
            parts.append("```{=html}\n" + "".join(block) + "```\n\n")
        else:
            parts.append("*This run wrote nothing for this step.*\n\n")

    qmd.write_text("".join(parts), encoding="utf-8", newline="\n")
    if not render:
        return qmd

    quarto = shutil.which("quarto")
    if quarto is None:
        raise RuntimeError(
            f"Quarto is not on the path, so {qmd.name} was written but not rendered.\n"
            f"  Render it where Quarto is installed:  quarto render {qmd}\n"
            "  or, from that directory, with the Quarto container:\n"
            "    docker run --rm --user \"$(id -u):$(id -g)\" -e HOME=/tmp -v \"$PWD:/w\" "
            "-w /w ghcr.io/quarto-dev/quarto quarto render " + qmd.name)
    done = subprocess.run([quarto, "render", qmd.name], cwd=outdir,
                          capture_output=True, text=True)
    if done.returncode != 0:
        tail = "\n".join((done.stderr or done.stdout).strip().splitlines()[-12:])
        raise RuntimeError(f"quarto render {qmd.name} failed:\n{tail}")
    out = qmd.with_suffix(".html")
    from ..core.logging import get_logger

    get_logger(__name__).info("wrote %s (%s)", out, _human(out.stat().st_size))
    return out
