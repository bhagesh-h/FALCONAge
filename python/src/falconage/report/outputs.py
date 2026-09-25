"""What each file of a run is, and which step of the analysis produced it.

WHY A TABLE. A run writes a dozen tables and up to fifty figures into one
directory, and a directory lists them alphabetically: a failed QC table and the
headline scores sit side by side as files of equal standing. The report reads
this table to place every file under the step that produced it, in the order a
reader has to go through them, because a score is only as good as the coverage
and the QC before it. The idea, and the rule that every file must be accounted
for, follow cyRAVEN's run report (``report_table_note``, ``write_run_report``).

A file the table does not name still appears, in the last step, with its name
and size. Nothing in the output directory is left out of the report silently.

Descriptions of figures are not here: they live with the figures, in
``quarto.FIGURE_NOTES``, so a figure's text exists once.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path

__all__ = ["OUTPUTS", "STEPS", "Output", "Step", "collect"]


@dataclass(frozen=True)
class Step:
    number: int
    title: str
    purpose: str


@dataclass(frozen=True)
class Output:
    #: Path relative to the output directory; shell wildcards allowed.
    pattern: str
    step: int
    #: "table", "figure", "json", "text" or "file" (named, never embedded).
    kind: str
    title: str = ""
    description: str = ""


STEPS: tuple[Step, ...] = (
    Step(0, "Provenance",
         "What ran, on which data, with which coefficients: the package and registry "
         "versions, the SHA-256 of every coefficient file used, the device and every "
         "warning. Enough to reproduce each number below."),
    Step(1, "Input and quality control",
         "Whether the measurements are fit to score: missing values, the shape of the "
         "beta distribution, and whether declared sex agrees with the sex chromosomes. "
         "A sample that fails here makes every score after it unreliable."),
    Step(2, "Coverage",
         "How much of each clock the data carries, counted in features and in "
         "coefficient weight, and which clocks were not scored and why. A clock that "
         "lost its heaviest features returns an extrapolation, not a measurement."),
    Step(3, "Scores",
         "One score per sample and clock, and how the clocks relate to each other. "
         "Clocks trained on different targets are not interchangeable, even where they "
         "share a unit."),
    Step(4, "Agreement with chronological age",
         "For clocks reported in years: error against chronological age, and whether "
         "that error depends on age. Distance from the identity line is error, not "
         "acceleration."),
    Step(5, "Uncertainty",
         "How much of each score is measurement noise, and a distribution-free interval "
         "for a new sample. A difference smaller than these is not resolvable per "
         "individual."),
    Step(6, "Age acceleration",
         "The residual of each score on chronological age, fitted within this cohort, "
         "for the scales on which it is defined. It is refused for rates and scores "
         "without an age origin."),
    Step(7, "Group comparison",
         "Whether the groups differ, judged across clocks and generations rather than by "
         "the most significant one."),
    Step(8, "Interpretation",
         "What each score means, the published evidence behind it with its DOIs, and "
         "the warnings the run raised."),
    Step(9, "Further outputs",
         "Every other file in the output directory, named and sized."),
)

OUTPUTS: tuple[Output, ...] = (
    Output("run_manifest.json", 0, "json", "Run manifest",
           "Versions, configuration, the coefficient file and SHA-256 behind each clock, "
           "the device each clock ran on, and every warning."),

    Output("qc_per_sample.csv", 1, "table", "Quality control, per sample",
           "Fraction of missing values, and the mean and SD of each sample's betas. A "
           "sample far from the rest on either is the one to check first."),
    Output("figures/beta_density.png", 1, "figure"),
    Output("figures/missingness.png", 1, "figure"),
    Output("figures/sex_check.png", 1, "figure"),

    Output("qc.csv", 2, "table", "Coverage, per clock",
           "Share of each clock's features present, the share of its coefficient weight "
           "they carry, and how many values were filled and how."),
    Output("skipped_clocks.csv", 2, "table", "Clocks not scored",
           "Every clock considered and not scored, with the reason."),
    Output("figures/coverage_bar.png", 2, "figure"),
    Output("figures/platform_bias.png", 2, "figure"),
    Output("figures/platform_*.png", 2, "figure"),

    Output("scores_wide.csv", 3, "table", "Scores, one row per sample",
           "One column per clock, in the unit the clock reports."),
    Output("scores.csv", 3, "table", "Scores, one row per sample and clock",
           "The long form, with each clock's unit and scale type beside the value."),
    Output("figures/clock_corr.png", 3, "figure"),
    Output("figures/clock_pca.png", 3, "figure"),
    Output("figures/clock_chord.png", 3, "figure"),
    Output("figures/clock_radar.png", 3, "figure"),

    Output("figures/ba_vs_ca_*.png", 4, "figure"),
    Output("figures/bland_altman_*.png", 4, "figure"),
    Output("figures/calibration_*.png", 4, "figure"),
    Output("figures/study_*.png", 4, "figure"),

    Output("technical_se.csv", 5, "table", "Technical standard error, per sample",
           "The part of each score that repeat measurement of the same DNA would move, "
           "propagated from per-probe reliability through the clock's weights."),
    Output("reliability_diagnostics.csv", 5, "table", "Reliability, per clock",
           "The inputs to the technical standard error: how many probes carry a "
           "published reliability, their median, and the ratio of technical error to "
           "the spread of the cohort."),
    Output("conformal_interval.csv", 5, "table", "Prediction intervals",
           "Split-conformal intervals at the stated level, from residuals on healthy "
           "adult blood samples of known age. The coverage holds only for samples "
           "drawn like that calibration cohort, which nothing here can verify, so "
           "`exchangeable` is always false."),
    Output("figures/reliability_forest.png", 5, "figure"),
    Output("figures/score_interval_*.png", 5, "figure"),

    Output("acceleration.csv", 6, "table", "Age acceleration",
           "Residual of each score on chronological age, for clocks whose scale admits "
           "it."),
    Output("figures/acceleration_heatmap.png", 6, "figure"),
    Output("figures/acceleration_density_*.png", 6, "figure"),

    Output("consensus_verdict.txt", 7, "text", "Consensus verdict",
           "The verdict across clocks, and the reason for it."),
    Output("consensus.csv", 7, "table", "Consensus test, every clock",
           "Effect size, uncorrected p and both corrected thresholds per clock, in the "
           "order tested rather than by p, so the table cannot be read as a ranking."),
    Output("figures/consensus_plot.png", 7, "figure"),
    Output("figures/acceleration_group_*.png", 7, "figure"),

    Output("interpretation.csv", 8, "table", "What each score means",
           "Per clock: what it predicts and was trained on, its unit and permitted "
           "operations, its coverage in this run, published reliability, caveats, and a "
           "digest of published associations."),
    Output("evidence.csv", 8, "table", "Published evidence",
           "Effect sizes from the literature for each clock, with the DOI of the paper "
           "that reports them."),

    Output("figures/SKIPPED.txt", 9, "text", "Figures not drawn",
           "Each figure the data could not support, and why. An absent figure with a "
           "reason reads as an absence; an empty one would read as a measurement."),
    Output("report.html", 9, "file", "One-page report",
           "The compact self-contained report, for sending."),
)

#: The Quarto report's own files: never listed inside it.
OWN = ("falconage_report.qmd", "falconage_report.html", "falconage_report_files/*")


def classify(relpath: str) -> Output | None:
    """The first table entry whose pattern matches, or None."""
    for o in OUTPUTS:
        if fnmatch.fnmatchcase(relpath, o.pattern):
            return o
    return None


def collect(outdir: str | Path) -> dict[int, list[tuple[Path, Output]]]:
    """Every file under ``outdir``, placed under its step, in table order.

    Files the table does not name go to the last step as ``kind="file"``.
    Within a step, files follow the table's order, then their names, so a
    report of the same directory is the same document every time.
    """
    root = Path(outdir)
    placed: dict[int, list[tuple[int, str, Path, Output]]] = {s.number: [] for s in STEPS}
    order = {o.pattern: i for i, o in enumerate(OUTPUTS)}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if any(fnmatch.fnmatchcase(rel, pat) for pat in OWN):
            continue
        o = classify(rel)
        if o is None:
            o = Output(rel, STEPS[-1].number, "file", rel, "")
            rank = len(OUTPUTS)
        else:
            rank = order[o.pattern]
        placed[o.step].append((rank, rel, p, o))
    return {n: [(p, o) for _, _, p, o in sorted(items, key=lambda t: (t[0], t[1]))]
            for n, items in placed.items()}
