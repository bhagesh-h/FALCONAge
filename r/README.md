# FALCONAge for R

<!-- badges: start -->
[![R-CMD-check](https://github.com/bhagesh-h/FALCONAge/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/bhagesh-h/FALCONAge/actions/workflows/R-CMD-check.yaml)
[![R 4.1+](https://img.shields.io/badge/R-4.1%2B-blue)](https://www.r-project.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
<!-- badges: end -->

Biological age and aging-clock scoring from DNA methylation, clinical chemistry and NMR
metabolomics, against a catalogue of 68 published human clocks. The input is data that is
already preprocessed and quality controlled, such as the beta matrix from an array pipeline.

This is the R half. It is a native R package: S3 classes, `data.frame` in and out, roxygen2
documentation, ggplot2 figures, but it does not reimplement anything. Every number comes from one
Python core that this package calls through reticulate, so **an R result and a Python result are
the same bits**, and the test suite asserts that at tolerance exactly zero rather than to six
decimals.

That design decision is the reason this page exists in the form it does: if you are reading the R
reference and something is unclear, [the Python reference](https://bhagesh-h.github.io/FALCONAge/reference/index.html) documents the
same computation, and the table below tells you which entry to look at.

## Install

Not on CRAN yet. The package is the `r/` subdirectory of the repository, hence `subdir`:

```r
remotes::install_github("bhagesh-h/FALCONAge", subdir = "r")
# or, resolving dependencies better and taking the subdirectory inline:
pak::pak("bhagesh-h/FALCONAge/r")
```

Then, once per machine:

```r
library(FALCONAge)
falconage_install()   # builds the managed Python environment; a few minutes
falconage_config()    # what resolved: versions, registry size
```

`falconage_install()` fetches the Python core from the same GitHub tag this package was built
from, so the two halves cannot drift apart. `falconage_config()` is the first thing to run when
anything looks wrong.

## A whole analysis

```r
d   <- prepare(read_betas("betas.csv"))
res <- score(d, clocks = "compatible")

summary(res)
as.data.frame(res)                        # samples x clocks
as.data.frame(res, form = "long")         # one row per sample per clock, with its scale

coverage(res)                             # what was scored, what was refused, and why
manifest(res)                             # versions, precision, coefficient checksums

acceleration(res, method = "both")
plot_ba_vs_ca(res, "horvath2013")
report(res, "report.html", group = "condition")
```

`clocks = "compatible"` scores what the data can support and reports the rest as skipped with a
reason. Naming clocks explicitly is different: then every one must work, because an explicit
request should never be dropped quietly.

## Reading the R reference alongside the Python one

The two APIs are one-to-one in intent and deliberately not in spelling. R gets
`as.data.frame(res, form = "long")` where Python gets `FalconResult.long()`, because forcing
identical names would make one language read like a translation of the other, which is exactly
what the reticulate bridge exists to avoid.

So the mapping is worth having explicitly. Every row is the same computation reached two ways.

<!-- BEGIN GENERATED: api-map -->

| | R | Python |
|---|---|---|
| **Start here** | `as.data.frame.falcon_result()` | `fa.FalconResult()` |
|  | `read_betas()` | `fa.read_betas()` |
|  | `prepare()` | `fa.prepare()` |
|  | `technical_se()` | `fa.technical_se()` |
|  | `conformal_interval()` | `fa.conformal_interval()` |
|  | `consensus()` | `fa.consensus()` |
|  | `score()` | `fa.score()` |
|  | `interpretation()` | `res.interpretation()` |
|  | `acceleration()` | `fa.acceleration()` |
|  | `run_report()` | `fa.report.run_report()` |
|  | — | `fa.read()` |
|  | — | `fa.demo_data()` |
| **Reading data** | `read_series_matrix()` | `fa.read_series_matrix()` |
|  | `read_clinical()` | `fa.read_clinical()` |
|  | `falcon_data()` | `fa.FalconData()` |
|  | `obs()` | `data.obs` |
|  | `write_h5ad()` | `data.write_h5ad()` |
|  | — | `fa.read_nightingale()` |
|  | — | `fa.read_panel()` |
|  | — | `fa.read_computage_bench()` |
|  | — | `fa.list_computage_bench()` |
| **Preparing data** | `horvath_normalise()` | `fa.preprocess.horvath_normalise()` |
|  | `qc()` | `fa.qc()` |
|  | `probe_loss()` | `fa.probe_loss()` |
|  | — | `fa.prepare_clinical()` |
|  | — | `fa.preprocess.aggregate_replicate_probes()` |
|  | — | `fa.preprocess.platform_probes()` |
|  | — | `fa.preprocess.impute()` |
| **Uncertainty** | `interval()` | `fa.interval()` |
|  | `icc_from_replicates()` | `fa.icc_from_replicates()` |
|  | — | `fa.uncertainty.load_probe_icc()` |
|  | — | `fa.uncertainty.probe_icc_source()` |
|  | — | `fa.uncertainty.load_conformal()` |
| **Study design** | `power_n()` | `fa.power()` |
|  | — | `fa.detectable_effect()` |
|  | — | `fa.registry.evidence()` |
| **Scoring** | `combine()` | `fa.combine()` |
|  | `coverage()` | `res.coverage` |
|  | `manifest()` | `res.manifest` |
|  | `write_results()` | `res.write()` |
| **The clock catalogue** | `list_clocks()` | `fa.registry.ClockRegistry()` |
|  | `clock_info()` | `fa.registry.Clock()` |
|  | `register_local_weights()` | `fa.registry.register_local_weights()` |
|  | `cite_clock()` | `reg.get(id).cite()` |
|  | `compatible_clocks()` | `reg.compatible_with(data)` |
|  | — | `fa.registry.load()` |
| **Analysis** | `cell_composition()` | `fa.cell_composition()` |
|  | `associate()` | `fa.associate()` |
|  | `cox_hazard()` | `fa.cox_hazard()` |
|  | `agreement()` | `fa.agreement()` |
|  | `icc()` | `fa.icc()` |
|  | `run_benchmark()` | `fa.run_benchmark()` |
|  | — | `fa.leave_one_marker_out()` |
| **Your own clocks** | `compare_learners()` | `fa.compare_learners()` |
|  | `fit_clock()` | `fa.fit_clock()` |
|  | `stack_clocks()` | `fa.stack_clocks()` |
|  | `add_clock()` | `fa.add_clock()` |
|  | `clock_to_file()` | *the `.to_clock()` method of a fit or a stack* |
| **Clinical references** | `fit_kdm()` | `fa.models.fit_kdm()` |
|  | `fit_hd()` | `fa.models.fit_hd()` |
|  | — | `fa.models.clinical.phenoage()` |
|  | — | `fa.blood_count_ratios()` |
| **Model architectures** | — | `fa.models.AggregationClock()` |
|  | — | `fa.models.NeuralClock()` |
|  | — | `fa.models.read_neural_weights()` |
|  | — | `fa.models.ScAgeReference()` |
|  | — | `fa.models.fit_scage_reference()` |
|  | — | `fa.models.scage()` |
| **Figures and reports** | `plot_clock_coverage()` | `fa.plot.clock_coverage_manhattan()` |
|  | `save_figure()` | `fa.plot.save_figure()` |
|  | `figure_text()` | `fa.plot.figure_text()` |
|  | `plot_ba_vs_ca()` | `fa.plot.ba_vs_ca()` |
|  | `plot_bland_altman()` | `fa.plot.bland_altman()` |
|  | `plot_calibration()` | `fa.plot.calibration()` |
|  | `plot_acceleration()` | `fa.plot.acceleration_density()` |
|  | `plot_acceleration_by_group()` | `fa.plot.acceleration_by_group()` |
|  | `plot_acceleration_heatmap()` | `fa.plot.acceleration_heatmap()` |
|  | `plot_forest()` | `fa.plot.forest()` |
|  | `plot_kaplan_meier()` | `fa.plot.kaplan_meier()` |
|  | `plot_volcano()` | `fa.plot.volcano()` |
|  | `plot_agreement()` | `fa.plot.clock_corr()` |
|  | `plot_clock_radar()` | `fa.plot.clock_radar()` |
|  | `plot_clock_chord()` | `fa.plot.clock_chord()` |
|  | `plot_clock_pca()` | `fa.plot.clock_pca()` |
|  | `plot_clock_atlas()` | `fa.plot.clock_atlas()` |
|  | `plot_coverage()` | `fa.plot.coverage_bar()` |
|  | `plot_by_platform()` | `fa.plot.platform_comparison()` |
|  | `plot_benchmark()` | `fa.plot.benchmark_bars()` |
|  | `plot_benchmark_error_bias()` | `fa.plot.benchmark_error_bias()` |
|  | `plot_benchmark_heatmap()` | `fa.plot.benchmark_heatmap()` |
|  | `falcon_palette()` | `fa.plot.palette()` |
|  | `report()` | `fa.report.write_report()` |
|  | `quarto_report()` | `fa.report.write_quarto_report()` |
|  | `falcon_scheme()` | *R only: loads the shared palette into R* |
|  | `falcon_theme()` | *R only: a ggplot2 theme; Python styles via matplotlib* |
|  | — | `fa.plot.clock_coverage_points()` |
|  | — | `fa.plot.missingness()` |
|  | — | `fa.plot.beta_density()` |
|  | — | `fa.plot.study_comparison()` |
|  | — | `fa.plot.reliability_forest()` |
|  | — | `fa.plot.score_interval()` |
|  | — | `fa.plot.platform_bias()` |
|  | — | `fa.plot.consensus_plot()` |
|  | — | `fa.plot.save_all()` |
|  | — | `fa.plot.read_caption()` |
| **Downloading public data** | `download()` | `fa.download()` |
|  | `cache_info()` | `fa.download.cache_info()` |
|  | `clear_cache()` | `fa.download.clear_cache()` |
| **Configuration and setup** | `falconage_config()` | `fa.config()` |
|  | `falconage_install()` | *R only: builds the Python environment R calls into* |
|  | `falconage_available()` | *R only: is that environment resolvable yet* |
|  | — | `fa.configure()` |
|  | — | `fa.core.resolve()` |
|  | — | `fa.core.describe()` |
|  | — | `fa.FalconConfig()` |
|  | — | `fa.RunManifest()` |

<!-- END GENERATED: api-map -->

## What the R side does not have

`download()` and a few model architectures are Python-only by design: an R user reaches them
through the same core, and an S3 wrapper for each would add surface area with no users.

Where a group above shows a Python entry and no R counterpart, that is the reason.

## Where the numbers come from

A clock is an architecture plus a coefficient set. Every architecture here is written from its
published description; no clock implementation is imported from another package. All 68 clocks
ship their coefficients inside the package, or are formulas with none to ship, and every one
scores offline.

```r
list_clocks()                    # the catalogue, with each clock's scale and source
clock_info("horvath2013")        # one clock: features, scale, provenance, caveats
```

## Further reading

- [Your first run](https://bhagesh-h.github.io/FALCONAge/start/quickstart.html): the full walkthrough
- [Loading your data](https://bhagesh-h.github.io/FALCONAge/data/loading.html): supported input formats
- [Choosing a clock](https://bhagesh-h.github.io/FALCONAge/clocks/choosing.html): by research question and by what the scale permits
- [Clock catalogue](https://bhagesh-h.github.io/FALCONAge/clocks/catalogue.html): all 68 clocks, generated from the registry
- [Published formulas](https://bhagesh-h.github.io/FALCONAge/methods/formulas.html): the equations and the published constants
- [Architecture](https://bhagesh-h.github.io/FALCONAge/dev/index.html): which file computes what
- [Python reference](https://bhagesh-h.github.io/FALCONAge/reference/index.html): the other half of the table above

## Citation

`citation("FALCONAge")`, or [CITATION.cff](https://github.com/bhagesh-h/FALCONAge/blob/main/CITATION.cff).

The clock FALCONAge computed for you is somebody else's work, and citing FALCONAge does not cite
it. `cite_clock("horvath2013", "bibtex")` returns the reference for any clock in the registry.
