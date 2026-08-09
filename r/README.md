# FALCONAge for R

<!-- badges: start -->
[![R-CMD-check](https://github.com/bhagesh-h/FALCONAge/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/bhagesh-h/FALCONAge/actions/workflows/R-CMD-check.yaml)
[![R 4.1+](https://img.shields.io/badge/R-4.1%2B-blue)](https://www.r-project.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
<!-- badges: end -->

Biological age and aging-clock scoring from DNA methylation and clinical chemistry, against a
catalogue of 161 published clocks.

This is the R half. It is a native R package — S3 classes, `data.frame` in and out, roxygen2
documentation, ggplot2 figures — but it does not reimplement anything. Every number comes from one
Python core that this package calls through reticulate, so **an R result and a Python result are
the same bits**, and the test suite asserts that at tolerance exactly zero rather than to six
decimals.

That design decision is the reason this page exists in the form it does: if you are reading the R
reference and something is unclear, [the Python reference](../reference/index.html) documents the
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
falconage_config()    # what resolved: versions, devices, registry size
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
manifest(res)                             # versions, device, coefficient checksums

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
identical names would make one language read like a translation of the other — which is exactly
what the reticulate bridge exists to avoid.

So the mapping is worth having explicitly. Every row is the same computation reached two ways.

<!-- BEGIN GENERATED: api-map -->

| | R | Python |
|---|---|---|
| **Reading data** | `read_betas()` | `fa.read_betas()` |
|  | `read_series_matrix()` | `fa.read_series_matrix()` |
|  | `read_clinical()` | `fa.read_clinical()` |
|  | `read_rrbs_dir()` | `fa.read_rrbs_dir()` |
|  | `falcon_data()` | `fa.FalconData()` |
|  | `obs()` | — |
|  | `write_h5ad()` | — |
|  | — | `fa.read()` |
|  | — | `fa.read_bedmethyl()` |
|  | — | `fa.read_bedmethyl_dir()` |
|  | — | `fa.read_panel()` |
|  | — | `fa.read_computage_bench()` |
|  | — | `fa.list_computage_bench()` |
|  | — | `fa.preprocess.read_olink()` |
|  | — | `fa.preprocess.read_somascan()` |
|  | — | `fa.preprocess.read_counts()` |
| **Raw arrays** | — | `fa.idat_to_betas()` |
|  | — | `fa.read_idat_dir()` |
|  | — | `fa.preprocess.poobah()` |
|  | — | `fa.preprocess.noob()` |
|  | — | `fa.preprocess.dye_bias()` |
|  | — | `fa.preprocess.bmiq()` |
|  | — | `fa.preprocess.RawSignal()` |
|  | — | `fa.preprocess.load_manifest()` |
|  | — | `fa.preprocess.manifest_record()` |
|  | — | `fa.preprocess.load_mask()` |
|  | — | `fa.preprocess.mask_report()` |
|  | — | `fa.preprocess.apply_mask()` |
| **Preprocessing** | `prepare()` | `fa.prepare()` |
|  | `qc()` | `fa.qc()` |
|  | `probe_loss()` | — |
|  | `fit_batch_reference()` | — |
|  | `apply_batch_reference()` | — |
|  | — | `fa.prepare_clinical()` |
|  | — | `fa.probe_loss()` |
|  | — | `fa.preprocess.aggregate_replicate_probes()` |
|  | — | `fa.preprocess.impute()` |
|  | — | `fa.fit_batch_reference()` |
|  | — | `fa.apply_batch_reference()` |
|  | — | `fa.preprocess.BatchReference()` |
|  | — | `fa.preprocess.prepare_proteomic()` |
|  | — | `fa.preprocess.prepare_transcriptomic()` |
|  | — | `fa.preprocess.rle_normalise()` |
|  | — | `fa.preprocess.yugene()` |
|  | — | `fa.preprocess.median_centre()` |
| **Uncertainty** | `technical_se()` | — |
|  | `interval()` | — |
|  | `conformal_interval()` | — |
|  | `icc_from_replicates()` | — |
|  | — | `fa.technical_se()` |
|  | — | `fa.interval()` |
|  | — | `fa.conformal_interval()` |
|  | — | `fa.icc_from_replicates()` |
|  | — | `fa.uncertainty.load_probe_icc()` |
|  | — | `fa.uncertainty.probe_icc_source()` |
|  | — | `fa.uncertainty.load_conformal()` |
| **Study design** | `power_n()` | — |
|  | `consensus()` | — |
|  | — | `fa.power()` |
|  | — | `fa.detectable_effect()` |
|  | — | `fa.consensus()` |
|  | — | `fa.registry.evidence()` |
| **Scoring** | `score()` | `fa.score()` |
|  | `combine()` | `fa.combine()` |
|  | `as.data.frame.falcon_result()` | `fa.FalconResult()` |
|  | `coverage()` | — |
|  | `manifest()` | — |
|  | `interpretation()` | — |
|  | `write_results()` | — |
| **The clock catalogue** | `list_clocks()` | `fa.registry.ClockRegistry()` |
|  | `clock_info()` | `fa.registry.Clock()` |
|  | `register_local_weights()` | `fa.registry.register_local_weights()` |
|  | `cite_clock()` | — |
|  | `compatible_clocks()` | — |
|  | — | `fa.registry.load()` |
| **Analysis** | `acceleration()` | `fa.acceleration()` |
|  | `associate()` | `fa.associate()` |
|  | `cox_hazard()` | `fa.cox_hazard()` |
|  | `agreement()` | `fa.agreement()` |
|  | `icc()` | `fa.icc()` |
|  | `run_benchmark()` | `fa.run_benchmark()` |
|  | `cell_composition()` | — |
|  | — | `fa.cell_composition()` |
| **Clinical references** | `fit_kdm()` | `fa.models.fit_kdm()` |
|  | `fit_hd()` | `fa.models.fit_hd()` |
|  | — | `fa.models.clinical.phenoage()` |
| **Model architectures** | — | `fa.models.AggregationClock()` |
|  | — | `fa.models.NeuralClock()` |
|  | — | `fa.models.read_neural_weights()` |
|  | — | `fa.models.PCLinearClock()` |
|  | — | `fa.models.read_rotation()` |
|  | — | `fa.models.ScAgeReference()` |
|  | — | `fa.models.fit_scage_reference()` |
|  | — | `fa.models.scage()` |
| **Figures and reports** | `plot_ba_vs_ca()` | `fa.plot.ba_vs_ca()` |
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
|  | `falcon_scheme()` | — |
|  | `falcon_theme()` | — |
|  | — | `fa.plot.missingness()` |
|  | — | `fa.plot.beta_density()` |
|  | — | `fa.plot.study_comparison()` |
|  | — | `fa.plot.reliability_forest()` |
|  | — | `fa.plot.score_interval()` |
|  | — | `fa.plot.platform_bias()` |
|  | — | `fa.plot.consensus_plot()` |
|  | — | `fa.plot.save_all()` |
| **Downloading public data** | `download()` | `fa.download()` |
|  | `cache_info()` | `fa.download.cache_info()` |
|  | `clear_cache()` | `fa.download.clear_cache()` |
| **Configuration and setup** | `falconage_config()` | `fa.config()` |
|  | `falconage_install()` | — |
|  | `falconage_available()` | — |
|  | — | `fa.configure()` |
|  | — | `fa.core.resolve()` |
|  | — | `fa.core.describe()` |
|  | — | `fa.FalconConfig()` |
|  | — | `fa.RunManifest()` |

<!-- END GENERATED: api-map -->

## What the R side does not have

`download()` and the preprocessing entry points are thinner here than in Python, and a few things
are Python-only by design rather than by omission — an R user reaches them through the same core,
and wrapping every one in an S3 method would be surface area with no reader.

Where a group above shows a Python entry and no R counterpart, that is the reason.

## Where the numbers come from

A clock is an architecture plus a coefficient set, and the two have very different licences.

Every **architecture** here is written from its published description — no clock implementation is
imported from another package. **Coefficients** are fitted data and cannot be written, only
obtained, so the 161 clocks fall into three tiers:

| Tier | n | What you do |
|---|---:|---|
| A | 23 | Nothing. Coefficients ship inside the package, or the clock is a formula with none to ship. |
| B | 110 | Catalogued, but no primary source has been traced, so no coefficients ship. |
| C | 28 | Obtain a coefficient file and register it with `register_local_weights()`. The architecture is implemented and tested. |

```r
list_clocks(tier = "A")       # what runs offline, right now
list_clocks(tier = "C")       # what needs a licence, and where to get it
clock_info("grimage2")        # why, and which open clock answers the same question
```

## Further reading

- [Getting started](../guide/FALCONAge.html) — the full walkthrough, both languages side by side
- [Choosing a clock](../guide/clocks.html) — by the question you asked, and by what the scale permits
- [Clock catalogue](../clocks.html) — all 161, generated from the registry that scores them
- [The science of aging clocks](../science.html) — the biology, the equations and the published constants
- [Architecture](../architecture.html) — which file computes what, and how much of it exists
- [Python reference](../reference/index.html) — the other half of the table above

## Citation

`citation("FALCONAge")`, or [CITATION.cff](https://github.com/bhagesh-h/FALCONAge/blob/main/CITATION.cff).

The clock FALCONAge computed for you is somebody else's work, and citing FALCONAge does not cite
it. `cite_clock("grimage2", "bibtex")` returns the reference for any clock in the registry.
