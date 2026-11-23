# FALCONAge for R

Biological age and aging-clock scoring from DNA methylation, clinical
chemistry and NMR metabolomics, against a catalogue of 68 published
human clocks. The input is data that is already preprocessed and quality
controlled, such as the beta matrix from an array pipeline.

This is the R half. It is a native R package: S3 classes, `data.frame`
in and out, roxygen2 documentation, ggplot2 figures, but it does not
reimplement anything. Every number comes from one Python core that this
package calls through reticulate, so **an R result and a Python result
are the same bits**, and the test suite asserts that at tolerance
exactly zero rather than to six decimals.

That design decision is the reason this page exists in the form it does:
if you are reading the R reference and something is unclear, [the Python
reference](https://bhagesh-h.github.io/FALCONAge/reference/index.html)
documents the same computation, and the table below tells you which
entry to look at.

## Install

Not on CRAN yet. The package is the `r/` subdirectory of the repository,
hence `subdir`:

``` r

remotes::install_github("bhagesh-h/FALCONAge", subdir = "r")
# or, resolving dependencies better and taking the subdirectory inline:
pak::pak("bhagesh-h/FALCONAge/r")
```

Then, once per machine:

``` r

library(FALCONAge)
falconage_install()   # builds the managed Python environment; a few minutes
falconage_config()    # what resolved: versions, registry size
```

[`falconage_install()`](https://bhagesh-h.github.io/FALCONAge/r/reference/falconage_install.md)
fetches the Python core from the same GitHub tag this package was built
from, so the two halves cannot drift apart.
[`falconage_config()`](https://bhagesh-h.github.io/FALCONAge/r/reference/falconage_config.md)
is the first thing to run when anything looks wrong.

## A whole analysis

``` r

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

`clocks = "compatible"` scores what the data can support and reports the
rest as skipped with a reason. Naming clocks explicitly is different:
then every one must work, because an explicit request should never be
dropped quietly.

## Reading the R reference alongside the Python one

The two APIs are one-to-one in intent and deliberately not in spelling.
R gets `as.data.frame(res, form = "long")` where Python gets
`FalconResult.long()`, because forcing identical names would make one
language read like a translation of the other, which is exactly what the
reticulate bridge exists to avoid.

So the mapping is worth having explicitly. Every row is the same
computation reached two ways.

|  | R | Python |
|----|----|----|
| **Start here** | [`as.data.frame.falcon_result()`](https://bhagesh-h.github.io/FALCONAge/r/reference/as.data.frame.falcon_result.md) | `fa.FalconResult()` |
|  | [`read_betas()`](https://bhagesh-h.github.io/FALCONAge/r/reference/read_betas.md) | `fa.read_betas()` |
|  | [`prepare()`](https://bhagesh-h.github.io/FALCONAge/r/reference/prepare.md) | `fa.prepare()` |
|  | [`technical_se()`](https://bhagesh-h.github.io/FALCONAge/r/reference/technical_se.md) | `fa.technical_se()` |
|  | [`conformal_interval()`](https://bhagesh-h.github.io/FALCONAge/r/reference/conformal_interval.md) | `fa.conformal_interval()` |
|  | [`consensus()`](https://bhagesh-h.github.io/FALCONAge/r/reference/consensus.md) | `fa.consensus()` |
|  | [`score()`](https://bhagesh-h.github.io/FALCONAge/r/reference/score.md) | `fa.score()` |
|  | [`interpretation()`](https://bhagesh-h.github.io/FALCONAge/r/reference/interpretation.md) | `res.interpretation()` |
|  | [`acceleration()`](https://bhagesh-h.github.io/FALCONAge/r/reference/acceleration.md) | `fa.acceleration()` |
|  | [`run_report()`](https://bhagesh-h.github.io/FALCONAge/r/reference/run_report.md) | `fa.report.run_report()` |
|  | — | `fa.read()` |
|  | — | `fa.demo_data()` |
| **Reading data** | [`read_series_matrix()`](https://bhagesh-h.github.io/FALCONAge/r/reference/read_series_matrix.md) | `fa.read_series_matrix()` |
|  | [`read_clinical()`](https://bhagesh-h.github.io/FALCONAge/r/reference/read_clinical.md) | `fa.read_clinical()` |
|  | [`falcon_data()`](https://bhagesh-h.github.io/FALCONAge/r/reference/falcon_data.md) | `fa.FalconData()` |
|  | [`obs()`](https://bhagesh-h.github.io/FALCONAge/r/reference/obs.md) | `data.obs` |
|  | [`write_h5ad()`](https://bhagesh-h.github.io/FALCONAge/r/reference/write_h5ad.md) | `data.write_h5ad()` |
|  | — | `fa.read_nightingale()` |
|  | — | `fa.read_panel()` |
|  | — | `fa.read_computage_bench()` |
|  | — | `fa.list_computage_bench()` |
| **Preparing data** | [`horvath_normalise()`](https://bhagesh-h.github.io/FALCONAge/r/reference/horvath_normalise.md) | `fa.preprocess.horvath_normalise()` |
|  | [`qc()`](https://bhagesh-h.github.io/FALCONAge/r/reference/qc.md) | `fa.qc()` |
|  | [`probe_loss()`](https://bhagesh-h.github.io/FALCONAge/r/reference/probe_loss.md) | `fa.probe_loss()` |
|  | — | `fa.prepare_clinical()` |
|  | — | `fa.preprocess.aggregate_replicate_probes()` |
|  | — | `fa.preprocess.platform_probes()` |
|  | — | `fa.preprocess.impute()` |
| **Uncertainty** | [`interval()`](https://bhagesh-h.github.io/FALCONAge/r/reference/interval.md) | `fa.interval()` |
|  | [`icc_from_replicates()`](https://bhagesh-h.github.io/FALCONAge/r/reference/icc_from_replicates.md) | `fa.icc_from_replicates()` |
|  | — | `fa.uncertainty.load_probe_icc()` |
|  | — | `fa.uncertainty.probe_icc_source()` |
|  | — | `fa.uncertainty.load_conformal()` |
| **Study design** | [`power_n()`](https://bhagesh-h.github.io/FALCONAge/r/reference/power_n.md) | `fa.power()` |
|  | — | `fa.detectable_effect()` |
|  | — | `fa.registry.evidence()` |
| **Scoring** | [`combine()`](https://bhagesh-h.github.io/FALCONAge/r/reference/combine.md) | `fa.combine()` |
|  | [`coverage()`](https://bhagesh-h.github.io/FALCONAge/r/reference/coverage.md) | `res.coverage` |
|  | [`manifest()`](https://bhagesh-h.github.io/FALCONAge/r/reference/manifest.md) | `res.manifest` |
|  | [`write_results()`](https://bhagesh-h.github.io/FALCONAge/r/reference/write_results.md) | `res.write()` |
| **The clock catalogue** | [`list_clocks()`](https://bhagesh-h.github.io/FALCONAge/r/reference/list_clocks.md) | `fa.registry.ClockRegistry()` |
|  | [`clock_info()`](https://bhagesh-h.github.io/FALCONAge/r/reference/clock_info.md) | `fa.registry.Clock()` |
|  | [`register_local_weights()`](https://bhagesh-h.github.io/FALCONAge/r/reference/register_local_weights.md) | `fa.registry.register_local_weights()` |
|  | [`cite_clock()`](https://bhagesh-h.github.io/FALCONAge/r/reference/cite_clock.md) | `reg.get(id).cite()` |
|  | [`compatible_clocks()`](https://bhagesh-h.github.io/FALCONAge/r/reference/compatible_clocks.md) | `reg.compatible_with(data)` |
|  | — | `fa.registry.load()` |
| **Analysis** | [`cell_composition()`](https://bhagesh-h.github.io/FALCONAge/r/reference/cell_composition.md) | `fa.cell_composition()` |
|  | [`associate()`](https://bhagesh-h.github.io/FALCONAge/r/reference/associate.md) | `fa.associate()` |
|  | [`cox_hazard()`](https://bhagesh-h.github.io/FALCONAge/r/reference/cox_hazard.md) | `fa.cox_hazard()` |
|  | [`agreement()`](https://bhagesh-h.github.io/FALCONAge/r/reference/agreement.md) | `fa.agreement()` |
|  | [`icc()`](https://bhagesh-h.github.io/FALCONAge/r/reference/icc.md) | `fa.icc()` |
|  | [`run_benchmark()`](https://bhagesh-h.github.io/FALCONAge/r/reference/run_benchmark.md) | `fa.run_benchmark()` |
|  | — | `fa.leave_one_marker_out()` |
| **Your own clocks** | [`compare_learners()`](https://bhagesh-h.github.io/FALCONAge/r/reference/compare_learners.md) | `fa.compare_learners()` |
|  | [`fit_clock()`](https://bhagesh-h.github.io/FALCONAge/r/reference/fit_clock.md) | `fa.fit_clock()` |
|  | [`stack_clocks()`](https://bhagesh-h.github.io/FALCONAge/r/reference/stack_clocks.md) | `fa.stack_clocks()` |
|  | [`add_clock()`](https://bhagesh-h.github.io/FALCONAge/r/reference/add_clock.md) | `fa.add_clock()` |
|  | [`clock_to_file()`](https://bhagesh-h.github.io/FALCONAge/r/reference/clock_to_file.md) | *the `.to_clock()` method of a fit or a stack* |
| **Clinical references** | [`fit_kdm()`](https://bhagesh-h.github.io/FALCONAge/r/reference/fit_kdm.md) | `fa.models.fit_kdm()` |
|  | [`fit_hd()`](https://bhagesh-h.github.io/FALCONAge/r/reference/fit_hd.md) | `fa.models.fit_hd()` |
|  | — | `fa.models.clinical.phenoage()` |
|  | — | `fa.blood_count_ratios()` |
| **Model architectures** | — | `fa.models.AggregationClock()` |
|  | — | `fa.models.NeuralClock()` |
|  | — | `fa.models.read_neural_weights()` |
|  | — | `fa.models.ScAgeReference()` |
|  | — | `fa.models.fit_scage_reference()` |
|  | — | `fa.models.scage()` |
| **Figures and reports** | [`plot_clock_coverage()`](https://bhagesh-h.github.io/FALCONAge/r/reference/plot_clock_coverage.md) | `fa.plot.clock_coverage_manhattan()` |
|  | [`save_figure()`](https://bhagesh-h.github.io/FALCONAge/r/reference/save_figure.md) | `fa.plot.save_figure()` |
|  | [`figure_text()`](https://bhagesh-h.github.io/FALCONAge/r/reference/figure_text.md) | `fa.plot.figure_text()` |
|  | [`plot_ba_vs_ca()`](https://bhagesh-h.github.io/FALCONAge/r/reference/plot_ba_vs_ca.md) | `fa.plot.ba_vs_ca()` |
|  | [`plot_bland_altman()`](https://bhagesh-h.github.io/FALCONAge/r/reference/plot_bland_altman.md) | `fa.plot.bland_altman()` |
|  | [`plot_calibration()`](https://bhagesh-h.github.io/FALCONAge/r/reference/plot_calibration.md) | `fa.plot.calibration()` |
|  | [`plot_acceleration()`](https://bhagesh-h.github.io/FALCONAge/r/reference/plot_acceleration.md) | `fa.plot.acceleration_density()` |
|  | [`plot_acceleration_by_group()`](https://bhagesh-h.github.io/FALCONAge/r/reference/plot_acceleration_by_group.md) | `fa.plot.acceleration_by_group()` |
|  | [`plot_acceleration_heatmap()`](https://bhagesh-h.github.io/FALCONAge/r/reference/plot_acceleration_heatmap.md) | `fa.plot.acceleration_heatmap()` |
|  | [`plot_forest()`](https://bhagesh-h.github.io/FALCONAge/r/reference/plot_forest.md) | `fa.plot.forest()` |
|  | [`plot_kaplan_meier()`](https://bhagesh-h.github.io/FALCONAge/r/reference/plot_kaplan_meier.md) | `fa.plot.kaplan_meier()` |
|  | [`plot_volcano()`](https://bhagesh-h.github.io/FALCONAge/r/reference/plot_volcano.md) | `fa.plot.volcano()` |
|  | [`plot_agreement()`](https://bhagesh-h.github.io/FALCONAge/r/reference/plot_agreement.md) | `fa.plot.clock_corr()` |
|  | [`plot_clock_radar()`](https://bhagesh-h.github.io/FALCONAge/r/reference/plot_clock_radar.md) | `fa.plot.clock_radar()` |
|  | [`plot_clock_chord()`](https://bhagesh-h.github.io/FALCONAge/r/reference/plot_clock_chord.md) | `fa.plot.clock_chord()` |
|  | [`plot_clock_pca()`](https://bhagesh-h.github.io/FALCONAge/r/reference/plot_clock_pca.md) | `fa.plot.clock_pca()` |
|  | [`plot_clock_atlas()`](https://bhagesh-h.github.io/FALCONAge/r/reference/plot_clock_atlas.md) | `fa.plot.clock_atlas()` |
|  | [`plot_coverage()`](https://bhagesh-h.github.io/FALCONAge/r/reference/plot_coverage.md) | `fa.plot.coverage_bar()` |
|  | [`plot_by_platform()`](https://bhagesh-h.github.io/FALCONAge/r/reference/plot_by_platform.md) | `fa.plot.platform_comparison()` |
|  | [`plot_benchmark()`](https://bhagesh-h.github.io/FALCONAge/r/reference/plot_benchmark.md) | `fa.plot.benchmark_bars()` |
|  | [`plot_benchmark_error_bias()`](https://bhagesh-h.github.io/FALCONAge/r/reference/plot_benchmark_error_bias.md) | `fa.plot.benchmark_error_bias()` |
|  | [`plot_benchmark_heatmap()`](https://bhagesh-h.github.io/FALCONAge/r/reference/plot_benchmark_heatmap.md) | `fa.plot.benchmark_heatmap()` |
|  | [`falcon_palette()`](https://bhagesh-h.github.io/FALCONAge/r/reference/falcon_palette.md) | `fa.plot.palette()` |
|  | [`report()`](https://bhagesh-h.github.io/FALCONAge/r/reference/report.md) | `fa.report.write_report()` |
|  | [`quarto_report()`](https://bhagesh-h.github.io/FALCONAge/r/reference/quarto_report.md) | `fa.report.write_quarto_report()` |
|  | [`falcon_scheme()`](https://bhagesh-h.github.io/FALCONAge/r/reference/falcon_scheme.md) | *R only: loads the shared palette into R* |
|  | [`falcon_theme()`](https://bhagesh-h.github.io/FALCONAge/r/reference/falcon_theme.md) | *R only: a ggplot2 theme; Python styles via matplotlib* |
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
| **Downloading public data** | [`download()`](https://bhagesh-h.github.io/FALCONAge/r/reference/download.md) | `fa.download()` |
|  | [`cache_info()`](https://bhagesh-h.github.io/FALCONAge/r/reference/cache_info.md) | `fa.download.cache_info()` |
|  | [`clear_cache()`](https://bhagesh-h.github.io/FALCONAge/r/reference/cache_info.md) | `fa.download.clear_cache()` |
| **Configuration and setup** | [`falconage_config()`](https://bhagesh-h.github.io/FALCONAge/r/reference/falconage_config.md) | `fa.config()` |
|  | [`falconage_install()`](https://bhagesh-h.github.io/FALCONAge/r/reference/falconage_install.md) | *R only: builds the Python environment R calls into* |
|  | [`falconage_available()`](https://bhagesh-h.github.io/FALCONAge/r/reference/falconage_available.md) | *R only: is that environment resolvable yet* |
|  | — | `fa.configure()` |
|  | — | `fa.core.resolve()` |
|  | — | `fa.core.describe()` |
|  | — | `fa.FalconConfig()` |
|  | — | `fa.RunManifest()` |

## What the R side does not have

[`download()`](https://bhagesh-h.github.io/FALCONAge/r/reference/download.md)
and a few model architectures are Python-only by design: an R user
reaches them through the same core, and an S3 wrapper for each would add
surface area with no users.

Where a group above shows a Python entry and no R counterpart, that is
the reason.

## Where the numbers come from

A clock is an architecture plus a coefficient set. Every architecture
here is written from its published description; no clock implementation
is imported from another package. All 68 clocks ship their coefficients
inside the package, or are formulas with none to ship, and every one
scores offline.

``` r

list_clocks()                    # the catalogue, with each clock's scale and source
clock_info("horvath2013")        # one clock: features, scale, provenance, caveats
```

## Further reading

- [Your first
  run](https://bhagesh-h.github.io/FALCONAge/start/quickstart.html): the
  full walkthrough
- [Loading your
  data](https://bhagesh-h.github.io/FALCONAge/data/loading.html):
  supported input formats
- [Choosing a
  clock](https://bhagesh-h.github.io/FALCONAge/clocks/choosing.html): by
  research question and by what the scale permits
- [Clock
  catalogue](https://bhagesh-h.github.io/FALCONAge/clocks/catalogue.html):
  all 68 clocks, generated from the registry
- [Published
  formulas](https://bhagesh-h.github.io/FALCONAge/methods/formulas.html):
  the equations and the published constants
- [Architecture](https://bhagesh-h.github.io/FALCONAge/dev/index.html):
  which file computes what
- [Python
  reference](https://bhagesh-h.github.io/FALCONAge/reference/index.html):
  the other half of the table above

## Citation

`citation("FALCONAge")`, or
[CITATION.cff](https://github.com/bhagesh-h/FALCONAge/blob/main/CITATION.cff).

The clock FALCONAge computed for you is somebody else’s work, and citing
FALCONAge does not cite it. `cite_clock("horvath2013", "bibtex")`
returns the reference for any clock in the registry.
