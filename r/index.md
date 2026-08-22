# FALCONAge for R

Biological age and aging-clock scoring from DNA methylation and clinical
chemistry, against a catalogue of 161 published clocks.

This is the R half. It is a native R package: S3 classes, `data.frame`
in and out, roxygen2 documentation, ggplot2 figures, but it does not
reimplement anything. Every number comes from one Python core that this
package calls through reticulate, so **an R result and a Python result
are the same bits**, and the test suite asserts that at tolerance
exactly zero rather than to six decimals.

That design decision is the reason this page exists in the form it does:
if you are reading the R reference and something is unclear, [the Python
reference](https://bhagesh-h.github.io/FALCONAge/reference/index.md)
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
falconage_config()    # what resolved: versions, devices, registry size
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
manifest(res)                             # versions, device, coefficient checksums

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
| **Reading data** | [`read_betas()`](https://bhagesh-h.github.io/FALCONAge/r/reference/read_betas.md) | `fa.read_betas()` |
|  | [`read_series_matrix()`](https://bhagesh-h.github.io/FALCONAge/r/reference/read_series_matrix.md) | `fa.read_series_matrix()` |
|  | [`read_clinical()`](https://bhagesh-h.github.io/FALCONAge/r/reference/read_clinical.md) | `fa.read_clinical()` |
|  | [`read_rrbs_dir()`](https://bhagesh-h.github.io/FALCONAge/r/reference/read_rrbs_dir.md) | `fa.read_rrbs_dir()` |
|  | [`falcon_data()`](https://bhagesh-h.github.io/FALCONAge/r/reference/falcon_data.md) | `fa.FalconData()` |
|  | [`obs()`](https://bhagesh-h.github.io/FALCONAge/r/reference/obs.md) | `data.obs` |
|  | [`write_h5ad()`](https://bhagesh-h.github.io/FALCONAge/r/reference/write_h5ad.md) | `data.write_h5ad()` |
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
| **Preprocessing** | [`prepare()`](https://bhagesh-h.github.io/FALCONAge/r/reference/prepare.md) | `fa.prepare()` |
|  | [`qc()`](https://bhagesh-h.github.io/FALCONAge/r/reference/qc.md) | `fa.qc()` |
|  | [`probe_loss()`](https://bhagesh-h.github.io/FALCONAge/r/reference/probe_loss.md) | `fa.probe_loss()` |
|  | [`fit_batch_reference()`](https://bhagesh-h.github.io/FALCONAge/r/reference/fit_batch_reference.md) | `fa.fit_batch_reference()` |
|  | [`apply_batch_reference()`](https://bhagesh-h.github.io/FALCONAge/r/reference/apply_batch_reference.md) | `fa.apply_batch_reference()` |
|  | — | `fa.prepare_clinical()` |
|  | — | `fa.preprocess.aggregate_replicate_probes()` |
|  | — | `fa.preprocess.impute()` |
|  | — | `fa.preprocess.BatchReference()` |
|  | — | `fa.preprocess.prepare_proteomic()` |
|  | — | `fa.preprocess.prepare_transcriptomic()` |
|  | — | `fa.preprocess.rle_normalise()` |
|  | — | `fa.preprocess.yugene()` |
|  | — | `fa.preprocess.median_centre()` |
| **Uncertainty** | [`technical_se()`](https://bhagesh-h.github.io/FALCONAge/r/reference/technical_se.md) | `fa.technical_se()` |
|  | [`interval()`](https://bhagesh-h.github.io/FALCONAge/r/reference/interval.md) | `fa.interval()` |
|  | [`conformal_interval()`](https://bhagesh-h.github.io/FALCONAge/r/reference/conformal_interval.md) | `fa.conformal_interval()` |
|  | [`icc_from_replicates()`](https://bhagesh-h.github.io/FALCONAge/r/reference/icc_from_replicates.md) | `fa.icc_from_replicates()` |
|  | — | `fa.uncertainty.load_probe_icc()` |
|  | — | `fa.uncertainty.probe_icc_source()` |
|  | — | `fa.uncertainty.load_conformal()` |
| **Study design** | [`power_n()`](https://bhagesh-h.github.io/FALCONAge/r/reference/power_n.md) | `fa.power()` |
|  | [`consensus()`](https://bhagesh-h.github.io/FALCONAge/r/reference/consensus.md) | `fa.consensus()` |
|  | — | `fa.detectable_effect()` |
|  | — | `fa.registry.evidence()` |
| **Scoring** | [`score()`](https://bhagesh-h.github.io/FALCONAge/r/reference/score.md) | `fa.score()` |
|  | [`combine()`](https://bhagesh-h.github.io/FALCONAge/r/reference/combine.md) | `fa.combine()` |
|  | [`as.data.frame.falcon_result()`](https://bhagesh-h.github.io/FALCONAge/r/reference/as.data.frame.falcon_result.md) | `fa.FalconResult()` |
|  | [`interpretation()`](https://bhagesh-h.github.io/FALCONAge/r/reference/interpretation.md) | `res.interpretation()` |
|  | [`coverage()`](https://bhagesh-h.github.io/FALCONAge/r/reference/coverage.md) | `res.coverage` |
|  | [`manifest()`](https://bhagesh-h.github.io/FALCONAge/r/reference/manifest.md) | `res.manifest` |
|  | [`write_results()`](https://bhagesh-h.github.io/FALCONAge/r/reference/write_results.md) | `res.write()` |
| **The clock catalogue** | [`list_clocks()`](https://bhagesh-h.github.io/FALCONAge/r/reference/list_clocks.md) | `fa.registry.ClockRegistry()` |
|  | [`clock_info()`](https://bhagesh-h.github.io/FALCONAge/r/reference/clock_info.md) | `fa.registry.Clock()` |
|  | [`register_local_weights()`](https://bhagesh-h.github.io/FALCONAge/r/reference/register_local_weights.md) | `fa.registry.register_local_weights()` |
|  | [`cite_clock()`](https://bhagesh-h.github.io/FALCONAge/r/reference/cite_clock.md) | `reg.get(id).cite()` |
|  | [`compatible_clocks()`](https://bhagesh-h.github.io/FALCONAge/r/reference/compatible_clocks.md) | `reg.compatible_with(data)` |
|  | — | `fa.registry.load()` |
| **Analysis** | [`acceleration()`](https://bhagesh-h.github.io/FALCONAge/r/reference/acceleration.md) | `fa.acceleration()` |
|  | [`cell_composition()`](https://bhagesh-h.github.io/FALCONAge/r/reference/cell_composition.md) | `fa.cell_composition()` |
|  | [`associate()`](https://bhagesh-h.github.io/FALCONAge/r/reference/associate.md) | `fa.associate()` |
|  | [`cox_hazard()`](https://bhagesh-h.github.io/FALCONAge/r/reference/cox_hazard.md) | `fa.cox_hazard()` |
|  | [`agreement()`](https://bhagesh-h.github.io/FALCONAge/r/reference/agreement.md) | `fa.agreement()` |
|  | [`icc()`](https://bhagesh-h.github.io/FALCONAge/r/reference/icc.md) | `fa.icc()` |
|  | [`run_benchmark()`](https://bhagesh-h.github.io/FALCONAge/r/reference/run_benchmark.md) | `fa.run_benchmark()` |
| **Clinical references** | [`fit_kdm()`](https://bhagesh-h.github.io/FALCONAge/r/reference/fit_kdm.md) | `fa.models.fit_kdm()` |
|  | [`fit_hd()`](https://bhagesh-h.github.io/FALCONAge/r/reference/fit_hd.md) | `fa.models.fit_hd()` |
|  | — | `fa.models.clinical.phenoage()` |
| **Model architectures** | — | `fa.models.AggregationClock()` |
|  | — | `fa.models.NeuralClock()` |
|  | — | `fa.models.read_neural_weights()` |
|  | — | `fa.models.PCLinearClock()` |
|  | — | `fa.models.read_rotation()` |
|  | — | `fa.models.ScAgeReference()` |
|  | — | `fa.models.fit_scage_reference()` |
|  | — | `fa.models.scage()` |
| **Figures and reports** | [`plot_ba_vs_ca()`](https://bhagesh-h.github.io/FALCONAge/r/reference/plot_ba_vs_ca.md) | `fa.plot.ba_vs_ca()` |
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
|  | [`falcon_scheme()`](https://bhagesh-h.github.io/FALCONAge/r/reference/falcon_scheme.md) | *R only: loads the shared palette into R* |
|  | [`falcon_theme()`](https://bhagesh-h.github.io/FALCONAge/r/reference/falcon_theme.md) | *R only: a ggplot2 theme; Python styles via matplotlib* |
|  | — | `fa.plot.missingness()` |
|  | — | `fa.plot.beta_density()` |
|  | — | `fa.plot.study_comparison()` |
|  | — | `fa.plot.reliability_forest()` |
|  | — | `fa.plot.score_interval()` |
|  | — | `fa.plot.platform_bias()` |
|  | — | `fa.plot.consensus_plot()` |
|  | — | `fa.plot.save_all()` |
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
and the preprocessing entry points are thinner here than in Python, and
a few things are Python-only by design rather than by omission, an R
user reaches them through the same core, and wrapping every one in an S3
method would be surface area with no reader.

Where a group above shows a Python entry and no R counterpart: that is
the reason.

## Where the numbers come from

A clock is an architecture plus a coefficient set, and the two have very
different licences.

Every **architecture** here is written from its published description,
no clock implementation is imported from another package.
**Coefficients** are fitted data and cannot be written, only obtained,
so the 161 clocks fall into three tiers:

| Tier | n | What you do |
|----|---:|----|
| A | 23 | Nothing. Coefficients ship inside the package, or the clock is a formula with none to ship. |
| B | 110 | Catalogued, but no primary source has been traced, so no coefficients ship. |
| C | 28 | Obtain a coefficient file and register it with [`register_local_weights()`](https://bhagesh-h.github.io/FALCONAge/r/reference/register_local_weights.md). The architecture is implemented and tested. |

``` r

list_clocks(tier = "A")       # what runs offline, right now
list_clocks(tier = "C")       # what needs a licence, and where to get it
clock_info("grimage2")        # why, and which open clock answers the same question
```

## Further reading

- [Getting
  started](https://bhagesh-h.github.io/FALCONAge/guide/FALCONAge.md),
  the full walkthrough, both languages side by side
- [Choosing a
  clock](https://bhagesh-h.github.io/FALCONAge/guide/clocks.md), by the
  question you asked, and by what the scale permits
- [Clock catalogue](https://bhagesh-h.github.io/FALCONAge/clocks.md),
  all 161, generated from the registry that scores them
- [The science of aging
  clocks](https://bhagesh-h.github.io/FALCONAge/science.md), the
  biology, the equations and the published constants
- [Architecture](https://bhagesh-h.github.io/FALCONAge/architecture.md),
  which file computes what, and how much of it exists
- [Python
  reference](https://bhagesh-h.github.io/FALCONAge/reference/index.md),
  the other half of the table above

## Citation

`citation("FALCONAge")`, or
[CITATION.cff](https://github.com/bhagesh-h/FALCONAge/blob/main/CITATION.cff).

The clock FALCONAge computed for you is somebody else’s work, and citing
FALCONAge does not cite it. `cite_clock("grimage2", "bibtex")` returns
the reference for any clock in the registry.
