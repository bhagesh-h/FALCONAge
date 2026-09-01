# Package index

## Reading data

Every reader returns the same container, so nothing downstream can tell
which format the data arrived in.

- [`read_betas()`](https://bhagesh-h.github.io/FALCONAge/r/reference/read_betas.md)
  : Read a beta matrix
- [`read_series_matrix()`](https://bhagesh-h.github.io/FALCONAge/r/reference/read_series_matrix.md)
  : Read a GEO series matrix
- [`read_clinical()`](https://bhagesh-h.github.io/FALCONAge/r/reference/read_clinical.md)
  : Read a clinical chemistry table
- [`read_rrbs_dir()`](https://bhagesh-h.github.io/FALCONAge/r/reference/read_rrbs_dir.md)
  : Read RRBS coverage files
- [`falcon_data()`](https://bhagesh-h.github.io/FALCONAge/r/reference/falcon_data.md)
  [`print(`*`<falcon_data>`*`)`](https://bhagesh-h.github.io/FALCONAge/r/reference/falcon_data.md)
  [`summary(`*`<falcon_data>`*`)`](https://bhagesh-h.github.io/FALCONAge/r/reference/falcon_data.md)
  : Build a dataset from an R matrix or data frame
- [`obs()`](https://bhagesh-h.github.io/FALCONAge/r/reference/obs.md) :
  Sample annotation
- [`write_h5ad()`](https://bhagesh-h.github.io/FALCONAge/r/reference/write_h5ad.md)
  [`read_h5ad()`](https://bhagesh-h.github.io/FALCONAge/r/reference/write_h5ad.md)
  : Write a dataset to the interchange format both languages read

## Preprocessing

Getting a matrix into the shape every clock assumes: bare probe
identifiers, values in the beta range, a known platform. The EPIC v2
step is mandatory rather than optional – without it a v2 dataset
overlaps almost nothing and the clocks score imputed values.

- [`prepare()`](https://bhagesh-h.github.io/FALCONAge/r/reference/prepare.md)
  : Standard methylation preprocessing
- [`qc()`](https://bhagesh-h.github.io/FALCONAge/r/reference/qc.md) :
  Quality control before scoring
- [`probe_loss()`](https://bhagesh-h.github.io/FALCONAge/r/reference/probe_loss.md)
  : What each clock has lost on this dataset, before scoring
- [`fit_batch_reference()`](https://bhagesh-h.github.io/FALCONAge/r/reference/fit_batch_reference.md)
  : Fit a frozen batch-correction reference
- [`apply_batch_reference()`](https://bhagesh-h.github.io/FALCONAge/r/reference/apply_batch_reference.md)
  : Apply a frozen batch-correction reference

## Uncertainty

How much of a score is the assay, and how far it is likely to be from
the truth. Two different questions with two different answers, kept
apart on purpose: technical replicates of the same DNA differ by up to
nine years on prominent clocks, and a point estimate says nothing about
either.

- [`technical_se()`](https://bhagesh-h.github.io/FALCONAge/r/reference/technical_se.md)
  : Technical standard error on each score
- [`interval()`](https://bhagesh-h.github.io/FALCONAge/r/reference/interval.md)
  : Scores with their measurement interval
- [`conformal_interval()`](https://bhagesh-h.github.io/FALCONAge/r/reference/conformal_interval.md)
  : Distribution-free prediction intervals against chronological age
- [`icc_from_replicates()`](https://bhagesh-h.github.io/FALCONAge/r/reference/icc_from_replicates.md)
  : Per-probe reliability from your own technical replicates

## Study design

The commands that come before the first array and after the last one.
How many samples an effect needs, given the clock’s own reliability; and
whether a difference that turned up survives the multi-clock rule that a
re-analysis of six intervention datasets says it must.

- [`power_n()`](https://bhagesh-h.github.io/FALCONAge/r/reference/power_n.md)
  : How many samples to see an effect
- [`consensus()`](https://bhagesh-h.github.io/FALCONAge/r/reference/consensus.md)
  : Does a group difference hold up across clocks?

## Scoring

The predict loop, its result object, and the provenance it carries.

- [`score()`](https://bhagesh-h.github.io/FALCONAge/r/reference/score.md)
  [`print(`*`<falcon_result>`*`)`](https://bhagesh-h.github.io/FALCONAge/r/reference/score.md)
  [`summary(`*`<falcon_result>`*`)`](https://bhagesh-h.github.io/FALCONAge/r/reference/score.md)
  : Score a dataset against one or more clocks
- [`combine()`](https://bhagesh-h.github.io/FALCONAge/r/reference/combine.md)
  : Combine per-dataset results for a benchmark across studies
- [`coverage()`](https://bhagesh-h.github.io/FALCONAge/r/reference/coverage.md)
  : Per-clock coverage and skip reasons
- [`manifest()`](https://bhagesh-h.github.io/FALCONAge/r/reference/manifest.md)
  : The run manifest
- [`interpretation()`](https://bhagesh-h.github.io/FALCONAge/r/reference/interpretation.md)
  : How to read each score in a result
- [`write_results()`](https://bhagesh-h.github.io/FALCONAge/r/reference/write_results.md)
  : Write the standard results layout
- [`as.data.frame(`*`<falcon_result>`*`)`](https://bhagesh-h.github.io/FALCONAge/r/reference/as.data.frame.falcon_result.md)
  : Scores as a data frame

## The clock catalogue

161 clocks in three availability tiers. Twenty-eight are scaffolds whose
coefficients are research-use-only; the registry says which, why, and
what open clock answers the same question.

- [`list_clocks()`](https://bhagesh-h.github.io/FALCONAge/r/reference/list_clocks.md)
  : List the clock catalogue
- [`clock_info()`](https://bhagesh-h.github.io/FALCONAge/r/reference/clock_info.md)
  : Everything the registry knows about one clock
- [`cite_clock()`](https://bhagesh-h.github.io/FALCONAge/r/reference/cite_clock.md)
  : Cite a clock
- [`compatible_clocks()`](https://bhagesh-h.github.io/FALCONAge/r/reference/compatible_clocks.md)
  : Which clocks this dataset can actually be scored on
- [`register_local_weights()`](https://bhagesh-h.github.io/FALCONAge/r/reference/register_local_weights.md)
  : Supply a coefficient file for a clock FALCONAge does not distribute

## Analysis

Age acceleration in its three conventions, association and survival
models, reliability, and the AA1/AA2 benchmark. Each is gated on the
clock’s scale type – acceleration on a pace-of-aging clock is a units
error, not a conservative choice.

- [`acceleration()`](https://bhagesh-h.github.io/FALCONAge/r/reference/acceleration.md)
  : Age acceleration
- [`cell_composition()`](https://bhagesh-h.github.io/FALCONAge/r/reference/cell_composition.md)
  : Cell-type proportions estimated in the same run
- [`associate()`](https://bhagesh-h.github.io/FALCONAge/r/reference/associate.md)
  : Associate clock scores with an outcome
- [`cox_hazard()`](https://bhagesh-h.github.io/FALCONAge/r/reference/cox_hazard.md)
  : Univariable Cox hazard ratio per clock
- [`agreement()`](https://bhagesh-h.github.io/FALCONAge/r/reference/agreement.md)
  : Between-clock agreement
- [`icc()`](https://bhagesh-h.github.io/FALCONAge/r/reference/icc.md) :
  Intraclass correlation, ICC(2,1)
- [`run_benchmark()`](https://bhagesh-h.github.io/FALCONAge/r/reference/run_benchmark.md)
  [`print(`*`<falcon_benchmark>`*`)`](https://bhagesh-h.github.io/FALCONAge/r/reference/run_benchmark.md)
  : The AA1 and AA2 benchmark

## Clinical references

KDM and homeostatic dysregulation have no fixed coefficients: both are
defined relative to a reference cohort, and the manifest records which.

- [`fit_kdm()`](https://bhagesh-h.github.io/FALCONAge/r/reference/fit_kdm.md)
  : Fit a Klemera-Doubal reference
- [`fit_hd()`](https://bhagesh-h.github.io/FALCONAge/r/reference/fit_hd.md)
  : Fit a homeostatic dysregulation reference

## Figures and reports

Every plot returns its data as well as its figure, which is how the two
languages draw the same numbers with different engines. `clock_atlas` is
the exception in scale rather than in kind: one figure covering every
clock across every pooled study, for the case where a per-clock panel
would need forty of them.

- [`plot_ba_vs_ca()`](https://bhagesh-h.github.io/FALCONAge/r/reference/plot_ba_vs_ca.md)
  : Predicted against chronological age
- [`plot_bland_altman()`](https://bhagesh-h.github.io/FALCONAge/r/reference/plot_bland_altman.md)
  : Bland-Altman agreement across the age range
- [`plot_calibration()`](https://bhagesh-h.github.io/FALCONAge/r/reference/plot_calibration.md)
  : Residual against chronological age
- [`plot_acceleration()`](https://bhagesh-h.github.io/FALCONAge/r/reference/plot_acceleration.md)
  : Distribution of age acceleration
- [`plot_acceleration_by_group()`](https://bhagesh-h.github.io/FALCONAge/r/reference/plot_acceleration_by_group.md)
  : Age acceleration by group
- [`plot_acceleration_heatmap()`](https://bhagesh-h.github.io/FALCONAge/r/reference/plot_acceleration_heatmap.md)
  : Age acceleration across clocks and samples
- [`plot_forest()`](https://bhagesh-h.github.io/FALCONAge/r/reference/plot_forest.md)
  : Forest plot of benchmark effect sizes
- [`plot_agreement()`](https://bhagesh-h.github.io/FALCONAge/r/reference/plot_agreement.md)
  : Agreement between clocks
- [`plot_clock_radar()`](https://bhagesh-h.github.io/FALCONAge/r/reference/plot_clock_radar.md)
  : Multi-clock radar profile
- [`plot_clock_chord()`](https://bhagesh-h.github.io/FALCONAge/r/reference/plot_clock_chord.md)
  : Circos chord diagram of CpG sharing between clocks
- [`plot_clock_pca()`](https://bhagesh-h.github.io/FALCONAge/r/reference/plot_clock_pca.md)
  : Samples embedded in clock space
- [`plot_clock_atlas()`](https://bhagesh-h.github.io/FALCONAge/r/reference/plot_clock_atlas.md)
  : Clock atlas: every algorithm across every pooled study
- [`plot_coverage()`](https://bhagesh-h.github.io/FALCONAge/r/reference/plot_coverage.md)
  : Per-clock feature coverage
- [`plot_by_platform()`](https://bhagesh-h.github.io/FALCONAge/r/reference/plot_by_platform.md)
  [`plot_by_study()`](https://bhagesh-h.github.io/FALCONAge/r/reference/plot_by_platform.md)
  : Score distribution by platform or by study
- [`plot_kaplan_meier()`](https://bhagesh-h.github.io/FALCONAge/r/reference/plot_kaplan_meier.md)
  : Survival by age acceleration
- [`plot_volcano()`](https://bhagesh-h.github.io/FALCONAge/r/reference/plot_volcano.md)
  : Volcano plot of association results
- [`plot_benchmark()`](https://bhagesh-h.github.io/FALCONAge/r/reference/plot_benchmark.md)
  : AA2 and AA1 counts per clock
- [`plot_benchmark_error_bias()`](https://bhagesh-h.github.io/FALCONAge/r/reference/plot_benchmark_error_bias.md)
  : Error against bias on healthy controls
- [`plot_benchmark_heatmap()`](https://bhagesh-h.github.io/FALCONAge/r/reference/plot_benchmark_heatmap.md)
  : Effect size per clock and dataset
- [`falcon_palette()`](https://bhagesh-h.github.io/FALCONAge/r/reference/falcon_palette.md)
  : The shared categorical palette
- [`falcon_scheme()`](https://bhagesh-h.github.io/FALCONAge/r/reference/falcon_scheme.md)
  : The shared visual specification
- [`falcon_theme()`](https://bhagesh-h.github.io/FALCONAge/r/reference/falcon_theme.md)
  : The shared ggplot2 theme
- [`report()`](https://bhagesh-h.github.io/FALCONAge/r/reference/report.md)
  : Write a self-contained HTML report

## Downloading public data

Accession in, files and a normalised sample table out. Credentialed
archives are documented rather than automated – the access agreement is
the user’s, not the tool’s.

- [`download()`](https://bhagesh-h.github.io/FALCONAge/r/reference/download.md)
  : Download public data by accession
- [`cache_info()`](https://bhagesh-h.github.io/FALCONAge/r/reference/cache_info.md)
  [`clear_cache()`](https://bhagesh-h.github.io/FALCONAge/r/reference/cache_info.md)
  : Inspect or clear the download cache

## Configuration and setup

What resolved, on this machine, in this environment.

- [`falconage_install()`](https://bhagesh-h.github.io/FALCONAge/r/reference/falconage_install.md)
  : Install the Python core FALCONAge computes with
- [`falconage_config()`](https://bhagesh-h.github.io/FALCONAge/r/reference/falconage_config.md)
  [`print(`*`<falcon_config>`*`)`](https://bhagesh-h.github.io/FALCONAge/r/reference/falconage_config.md)
  : What this installation resolved to
- [`falconage_available()`](https://bhagesh-h.github.io/FALCONAge/r/reference/falconage_available.md)
  : Is the Python core importable?
