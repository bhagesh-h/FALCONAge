# Clock atlas: every algorithm across every pooled study

The one figure that answers "of these twenty-odd algorithms, which are
measuring aging at all". Rows are clocks on a shared vertical axis;
panels are, left to right: type badge, MedAE, signed MedE, per-study
detection dots, AA2/AA1 counts, and mean coverage.

## Usage

``` r
plot_clock_atlas(
  x,
  bench,
  dataset_col = "dataset",
  min_datasets = 2L,
  coverage_floor = 0.8,
  max_clocks = 40L
)
```

## Arguments

- x:

  A combined `falcon_result`, normally from
  [`combine()`](https://bhagesh-h.github.io/FALCONAge/r/reference/combine.md).

- bench:

  The matching output of
  [`run_benchmark()`](https://bhagesh-h.github.io/FALCONAge/r/reference/run_benchmark.md).

- dataset_col:

  Column naming the study.

- min_datasets:

  Refuse below this many studies. Two is the floor at which "consistent
  across cohorts" means anything.

- coverage_floor:

  Drawn on panel F; use the value the run was scored with.

- max_clocks:

  Keep the highest-scoring this many, so a full catalogue still fits a
  page.

## Value

A patchwork of ggplots when patchwork is installed, otherwise a named
list of the six panels, so the figure remains usable without it.

## How to read it

Panel D, down the page. A row of hollow circles on zero is an algorithm
that detected nothing in any cohort; filled dots pushed right are the
ones that did, and several in a row means the effect held across studies
rather than in one. Panels B and C are diagnostics for reading D, not
scores – a clock that merely returned chronological age would be perfect
in B, empty in D, and useless for every purpose anyone scores a clock
for. Panel F separates the two ways of finding nothing: a clock that saw
its features and found no difference, from one that never had the
features to look with.

## Optional by design

It needs several studies to say anything, so it is not part of the
default figure set and errors below `min_datasets`.

## Examples

``` r
if (FALSE) { # \dontrun{
res <- combine(list(r1, r2, r3))
b   <- run_benchmark(res, dataset_col = "dataset")
plot_clock_atlas(res, b)
} # }
```
