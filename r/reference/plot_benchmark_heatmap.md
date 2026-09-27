# Effect size per clock and dataset

Read the columns. A dataset blank across every clock is either a
condition the clocks cannot see or a cohort too small to show it.

## Usage

``` r
plot_benchmark_heatmap(bench)
```

## Arguments

- bench:

  Output of
  [`run_benchmark()`](https://bhagesh-h.github.io/FALCONAge/r/reference/run_benchmark.md).

## Value

A ggplot.

## Examples

``` r
if (FALSE) { # \dontrun{
plot_benchmark_heatmap(run_benchmark(res))
} # }
```
