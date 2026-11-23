# Forest plot of benchmark effect sizes

The interval is what makes this honest. An eight-versus-eight comparison
with a twenty-year point estimate has an interval wide enough to say so,
and a bar chart of the point estimates alone would not.

## Usage

``` r
plot_forest(bench, top = NULL)
```

## Arguments

- bench:

  Output of
  [`run_benchmark()`](https://bhagesh-h.github.io/FALCONAge/r/reference/run_benchmark.md).

- top:

  Keep only the largest effects in each direction.

## Value

A ggplot.

## Examples

``` r
if (FALSE) { # \dontrun{
plot_forest(run_benchmark(res))
} # }
```
