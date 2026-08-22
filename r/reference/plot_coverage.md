# Per-clock feature coverage

The plot to read before the scores. Below the dashed line a clock is
mostly scoring imputed values, which is why it was refused.

## Usage

``` r
plot_coverage(x, floor = 0.8)
```

## Arguments

- x:

  A `falcon_result`.

- floor:

  Coverage threshold, matching the one used at score time.

## Value

A ggplot.

## Examples

``` r
if (FALSE) { # \dontrun{
plot_coverage(res)
} # }
```
