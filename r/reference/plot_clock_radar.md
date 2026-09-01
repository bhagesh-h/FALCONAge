# Multi-clock radar profile

Z-scored within clock against the cohort, because the axes are otherwise
in incompatible units and the polygon would be a picture of the scales
rather than of the samples.

## Usage

``` r
plot_clock_radar(x, group = NULL, max_clocks = 12L)
```

## Arguments

- x:

  A `falcon_result`.

- group:

  Optional grouping column; one polygon per level.

- max_clocks:

  Cap on the number of axes.

## Value

A ggplot in polar coordinates.

## Examples

``` r
if (FALSE) { # \dontrun{
plot_clock_radar(res, group = "condition")
} # }
```
