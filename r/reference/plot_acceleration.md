# Distribution of age acceleration

Counted, not smoothed. A kernel density on a dozen cases invents
structure that is not in the data, and the figure is usually read before
anyone checks the sample size.

## Usage

``` r
plot_acceleration(acc, clock, obs = NULL, group = NULL)
```

## Arguments

- acc:

  Output of
  [`acceleration()`](https://bhagesh-h.github.io/FALCONAge/r/reference/acceleration.md).

- clock:

  A clock id.

- obs:

  Sample annotation, from
  [`obs()`](https://bhagesh-h.github.io/FALCONAge/r/reference/obs.md).

- group:

  Optional grouping column.

## Value

A ggplot.

## Examples

``` r
if (FALSE) { # \dontrun{
plot_acceleration(acceleration(res), "horvath2013", obs(res), "condition")
} # }
```
