# Age acceleration by group

The case/control workhorse: box for the median and interquartile range,
points for every sample, because a box on twelve samples hides how few.

## Usage

``` r
plot_acceleration_by_group(acc, clock, obs, group)
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

  Grouping column.

## Value

A ggplot.

## Examples

``` r
if (FALSE) { # \dontrun{
plot_acceleration_by_group(acceleration(res), "horvath2013", obs(res), "condition")
} # }
```
