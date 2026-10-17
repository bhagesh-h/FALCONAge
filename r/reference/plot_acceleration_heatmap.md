# Age acceleration across clocks and samples

Z-scored per clock, because the clocks are on wildly different scales;
without it one clock's variance dominates the colour map.

## Usage

``` r
plot_acceleration_heatmap(acc)
```

## Arguments

- acc:

  Output of
  [`acceleration()`](https://bhagesh-h.github.io/FALCONAge/r/reference/acceleration.md).

## Value

A ggplot.

## Examples

``` r
if (FALSE) { # \dontrun{
plot_acceleration_heatmap(acceleration(res))
} # }
```
