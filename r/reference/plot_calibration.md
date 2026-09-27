# Residual against chronological age

The slope is the diagnostic. A negative one means the clock over-ages
the young and under-ages the old – regression to the mean, the most
common artefact in this field and the one most often reported as a
finding.

## Usage

``` r
plot_calibration(x, clock, age_col = "age")
```

## Arguments

- x:

  A `falcon_result`.

- clock:

  A clock id.

- age_col:

  Chronological age column in the sample annotation.

## Value

A ggplot.

## Examples

``` r
if (FALSE) { # \dontrun{
plot_calibration(res, "horvath2013")
} # }
```
