# Fit a homeostatic dysregulation reference

The reference should be the healthy young subset, not the whole cohort.
Fitting the centre on everybody makes the average unhealthy person the
definition of normal, which is a different measurement with the same
units.

## Usage

``` r
fit_hd(reference, markers)
```

## Arguments

- reference:

  A data frame of the healthy reference population.

- markers:

  Character vector of column names.

## Value

A reference object to pass to
[`score()`](https://bhagesh-h.github.io/FALCONAge/r/reference/score.md).

## Examples

``` r
if (FALSE) { # \dontrun{
ref <- fit_hd(nhanes3_hdtrain, markers = markers)
score(d, clocks = "hd", reference = ref)
} # }
```
