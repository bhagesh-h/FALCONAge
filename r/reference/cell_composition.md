# Cell-type proportions estimated in the same run

Every clock in a result whose scale is `proportion` – the
reference-based deconvolution models – one column each, ready to pass to
[`acceleration()`](https://bhagesh-h.github.io/FALCONAge/r/reference/acceleration.md)
as `adjust`.

## Usage

``` r
cell_composition(x, min_clocks = 2L)
```

## Arguments

- x:

  A `falcon_result`.

- min_clocks:

  Minimum number of deconvolution clocks before a frame is returned
  rather than an empty one.

## Value

A data frame of samples by cell types, empty when the run had no
deconvolution clocks – absence of an adjustment is data, not an error.

## Why this matters

Blood composition changes with age and with everything else happening to
a person. A study of more than 10,000 blood samples found significant
associations between immune cell composition and epigenetic age
acceleration for every one of six widely used clocks (Aging Cell
2024;23:e14071), which means an unadjusted acceleration measures two
things and reports one number. The proportions needed to separate them
are usually already in the same result.

## Examples

``` r
if (FALSE) { # \dontrun{
cell_composition(res)
acceleration(res, adjust = "cell_composition")
} # }
```
