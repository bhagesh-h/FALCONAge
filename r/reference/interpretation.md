# How to read each score in a result

One row per clock giving the scale, the unit, which operations that
scale permits, both coverage measures, the published reliability where
one is established, and any documented disagreement between the clock's
paper and the coefficients that circulate for it.

## Usage

``` r
interpretation(x)
```

## Arguments

- x:

  A `falcon_result`.

## Value

A data frame, one row per clock.

## Details

The point is that this travels with the numbers. A scale type on a
documentation page warns nobody reading a table in an R session.

## Examples

``` r
if (FALSE) { # \dontrun{
interpretation(res)
} # }
```
