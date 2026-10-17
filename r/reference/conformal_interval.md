# Distribution-free prediction intervals against chronological age

A different question from
[`technical_se()`](https://bhagesh-h.github.io/FALCONAge/r/reference/technical_se.md),
with a larger answer. Technical error asks what a repeat of the same DNA
would do; this asks how far the number is likely to be from the truth.

## Usage

``` r
conformal_interval(x, level = 0.9, clocks = NULL)
```

## Arguments

- x:

  A `falcon_result`.

- level:

  Coverage. One of 0.80, 0.90, 0.95.

- clocks:

  Optional clock names; only age-scale clocks are calibrated.

## Value

A data frame with value, lo, hi, half_width, median_bias and mae.

## Details

Split conformal: the half-width is a quantile of the absolute residual
on a calibration set of healthy blood samples with known ages, so on any
sample exchangeable with that cohort the interval covers the truth at
the stated rate, with no distributional assumption and a finite-sample
guarantee.

## The limit, which is not small

Coverage holds for data *exchangeable with the calibration cohort* –
public blood data, adult, overwhelmingly of European ancestry. On a
paediatric or non-European cohort the guarantee does not transfer, which
is why every row carries `exchangeable = FALSE`: nothing here can verify
it, so nothing here implies it.
