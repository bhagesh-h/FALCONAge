# Age acceleration

Age acceleration

## Usage

``` r
acceleration(
  x,
  method = c("residual", "absolute", "within_group"),
  age_col = "age",
  group = NULL,
  clocks = NULL,
  adjust = NULL
)
```

## Arguments

- x:

  A `falcon_result`.

- method:

  `"absolute"` – predicted minus chronological. Interpretable in years,
  and confounded by the clock's own bias: a clock that over-predicts
  everyone by three years gives everyone three years of acceleration.

  `"residual"` – the residual from regressing predicted on chronological
  age. Centred at zero by construction, which removes that bias and also
  removes any real cohort-wide effect. The field's default.

  `"within_group"` – residual from a regression fitted separately within
  each level of `group`. What the AA2 benchmark needs: it asks whether
  cases accelerate relative to *their own* controls.

- age_col:

  Column in the sample annotation holding chronological age.

- group:

  Grouping column, required for `"within_group"`.

- clocks:

  Optional clock names. Naming them means every one must be valid, and a
  pace or log-hazard clock raises. Leaving it `NULL` means "the ones
  this makes sense for" and quietly excludes the others.

- adjust:

  Extra covariates to regress out alongside chronological age.

  `"cell_composition"` uses the deconvolution clocks scored in the same
  run. An acceleration adjusted this way answers "is this blood ageing
  faster", where the unadjusted version answers "is it ageing faster
  *or* is its cell mix different" and reports both as one number.

  A character vector instead names columns of the sample annotation, for
  measured counts or anything else. Only available with
  `method = "residual"`.

## Value

A data frame of samples by clocks.

## Which convention a paper used

Often not stated, and the three disagree by several years on the same
data. The convention is recorded in the returned frame's `method`
attribute.

## Examples

``` r
if (FALSE) { # \dontrun{
acceleration(res)
acceleration(res, method = "within_group", group = "dataset")
} # }
```
