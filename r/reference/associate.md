# Associate clock scores with an outcome

Ordinary least squares of each clock on an outcome, adjusted for
covariates, with Benjamini-Hochberg correction across clocks.

## Usage

``` r
associate(x, outcome, covariates = c("age", "sex"), clocks = NULL)
```

## Arguments

- x:

  A `falcon_result`.

- outcome:

  Column in the sample annotation.

- covariates:

  Character vector of adjustment columns.

- clocks:

  Optional clock names.

## Value

A data frame with `beta`, `se`, `t`, `p` and `q`, sorted by p.

## Examples

``` r
if (FALSE) { # \dontrun{
associate(res, outcome = "bmi", covariates = c("age", "sex"))
} # }
```
