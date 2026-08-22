# Fit a Klemera-Doubal reference

KDM has no fixed coefficients. Each biomarker is regressed on
chronological age in a reference cohort and the panel is inverted to a
maximum-likelihood age, so the same person scored against NHANES III and
against a hospital cohort gets two different numbers, both correct. The
manifest records which reference was used.

## Usage

``` r
fit_kdm(reference, markers, age_col = "age")
```

## Arguments

- reference:

  A data frame with the markers and an age column.

- markers:

  Character vector of column names.

- age_col:

  Age column name.

## Value

A reference object to pass to
[`score()`](https://bhagesh-h.github.io/FALCONAge/r/reference/score.md).

## Examples

``` r
if (FALSE) { # \dontrun{
ref <- fit_kdm(nhanes3, markers = c("albumin", "creatinine", "glucose"))
score(d, clocks = "kdm", reference = ref)
} # }
```
