# Bland-Altman agreement across the age range

A correlation coefficient cannot show that a clock's error depends on
age. This can, and age-dependent error is the failure mode that makes a
single MedAE meaningless.

## Usage

``` r
plot_bland_altman(x, clock, age_col = "age")
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
plot_bland_altman(res, "horvath2013")
} # }
```
