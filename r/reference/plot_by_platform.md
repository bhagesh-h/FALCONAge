# Score distribution by platform or by study

A shift between platforms in the same tissue is a technical effect, and
it is the reason cross-study comparisons need the platform as a
covariate. Between-study spread is usually larger than the within-study
effect being tested, which is why AA2 compares cases with their own
controls.

## Usage

``` r
plot_by_platform(x, clock, col = "platform")

plot_by_study(x, clock, col = "dataset")
```

## Arguments

- x:

  A `falcon_result`.

- clock:

  A clock id.

- col:

  Column in the sample annotation to split by.

## Value

A ggplot.

## Examples

``` r
if (FALSE) { # \dontrun{
plot_by_platform(res, "horvath2013")
plot_by_study(res, "horvath2013")
} # }
```
