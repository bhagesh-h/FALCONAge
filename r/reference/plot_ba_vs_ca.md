# Predicted against chronological age

With the identity line, not a regression line. A regression line always
looks like a good fit; the identity line is what exposes a clock running
five years high on everybody, which is what MedE measures and what
quietly wins AA1 in a benchmark that does not discount for it.

## Usage

``` r
plot_ba_vs_ca(x, clock, age_col = "age", group = NULL)
```

## Arguments

- x:

  A `falcon_result`.

- clock:

  A clock id.

- age_col:

  Chronological age column in the sample annotation.

- group:

  Optional grouping column.

## Value

A ggplot.

## Examples

``` r
if (FALSE) { # \dontrun{
plot_ba_vs_ca(res, "horvath2013", group = "condition")
} # }
```
