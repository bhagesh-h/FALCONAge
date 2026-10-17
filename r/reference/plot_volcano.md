# Volcano plot of association results

Effect size against evidence, for the table
[`associate()`](https://bhagesh-h.github.io/FALCONAge/r/reference/associate.md)
returns.

## Usage

``` r
plot_volcano(assoc, effect = "beta", p = "p", fdr = 0.05, label_top = 10)
```

## Arguments

- assoc:

  Output of
  [`associate()`](https://bhagesh-h.github.io/FALCONAge/r/reference/associate.md).

- effect, p:

  Column names for the effect size and p-value.

- fdr:

  False-discovery rate for the significance threshold.

- label_top:

  How many of the strongest hits to label.

## Value

A ggplot.

## The threshold that is drawn

The dashed line is the Benjamini-Hochberg cut at `fdr`, taken from the
`q` column rather than recomputed. Drawing a raw p-value cut instead is
the common error: across many tests the two differ by orders of
magnitude, and the raw one calls noise significant.

## Examples

``` r
if (FALSE) { # \dontrun{
plot_volcano(associate(res, "mortality"))
} # }
```
