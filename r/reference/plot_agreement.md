# Agreement between clocks

Rank correlation, not Pearson: clocks on different scales have no
meaningful linear correlation. Blocks of agreement are usually shared
training cohorts rather than shared biology, which is what
[`plot_clock_chord()`](https://bhagesh-h.github.io/FALCONAge/r/reference/plot_clock_chord.md)
can confirm.

## Usage

``` r
plot_agreement(x, method = c("spearman", "pearson"), cluster = TRUE)
```

## Arguments

- x:

  A `falcon_result`.

- method:

  `"spearman"` or `"pearson"`.

- cluster:

  Order rows and columns by hierarchical clustering.

## Value

A ggplot.

## Examples

``` r
if (FALSE) { # \dontrun{
plot_agreement(res)
} # }
```
