# Between-clock agreement

Between-clock agreement

## Usage

``` r
agreement(x, method = c("spearman", "pearson"))
```

## Arguments

- x:

  A `falcon_result`.

- method:

  `"spearman"` (the default) or `"pearson"`. Spearman because two clocks
  on different scales – years and a log-hazard – have no meaningful
  Pearson correlation but a perfectly meaningful rank one, and mixing
  scales is the normal case rather than the exception.

## Value

A correlation matrix.

## Examples

``` r
if (FALSE) { # \dontrun{
round(agreement(res), 2)
} # }
```
