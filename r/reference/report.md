# Write a self-contained HTML report

One file: inlined CSS, base64 figures, the tables embedded. A report
that references `figures/ba_vs_ca.png` stops working the moment somebody
emails it, which is the only thing anybody does with a report.

## Usage

``` r
report(x, path, age_col = "age", group = NULL, title = "FALCONAge report")
```

## Arguments

- x:

  A `falcon_result`.

- path:

  Destination `.html`.

- age_col:

  Column in the sample annotation holding chronological age.

- group:

  Optional grouping column for the figures.

- title:

  Page title.

## Value

The path, invisibly.

## Examples

``` r
if (FALSE) { # \dontrun{
report(res, "report.html", group = "condition")
} # }
```
