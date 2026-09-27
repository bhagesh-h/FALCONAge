# Intraclass correlation, ICC(2,1)

Two-way random effects, absolute agreement, single measure – the variant
that answers "would a repeat measurement of this person give the same
number". ICC(3,1) assumes the raters are the only ones of interest and
reports a higher number for the same data; papers rarely say which they
used.

## Usage

``` r
icc(values, subject_col, value_col)
```

## Arguments

- values:

  A data frame with one row per measurement.

- subject_col, value_col:

  Column names.

## Value

A single numeric.

## Examples

``` r
if (FALSE) { # \dontrun{
icc(replicates, "subject", "horvath2013")
} # }
```
