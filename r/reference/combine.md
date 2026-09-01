# Combine per-dataset results for a benchmark across studies

Datasets are scored separately and combined afterwards, never merged
before scoring. A 27K study and an EPIC study have different probe
spaces, and feature coverage is a property of a dataset – merged, a
clock that covers 99% of one study and 40% of another reports a single
meaningless average, and the AA2 test compares cases in the well-covered
study against controls in the badly-covered one.

## Usage

``` r
combine(results, keys = NULL)
```

## Arguments

- results:

  A list of `falcon_result` objects.

- keys:

  Optional dataset names, one per result.

## Value

A `falcon_result`.

## Examples

``` r
if (FALSE) { # \dontrun{
res <- combine(list(r1, r2, r3), keys = c("GSE1", "GSE2", "GSE3"))
} # }
```
