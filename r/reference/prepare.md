# Standard methylation preprocessing

Harmonises probe identifiers, collapses EPIC v2 replicate suffixes,
clips to the beta range, and identifies the platform.

## Usage

``` r
prepare(data, aggregate_epicv2 = TRUE, clip = TRUE)
```

## Arguments

- data:

  A
  [falcon_data](https://bhagesh-h.github.io/FALCONAge/r/reference/falcon_data.md).

- aggregate_epicv2:

  Collapse EPIC v2 replicate probes.

- clip:

  Clip values into `[0, 1]`.

## Value

A
[falcon_data](https://bhagesh-h.github.io/FALCONAge/r/reference/falcon_data.md).

## Details

The EPIC v2 step is not optional in effect. Illumina renamed
`cg00000029` to `cg00000029_TC21`, and every clock in the registry
matches on the bare identifier – so without aggregation an EPIC v2
dataset overlaps almost nothing, the imputation step fills everything,
and the clock returns a confident number computed from imputed values
rather than an error.

## Examples

``` r
if (FALSE) { # \dontrun{
d <- prepare(read_series_matrix("GSE330325_series_matrix.txt.gz"))
} # }
```
