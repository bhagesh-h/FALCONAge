# Read a GEO series matrix

Metadata and beta values in one gzipped file, which is what roughly 60%
of GEO methylation series publish and nothing else. The characteristics
block becomes the sample annotation, with GEO's own key names kept
verbatim – an `age` column that silently turned out to be
`age at diagnosis` is worse than no column.

## Usage

``` r
read_series_matrix(path)
```

## Arguments

- path:

  Path to a `*_series_matrix.txt.gz`.

## Value

A
[falcon_data](https://bhagesh-h.github.io/FALCONAge/r/reference/falcon_data.md).

## Details

Some series carry metadata only and put the values in a supplementary
file. That is read too, and flagged, rather than treated as an error.

## Examples

``` r
if (FALSE) { # \dontrun{
d <- read_series_matrix("GSE66459_series_matrix.txt.gz")
head(obs(d))
} # }
```
