# Read a clinical chemistry table

Read a clinical chemistry table

## Usage

``` r
read_clinical(path, units = NULL)
```

## Arguments

- path:

  CSV, TSV or parquet.

- units:

  Named list mapping marker to unit, e.g.
  `list(albumin = "g/L", creatinine = "umol/L")`. Optional here so that
  a file can be read to look at it; the clinical clocks require it and
  raise with the exact list to supply if it is missing.

## Value

A
[falcon_data](https://bhagesh-h.github.io/FALCONAge/r/reference/falcon_data.md).

## Examples

``` r
if (FALSE) { # \dontrun{
d <- read_clinical("nhanes.csv",
                   units = list(albumin = "g/L", creatinine = "umol/L",
                                glucose = "mmol/L", crp = "mg/dL"))
} # }
```
