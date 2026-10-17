# Read RRBS coverage files

Read RRBS coverage files

## Usage

``` r
read_rrbs_dir(paths, min_coverage = 5L)
```

## Arguments

- paths:

  Character vector of per-sample site files.

- min_coverage:

  Minimum read depth for a site to be kept. A ratio from four reads and
  one from four hundred are not the same measurement, and a clock handed
  both without distinction reports sequencing depth as biology.

## Value

A
[falcon_data](https://bhagesh-h.github.io/FALCONAge/r/reference/falcon_data.md)
with `modality = "rrbs"`.

## Examples

``` r
if (FALSE) { # \dontrun{
d <- read_rrbs_dir(list.files("mouse", full.names = TRUE), min_coverage = 5)
} # }
```
