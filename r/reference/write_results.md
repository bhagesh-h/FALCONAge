# Write the standard results layout

`scores.csv`, `scores_wide.csv`, `qc.csv` and `run_manifest.json`.

## Usage

``` r
write_results(x, outdir)
```

## Arguments

- x:

  A `falcon_result`.

- outdir:

  Destination directory, created if absent.

## Value

A named character vector of what was written, invisibly.

## Examples

``` r
if (FALSE) { # \dontrun{
write_results(res, "results/")
} # }
```
