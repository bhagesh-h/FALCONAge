# Read a beta matrix

Read a beta matrix

## Usage

``` r
read_betas(path, samples_are = "auto", obs = NULL)
```

## Arguments

- path:

  CSV, TSV or parquet. Gzip is handled by extension.

- samples_are:

  `"auto"`, `"rows"` or `"columns"`. Auto decides from which axis
  carries probe-shaped identifiers, which is reliable because `cg` ids
  are unmistakable – guessing from the shape alone gets a 1000-sample
  450K matrix right and a 500,000-probe cohort wrong.

- obs:

  Optional per-sample annotation, row names matching the samples.

## Value

A
[falcon_data](https://bhagesh-h.github.io/FALCONAge/r/reference/falcon_data.md).

## Examples

``` r
if (FALSE) { # \dontrun{
d <- read_betas("betas.csv")
d <- read_betas("betas.parquet", obs = pheno)
} # }
```
