# Build a dataset from an R matrix or data frame

Build a dataset from an R matrix or data frame

## Usage

``` r
falcon_data(
  x,
  obs = NULL,
  modality = "dna_methylation",
  platform = NULL,
  species = "Homo sapiens",
  units = NULL
)

# S3 method for class 'falcon_data'
print(x, ...)

# S3 method for class 'falcon_data'
summary(object, ...)
```

## Arguments

- x:

  A `falcon_data`.

- obs:

  Per-sample annotation with matching row names.

- modality:

  `"dna_methylation"`, `"clinical_chemistry"` or `"rrbs"`.

- platform:

  Optional, e.g. `"450K"`. Detected from the probe identifiers when
  omitted.

- species:

  Which organism the samples came from. Checked, not assumed: the
  mammalian array carries 96% of Horvath2013's CpGs, so a zebra scores
  at high coverage and returns a confident number from a clock fitted on
  people.

- units:

  For clinical chemistry, a named list of marker to unit.

- ...:

  Ignored.

- object:

  A `falcon_data`.

## Value

A falcon_data.

## Examples

``` r
if (FALSE) { # \dontrun{
d <- falcon_data(betas, obs = pheno, modality = "dna_methylation")
} # }
```
