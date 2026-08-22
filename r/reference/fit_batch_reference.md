# Fit a frozen batch-correction reference

ComBat estimates its parameters from every sample it is given at once,
so adding a plate and re-running changes every previously corrected
value – measured at up to 2.20 years of shift in already-reported
epigenetic ages (PMC12495439). Freezing the global parameters and the
empirical-Bayes priors on a reference cohort removes that: every later
batch is standardised against the frozen values and gets only its own
effects estimated.

## Usage

``` r
fit_batch_reference(
  data,
  batch_col,
  covariates = character(0),
  protect = c("condition", "group")
)
```

## Arguments

- data:

  A `falcon_data` for the reference cohort.

- batch_col:

  Column naming the batch.

- covariates:

  Columns whose effect should be preserved rather than removed,
  typically age and sex.

- protect:

  Columns checked for being nested inside batch. A confounded design is
  refused, because correcting it removes the effect along with the
  artefact and returns a clean-looking null.

## Value

A `falcon_batch_reference`, with a digest for the run manifest.

## Details

The reference is an artefact you keep and version-control, like a
coefficient file. That is the whole design; without it this is ComBat
with extra steps.
