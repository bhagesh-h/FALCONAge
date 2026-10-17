# How many samples to see an effect

The first thing a laboratory needs, and it is needed before any array is
run. Two independent groups, two-sided.

## Usage

``` r
power_n(
  clock,
  effect,
  sd = NULL,
  result = NULL,
  icc = NULL,
  alpha = 0.05,
  power = 0.8,
  replicates = 1L
)
```

## Arguments

- clock:

  Clock name.

- effect:

  The difference worth detecting, in the clock's own unit. No default: a
  power calculation with an assumed effect size is a way of writing down
  an assumption without noticing.

- sd:

  Population SD of the score. Measured from `result` when given.

- result:

  A scored pilot. Supplies `sd`, and – if
  [`technical_se()`](https://bhagesh-h.github.io/FALCONAge/r/reference/technical_se.md)
  has been called on it – a measured ICC for this laboratory.

- icc:

  Override the reliability figure.

- alpha, power:

  Significance and target power.

- replicates:

  Assay each sample this many times and average.

## Value

A list with `n_per_group`, `n_total`, the reliability used and where it
came from, and the n a perfectly repeatable assay would need.

## Details

Reliability is part of the answer: the SD a user measures already
contains the assay's noise, so splitting it out with the clock's
test-retest ICC says how much of the sample size is buying signal and
how much is averaging out the instrument. That is the arithmetic behind
the finding that the original clocks need 3-16 replicates per condition
where their principal-component versions need 1-2.

## Examples

``` r
if (FALSE) { # \dontrun{
power_n("horvath2013", effect = 1, sd = 5, icc = 0.9)
} # }
```
