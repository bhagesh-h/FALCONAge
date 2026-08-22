# Technical standard error on each score

How much of a score is the assay rather than the person. Every
implementation of every aging clock, including this one at first,
reported a point estimate for a quantity whose technical replicates
differ by up to nine years (Nat Aging 2022, s43587-022-00248-2).

## Usage

``` r
technical_se(x, data = NULL, source = c("auto", "probe", "clock"))
```

## Arguments

- x:

  A `falcon_result`.

- data:

  The `falcon_data` that was scored. Required for the per-probe path:
  the spread of each feature has to come from the matrix, not from the
  scores.

- source:

  `"auto"` (per-probe where possible, per-clock otherwise), `"probe"`,
  or `"clock"`.

## Value

A list with `se` (samples by clocks), `diagnostics` (per clock: how many
probes had a published ICC, how many fell back to the median, and the
ICC this cohort implies) and `refused` (clocks with no usable source).

## Details

For a linear clock the propagation is one line of algebra: with
per-probe measurement variance `s^2 (1 - ICC)`, the score's variance is
the weighted sum `sum_j w_j^2 s_j^2 (1 - ICC_j)`, carried through the
clock's output transform by the delta method.

## What this is not

Measurement error, not prediction error. A clock can be perfectly
repeatable and still be a poor estimate of anything; see
[`conformal_interval()`](https://bhagesh-h.github.io/FALCONAge/r/reference/conformal_interval.md)
for that question. It also says nothing about biological variability –
the same person sampled a fortnight later is a different measurement of
a different thing.

## An imputed feature widens the interval

Deliberately, and it is the part most likely to look wrong. An imputed
probe is the cohort mean, so it carries no information about the sample
in front of you and contributes its whole between-sample variance.
Treating it as well measured would make worse data produce a narrower
interval.

## Examples

``` r
if (FALSE) { # \dontrun{
u <- technical_se(res, d)
u$diagnostics
} # }
```
