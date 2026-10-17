# What each clock has lost on this dataset, before scoring

One row per clock: how many of its features are present, and – where the
coefficients are available – how much of the model's total weight those
present features carry, plus the heaviest probes that are missing.

## Usage

``` r
probe_loss(x, clocks = "all", top = 3L)
```

## Arguments

- x:

  A `falcon_data`.

- clocks:

  `"all"`, `"scoreable"` for the ones whose coefficients are available,
  or a character vector of clock names.

- top:

  How many of the heaviest absent features to name per clock.

## Value

A data frame, worst mass coverage first. `mass_coverage` is `NA` for a
clock whose coefficients are not available – the weights are what the
column is computed from.

## Why both numbers

A count treats every probe as interchangeable, and an elastic net's
weights are nothing like uniform. "92% of probes present" covers both
"the missing 8% are negligible" and "the missing 8% carry a third of the
model". EPIC v2 dropped probes that several first-generation clocks lean
on, which is why those clocks shift on v2 arrays while the
principal-component versions barely move – the same probe loss, very
different consequences.

Run this on an array you have not used before, before
[`score()`](https://bhagesh-h.github.io/FALCONAge/r/reference/score.md).
It answers "will this dataset support these clocks" without producing a
number anyone can quote.

## Examples

``` r
if (FALSE) { # \dontrun{
probe_loss(d, clocks = "scoreable")
} # }
```
