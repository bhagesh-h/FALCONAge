# Horvath's gold-standard normalisation

Calibrates each sample to the `goldstandard2` profile of Horvath (2013),
Additional file 22, by his `BMIQcalibration` (Additional file 24),
ported so that it reproduces the R code to about 1e-9 in beta. Opt-in:
use it before a clock whose authors normalised this way, such as
`horvath2013` and `knight`, whose published code runs it.

## Usage

``` r
horvath_normalise(
  data,
  absent = c("drop", "fill"),
  sample_kind = c("Rejection", "Rounding")
)
```

## Arguments

- data:

  A
  [falcon_data](https://bhagesh-h.github.io/FALCONAge/r/reference/falcon_data.md).

- absent:

  `"drop"` calibrates on the gold-standard probes the data has, as
  Knight et al.'s published code does; `"fill"` sets absent ones to
  their gold-standard value.

- sample_kind:

  R's `sample.kind` for the 20,000-probe draw the fits use:
  `"Rejection"` (R 3.6.0 and later) or `"Rounding"` (earlier R, which
  reproduces output published from it).

## Value

A
[falcon_data](https://bhagesh-h.github.io/FALCONAge/r/reference/falcon_data.md).

## Details

Only the 21,368 gold-standard probes are calibrated. A missing value is
set to its gold-standard value first; the counts are in the result's
`uns`.

## Examples

``` r
if (FALSE) { # \dontrun{
s <- score(horvath_normalise(d), clocks = "knight")
} # }
```
