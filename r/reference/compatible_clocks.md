# Which clocks this dataset can actually be scored on

Compatibility is coverage, not platform. A clock trained on 450K runs
perfectly well on EPIC data that carries its probes, and fails on 450K
data filtered down to 20,000 probes – only the feature list can answer
it.

## Usage

``` r
compatible_clocks(data, min_coverage = 0.8)
```

## Arguments

- data:

  A `falcon_data`.

- min_coverage:

  Fraction of a clock's features that must be present.

## Value

A character vector of clock ids.

## Examples

``` r
if (FALSE) { # \dontrun{
compatible_clocks(d)
} # }
```
