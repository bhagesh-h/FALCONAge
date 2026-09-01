# Quality control before scoring

Reports rather than fixes. A sample that is 40% missing may be a failed
array or may be a 27K matrix aligned against an EPIC feature space, and
the right response differs – so this says what it sees and leaves the
decision where it belongs.

## Usage

``` r
qc(data)
```

## Arguments

- data:

  A
  [falcon_data](https://bhagesh-h.github.io/FALCONAge/r/reference/falcon_data.md).

## Value

A list with `summary` (a named vector) and `warnings` (character).

## Examples

``` r
if (FALSE) { # \dontrun{
r <- qc(d)
r$summary
r$warnings
} # }
```
