# Score a catalogue clock from a coefficient file you supply

Replaces a catalogue clock's bundled coefficients with your own copy,
for this session: a corrected or re-fitted set, or a copy you want to
check. The file is validated against the clock (two columns, unique
feature ids, finite values, the declared feature count) and a mismatch
is refused with the discrepancy named. The digest goes into the run
manifest as `user_supplied`. To score a clock that is not in the
catalogue, use
[`add_clock()`](https://bhagesh-h.github.io/FALCONAge/r/reference/add_clock.md).

## Usage

``` r
register_local_weights(clock_id, path, sha256 = NULL)
```

## Arguments

- clock_id:

  A clock identifier, e.g. `"hannum"`.

- path:

  A CSV with `feature_id,coefficient` columns.

- sha256:

  Optional expected digest; a mismatch is an error.

## Value

The file's SHA-256, invisibly.

## Examples

``` r
if (FALSE) { # \dontrun{
register_local_weights("hannum", "~/my_copy/hannum.csv")
score(d, clocks = "hannum")
} # }
```
