# Write a fitted clock or stack as a clock FALCONAge can add

Writes the entry, the coefficients and the reference values.

## Usage

``` r
clock_to_file(fit, clock_id, outdir)
```

## Arguments

- fit:

  A `falcon_clockfit` from
  [`fit_clock`](https://bhagesh-h.github.io/FALCONAge/r/reference/fit_clock.md)
  or a `falcon_stack` from
  [`stack_clocks`](https://bhagesh-h.github.io/FALCONAge/r/reference/stack_clocks.md).

- clock_id:

  The id the clock will be scored under.

- outdir:

  Folder to write `<clock_id>.yaml` and its files into.

## Value

The path of the YAML entry, for
[`add_clock`](https://bhagesh-h.github.io/FALCONAge/r/reference/add_clock.md).
