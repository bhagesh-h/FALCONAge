# Scores with their measurement interval

Scores with their measurement interval

## Usage

``` r
interval(x, data = NULL, level = 0.95)
```

## Arguments

- x:

  A `falcon_result`.

- data:

  The `falcon_data` that was scored.

- level:

  Coverage, default 0.95.

## Value

A data frame with one row per sample per clock: value, se, lo, hi.
