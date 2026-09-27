# Convert a pandas Series to a named R vector

Same reasoning as
[`as_df()`](https://bhagesh-h.github.io/FALCONAge/r/reference/as_df.md):
reticulate's own Series conversion is version-dependent, and a summary
that comes back as an opaque environment on one machine and a vector on
another is not a summary.

## Usage

``` r
as_series(x)
```

## Arguments

- x:

  A pandas Series proxy.

## Value

A named vector.
