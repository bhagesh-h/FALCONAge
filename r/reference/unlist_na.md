# Flatten a Python list to an atomic vector, turning None into NA

[`unlist()`](https://rdrr.io/r/base/unlist.html) drops NULLs rather than
preserving position, which would silently shorten a column and misalign
every row after the first missing value.

## Usage

``` r
unlist_na(v)
```

## Arguments

- v:

  A list from `py_to_r()`.

## Value

An atomic vector of the same length.
