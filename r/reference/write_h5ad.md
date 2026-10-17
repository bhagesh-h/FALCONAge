# Write a dataset to the interchange format both languages read

Write a dataset to the interchange format both languages read

## Usage

``` r
write_h5ad(data, path)

read_h5ad(path)
```

## Arguments

- data:

  A
  [falcon_data](https://bhagesh-h.github.io/FALCONAge/r/reference/falcon_data.md).

- path:

  Destination `.h5ad`.

## Value

The path, invisibly.

## Examples

``` r
if (FALSE) { # \dontrun{
write_h5ad(d, "prepared.h5ad")
} # }
```
