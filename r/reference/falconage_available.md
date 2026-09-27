# Is the Python core importable?

Is the Python core importable?

## Usage

``` r
falconage_available()
```

## Value

`TRUE` when `falconage` can be imported in the configured interpreter,
`FALSE` otherwise. Never raises, so it is safe in a conditional or a
`skip_if_not()`.

## Examples

``` r
falconage_available()
#> [1] TRUE
```
