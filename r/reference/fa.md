# The Python module handle

Imported once and cached. Delayed rather than loaded in `.onLoad` so
that [`library(FALCONAge)`](https://github.com/bhagesh-h/FALCONAge)
works on a machine with no Python at all – which is the state every
machine is in before
[`falconage_install()`](https://bhagesh-h.github.io/FALCONAge/r/reference/falconage_install.md)
runs.

## Usage

``` r
fa()
```

## Value

The `falconage` Python module.
