# The pip specifier for the Python core

FALCONAge is not on PyPI, so `pip install falconage` finds nothing (or,
worse, finds some unrelated name). The core lives in the `python/`
subdirectory of the GitHub repository, which is what the
`#subdirectory=` fragment selects.

## Usage

``` r
pip_spec()
```

## Value

A single string suitable for `pip install`.

## Details

The ref is pinned to `v<Version>` from DESCRIPTION rather than left on
`main`, so an R package built from a tag installs the Python core from
the same tag. Those two halves have to agree: the R package asserts bit
equality against the core, and a mismatched pair fails that assertion in
a way that looks like a numerical bug rather than a version skew.
