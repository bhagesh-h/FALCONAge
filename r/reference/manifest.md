# The run manifest

Versions, device, dtype, the SHA-256 of every coefficient file used,
per-clock coverage, and every warning raised. Two runs reporting the
same score either used the same coefficients or the manifest says they
did not.

## Usage

``` r
manifest(x)

# S3 method for class 'falcon_result'
manifest(x)
```

## Arguments

- x:

  A `falcon_result`.

## Value

A named list.

## Examples

``` r
if (FALSE) { # \dontrun{
m <- manifest(res)
m$weights$horvath2013$sha256
} # }
```
