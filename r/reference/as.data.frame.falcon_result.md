# Scores as a data frame

Scores as a data frame

## Usage

``` r
# S3 method for class 'falcon_result'
as.data.frame(
  x,
  row.names = NULL,
  optional = FALSE,
  form = c("wide", "long"),
  ...
)
```

## Arguments

- x:

  A `falcon_result`.

- row.names:

  Ignored; sample ids are always the row names.

- optional:

  Ignored.

- form:

  `"wide"` (samples x clocks) or `"long"` (one row per sample per clock,
  carrying the scale and the provenance).

- ...:

  Ignored.

## Value

A data frame.

## Why long form carries the scale

It stops a reader averaging a mortality log-hazard with an age in years
because both were numbers in a column called `value`.

## Examples

``` r
if (FALSE) { # \dontrun{
as.data.frame(res)
as.data.frame(res, form = "long")
} # }
```
