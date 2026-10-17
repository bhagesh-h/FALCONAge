# The shared visual specification

Reads `colorscheme.yaml` – the single file that decides how every
FALCONAge figure looks and what it says, in both languages.

## Usage

``` r
falcon_scheme(path = NULL, reload = FALSE)
```

## Arguments

- path:

  Optional path to a colour scheme file.

- reload:

  Force a re-read, after editing the file in a live session.

## Value

A nested list: `palette`, `theme`, `plots`.

## Details

Searched for in three places, in order: the `path` argument, the
`FALCONAGE_COLORSCHEME` environment variable, then the copy that ships
inside the Python core. That order lets you restyle a whole report
without touching the installation.

## Examples

``` r
if (FALSE) { # \dontrun{
sch <- falcon_scheme()
sch$palette$categorical
sch$plots$ba_vs_ca$description
} # }
```
