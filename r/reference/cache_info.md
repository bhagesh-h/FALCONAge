# Inspect or clear the download cache

The cache is content-addressed by URL, so two accessions that share a
file share the copy.

## Usage

``` r
cache_info()

clear_cache(confirm = FALSE)
```

## Arguments

- confirm:

  Required for `clear_cache()`. Without it the function reports how much
  would be deleted and stops – a cache holding a re-downloadable
  gigabyte is still an hour of somebody's time.

## Value

`cache_info()` a data frame; `clear_cache()` the bytes freed.

## Examples

``` r
if (FALSE) { # \dontrun{
cache_info()
clear_cache(confirm = TRUE)
} # }
```
