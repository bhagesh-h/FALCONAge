# List the clock catalogue

List the clock catalogue

## Usage

``` r
list_clocks(
  tier = NULL,
  data_type = NULL,
  generation = NULL,
  untraced = FALSE,
  search = NULL
)
```

## Arguments

- tier:

  `"bundled"` or `NULL` for all. Every catalogued clock is bundled: it
  ships its coefficients and runs offline.

- data_type:

  `"dna_methylation"` or `"clinical_chemistry"`.

- generation:

  `"first"`, `"second"`, `"pace"`, `"causal"`, `"mitotic"`, `"system"`
  or `"other"`.

- untraced:

  Only clocks with no established primary source.

- search:

  Substring match over id, name, what it predicts, and citation.

## Value

A data frame, one row per clock.

## Examples

``` r
if (FALSE) { # \dontrun{
list_clocks(tier = "bundled")
list_clocks(search = "mortality")
} # }
```
