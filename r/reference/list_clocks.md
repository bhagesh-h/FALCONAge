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

  `"bundled"`, `"untraced"` or `"licensed"`, or `NULL` for all. The
  retired `"A"`, `"B"`, `"C"` are still accepted. Bundled ships with
  coefficients and runs offline; untraced is catalogued but has no
  traced primary source yet; licensed is implemented but its
  coefficients are research-use-only.

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
list_clocks(tier = "licensed")   # the ones needing author permission
} # }
```
