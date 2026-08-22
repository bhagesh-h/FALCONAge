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

  `"A"`, `"B"` or `"C"`, or `NULL` for all. A ships with coefficients
  and runs offline; B is catalogued but has no traced primary source
  yet; C is a scaffold whose coefficients are research-use-only.

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
list_clocks(tier = "A")
list_clocks(search = "mortality")
list_clocks(tier = "C")   # the ones needing author permission
} # }
```
