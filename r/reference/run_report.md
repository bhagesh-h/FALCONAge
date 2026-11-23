# Run the whole analysis and write every deliverable to a directory

Quality control, scoring, technical standard error, conformal intervals,
age acceleration, a consensus test when `group` names a column,
interpretation and evidence tables, figures and the one-page report: the
directory `falconage report` writes. With `quarto = TRUE` it also writes
`falconage_report.qmd`, with every file placed under the step of the
analysis that produced it, and renders it to one self-contained
`falconage_report.html` when Quarto is on the path.

## Usage

``` r
run_report(
  data,
  outdir,
  clocks = "compatible",
  group = NULL,
  level = 0.9,
  min_coverage = 0.8,
  figures = TRUE,
  quarto = FALSE,
  render = TRUE
)
```

## Arguments

- data:

  A `falcon_data`, already prepared.

- outdir:

  Output directory.

- clocks:

  `"compatible"`, `"all"`, or a character vector of clock ids.

- group:

  Optional column of the sample annotation to compare groups by.

- level:

  Coverage of the conformal intervals.

- min_coverage:

  Fraction of a clock's features that must be present.

- figures:

  Draw the figures.

- quarto:

  Also write the step-ordered Quarto report.

- render:

  With `quarto = TRUE`, render it when Quarto is available.

## Value

The paths of the main files, as a named character vector, invisibly.

## Examples

``` r
if (FALSE) { # \dontrun{
run_report(d, "results", group = "condition", quarto = TRUE)
} # }
```
