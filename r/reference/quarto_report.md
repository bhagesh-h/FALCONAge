# Write the step-ordered Quarto report for an output directory

Every file in `outdir` is placed under the step of the analysis that
produced it, in step order, with its description; a file the report does
not recognise is listed in the last step, so nothing is left out.
Rendering embeds every figure and table, so the HTML is one file that
references nothing. Without Quarto on the path, the `.qmd` is written
and the error names the command that renders it.

## Usage

``` r
quarto_report(outdir, x = NULL, title = "FALCONAge report", render = TRUE)
```

## Arguments

- outdir:

  A directory written by
  [`run_report()`](https://bhagesh-h.github.io/FALCONAge/r/reference/run_report.md)
  or `falconage report`.

- x:

  Optional `falcon_result`, to add what each scored category of clock
  means to the scores step.

- title:

  Page title.

- render:

  Render to HTML as well as writing the source.

## Value

The path of the `.html` (or the `.qmd` without rendering), invisibly.

## Examples

``` r
if (FALSE) { # \dontrun{
quarto_report("results")
} # }
```
