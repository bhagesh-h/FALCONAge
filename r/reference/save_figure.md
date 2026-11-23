# Write a figure as `<title>.png` with its legend text in `<title>.txt`

The PNG is named after the figure's title, lower case with underscores,
with `suffix` (usually a clock id) appended for per-clock figures. The
`.txt` beside it holds the title, the figure's own numbers and the
description, ready for a figure legend. The same names as the Python
`falconage.plot.save_figure`.

## Usage

``` r
save_figure(p, outdir, suffix = NULL, width = NULL, height = NULL, dpi = NULL)
```

## Arguments

- p:

  A plot returned by one of the `plot_*` functions.

- outdir:

  Directory to write into; created if absent.

- suffix:

  Optional text appended to the file name, such as a clock id.

- width, height:

  Size in inches; default the plot's own size where it sets one (the
  coverage map does), otherwise from `colorscheme.yaml`.

- dpi:

  Resolution; default from `colorscheme.yaml`.

## Value

The path of the PNG, invisibly.

## Examples

``` r
if (FALSE) { # \dontrun{
save_figure(plot_ba_vs_ca(res, "horvath2013"), "figures", suffix = "horvath2013")
} # }
```
