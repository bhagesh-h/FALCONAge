# The title, subtitle and description of a FALCONAge figure

Figures are drawn without a title or caption on the canvas. The text
lives in the plot object; this returns it, and
[`save_figure()`](https://bhagesh-h.github.io/FALCONAge/r/reference/save_figure.md)
writes it to a `.txt` beside the PNG.

## Usage

``` r
figure_text(p)
```

## Arguments

- p:

  A plot returned by one of the `plot_*` functions.

## Value

A list with `title`, `subtitle` and `description`.

## Examples

``` r
if (FALSE) { # \dontrun{
figure_text(plot_ba_vs_ca(res, "horvath2013"))$title
} # }
```
