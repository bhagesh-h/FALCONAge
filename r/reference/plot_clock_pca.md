# Samples embedded in clock space

PC1 is almost always chronological age; that is expected and is not a
finding. Structure on PC2 that tracks a plate and not a phenotype is a
batch effect, and this is where it shows up.

## Usage

``` r
plot_clock_pca(x, colour_by = NULL)
```

## Arguments

- x:

  A `falcon_result`.

- colour_by:

  Column in the sample annotation.

## Value

A ggplot.

## Examples

``` r
if (FALSE) { # \dontrun{
plot_clock_pca(res, colour_by = "condition")
} # }
```
