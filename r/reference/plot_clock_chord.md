# Circos chord diagram of CpG sharing between clocks

The correlation heatmap says two clocks agree. This says how much of
that agreement is built in: chord width is the number of CpGs the two
clocks have literally in common. A pair with a thick chord and a high
correlation has told you much less than a pair with a high correlation
and no chord.

## Usage

``` r
plot_clock_chord(x, min_shared = 5L, max_clocks = 24L)
```

## Arguments

- x:

  A `falcon_result`.

- min_shared:

  Minimum shared CpGs for a chord to be drawn.

- max_clocks:

  Cap on ring size, for legibility.

## Value

A ggplot.

## Details

Follows the circos convention of the single-cell aging literature: an
outer ring of colour-coded entities, chords in the interior weighted by
the strength of the relationship.

## Examples

``` r
if (FALSE) { # \dontrun{
plot_clock_chord(res)
} # }
```
