# The AA1 and AA2 benchmark

Tests a clock the way the field now expects: does it show higher age
acceleration in people with an aging-accelerating condition than in
their own controls? Median absolute error against chronological age does
not answer that – a perfect chronological oracle would score best on it
and be useless.

## Usage

``` r
run_benchmark(
  x,
  condition_col = "condition",
  control = "HC",
  dataset_col = NULL,
  age_col = "age",
  alpha = 0.05
)

# S3 method for class 'falcon_benchmark'
print(x, ...)
```

## Arguments

- x:

  A `falcon_benchmark`.

- condition_col, control:

  Column naming the condition, and the value that marks a control.

- dataset_col:

  Column naming the study, when several are combined.

- age_col:

  Chronological age column.

- alpha:

  FDR threshold.

- ...:

  Ignored.

## Value

A list with `summary` (per clock) and `per_dataset` (per comparison).

## Details

**AA2**, for a dataset with controls: is the condition group's
acceleration higher than its controls'? One-sided Mann-Whitney, BH
corrected.

**AA1**, for a dataset without controls: is it above zero? One-sided
Wilcoxon signed-rank.

**MedE**, the median signed error on healthy controls, discounts the AA1
credit: `total = AA2 + AA1 * (1 - max(0, MedE) / MedAE)`. Without it a
clock that simply over-predicts everybody sweeps AA1, because every
group looks accelerated when the baseline is wrong.

## Examples

``` r
if (FALSE) { # \dontrun{
b <- run_benchmark(res, condition_col = "condition", control = "HC",
                   dataset_col = "dataset")
b$summary
} # }
```
