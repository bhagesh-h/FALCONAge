# Compare learners for building a clock

Compares learners for building a clock, every number out of fold.

## Usage

``` r
compare_learners(
  data,
  target = "age",
  learners = c("elasticnet", "lasso", "ridge", "screened_elasticnet", "pc_elasticnet"),
  groups = NULL,
  transform = NULL,
  outer_folds = 5L,
  inner_folds = 3L
)
```

## Arguments

- data:

  A `falcon_data` whose `obs` holds `target`.

- target:

  The `obs` column to predict.

- learners:

  Character vector of learner names.

- groups:

  An `obs` column whose members always share a fold.

- transform:

  `NULL`, or `"horvath"`.

- outer_folds, inner_folds:

  Cross-validation folds.

## Value

A list: `table` (one row per learner), `selected` (the nested error of
picking the best learner, the number to report), `picks` and `best`.
