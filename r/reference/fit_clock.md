# Fit a clock to one cohort, judged out of fold

Every reported prediction is made by a model that never saw the sample.
See
[`compare_learners`](https://bhagesh-h.github.io/FALCONAge/r/reference/compare_learners.md)
to choose the learner first.

## Usage

``` r
fit_clock(
  data,
  target = "age",
  learner = "elasticnet",
  groups = NULL,
  transform = NULL,
  outer_folds = 10L,
  inner_folds = 10L
)
```

## Arguments

- data:

  A `falcon_data` whose `obs` holds `target`.

- target:

  The `obs` column to predict.

- learner:

  `"elasticnet"` (default), `"lasso"`, `"ridge"`,
  `"screened_elasticnet"`, `"pc_elasticnet"`, or a scikit-learn learner
  (`"random_forest"`, `"gradient_boosting"`, `"svr"`, `"mlp"`).

- groups:

  An `obs` column whose members always share a fold, such as the person
  when there are repeated visits.

- transform:

  `NULL`, or `"horvath"` for Horvath's log-linear age transform.

- outer_folds, inner_folds:

  Cross-validation folds.

## Value

A `falcon_clockfit`: `metrics`, `coefficients`, `predictions`, and the
Python object in `py`.
