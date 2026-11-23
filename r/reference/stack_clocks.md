# Combine several clocks into one score, judged out of fold

Learns weights for several clocks' scores out of fold and reports them
next to the best single clock and the plain mean.

## Usage

``` r
stack_clocks(
  x,
  target = "age",
  clocks = NULL,
  method = "nnls",
  event_col = NULL,
  groups = NULL
)
```

## Arguments

- x:

  A `falcon_result`.

- target:

  The `obs` column to predict (follow-up time for `"cox"`).

- clocks:

  Which scored clocks to combine; `NULL` for all.

- method:

  `"nnls"` (default), `"ridge"`, `"mean"` (one scale type only) or
  `"cox"`.

- event_col:

  Event indicator, for `method = "cox"`.

- groups:

  An `obs` column whose members always share a fold.

## Value

A `falcon_stack`: `weights`, `metrics`, `comparison` (each clock alone
and the plain mean), and `predictions`.
