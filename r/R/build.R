# =============================================================================
# Building, adding and stacking clocks
# =============================================================================

#' Fit a clock to one cohort, judged out of fold
#'
#' Every reported prediction is made by a model that never saw the sample. See
#' [compare_learners()] to choose the learner first.
#'
#' @param data A `falcon_data` whose `obs` holds `target`.
#' @param target The `obs` column to predict.
#' @param learner `"elasticnet"` (default), `"lasso"`, `"ridge"`,
#'   `"screened_elasticnet"`, `"pc_elasticnet"`, or a scikit-learn learner
#'   (`"random_forest"`, `"gradient_boosting"`, `"svr"`, `"mlp"`).
#' @param groups An `obs` column whose members always share a fold, such as the
#'   person when there are repeated visits.
#' @param transform `NULL`, or `"horvath"` for Horvath's log-linear age transform.
#' @param outer_folds,inner_folds Cross-validation folds.
#' @return A `falcon_clockfit`: `metrics`, `coefficients`, `predictions`, and the
#'   Python object in `py`.
#' @export
fit_clock <- function(data, target = "age", learner = "elasticnet", groups = NULL,
                      transform = NULL, outer_folds = 10L, inner_folds = 10L) {
  py <- py_do(fa()$fit_clock(data$py, target, learner = learner,
                             groups = or_none(groups), transform = or_none(transform),
                             outer_folds = as.integer(outer_folds),
                             inner_folds = as.integer(inner_folds)))
  coefs <- if (inherits(py$coefficients, "python.builtin.NoneType")) NULL
           else as_series(py$coefficients)
  structure(list(py = py, metrics = unlist(reticulate::py_to_r(py$metrics)),
                 coefficients = coefs, predictions = as_series(py$predictions)),
            class = "falcon_clockfit")
}

#' @export
print.falcon_clockfit <- function(x, ...) {
  m <- x$metrics
  cat(sprintf("fitted clock: out-of-fold MAE %.2f, MedAE %.2f, r %.3f (n = %d)\n",
              m[["mae"]], m[["medae"]], m[["r"]], as.integer(m[["n"]])))
  invisible(x)
}

#' Compare learners for building a clock
#'
#' @inheritParams fit_clock
#' @param learners Character vector of learner names.
#' @return A list: `table` (one row per learner), `selected` (the nested error of
#'   picking the best learner, the number to report), `picks` and `best`.
#' @export
compare_learners <- function(data, target = "age",
                             learners = c("elasticnet", "lasso", "ridge",
                                          "screened_elasticnet", "pc_elasticnet"),
                             groups = NULL, transform = NULL,
                             outer_folds = 5L, inner_folds = 3L) {
  out <- py_do(fa()$compare_learners(data$py, target,
                                     learners = reticulate::r_to_py(as.list(learners)),
                                     groups = or_none(groups), transform = or_none(transform),
                                     outer_folds = as.integer(outer_folds),
                                     inner_folds = as.integer(inner_folds)))
  list(table = as_df(out$table), selected = unlist(reticulate::py_to_r(out$selected)),
       picks = unlist(reticulate::py_to_r(out$picks)), best = reticulate::py_to_r(out$best))
}

#' Write a fitted clock or stack as a clock FALCONAge can add
#'
#' @param fit A `falcon_clockfit` from [fit_clock()] or a `falcon_stack` from
#'   [stack_clocks()].
#' @param clock_id The id the clock will be scored under.
#' @param outdir Folder to write `<clock_id>.yaml` and its files into.
#' @return The path of the YAML entry, for [add_clock()].
#' @export
clock_to_file <- function(fit, clock_id, outdir) {
  as.character(py_do(fit$py$to_clock(clock_id, outdir)))
}

#' Add a clock of your own for this session
#'
#' @param path A YAML clock entry, or a folder holding exactly one.
#' @param replace Replace a clock of your own added earlier under the same id.
#' @return The clock id.
#' @export
add_clock <- function(path, replace = FALSE) {
  reticulate::py_to_r(py_do(fa()$add_clock(path, replace = replace)))
}

#' Combine several clocks into one score, judged out of fold
#'
#' @param x A `falcon_result`.
#' @param target The `obs` column to predict (follow-up time for `"cox"`).
#' @param clocks Which scored clocks to combine; `NULL` for all.
#' @param method `"nnls"` (default), `"ridge"`, `"mean"` (one scale type only) or
#'   `"cox"`.
#' @param event_col Event indicator, for `method = "cox"`.
#' @param groups An `obs` column whose members always share a fold.
#' @return A `falcon_stack`: `weights`, `metrics`, `comparison` (each clock alone
#'   and the plain mean), and `predictions`.
#' @export
stack_clocks <- function(x, target = "age", clocks = NULL, method = "nnls",
                         event_col = NULL, groups = NULL) {
  py <- py_do(fa()$stack_clocks(x$py, target,
                                clocks = if (is.null(clocks)) reticulate::py_none()
                                         else reticulate::r_to_py(as.list(clocks)),
                                method = method, event_col = or_none(event_col),
                                groups = or_none(groups)))
  structure(list(py = py, weights = as_series(py$weights),
                 metrics = unlist(reticulate::py_to_r(py$metrics)),
                 comparison = as_df(py$comparison),
                 predictions = as_series(py$predictions)),
            class = "falcon_stack")
}
