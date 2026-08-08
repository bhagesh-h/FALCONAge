# =============================================================================
# Downstream statistics
# =============================================================================

#' Age acceleration
#'
#' @param x A `falcon_result`.
#' @param method
#'   `"absolute"` -- predicted minus chronological. Interpretable in years, and
#'   confounded by the clock's own bias: a clock that over-predicts everyone by
#'   three years gives everyone three years of acceleration.
#'
#'   `"residual"` -- the residual from regressing predicted on chronological
#'   age. Centred at zero by construction, which removes that bias and also
#'   removes any real cohort-wide effect. The field's default.
#'
#'   `"within_group"` -- residual from a regression fitted separately within
#'   each level of `group`. What the AA2 benchmark needs: it asks whether cases
#'   accelerate relative to *their own* controls.
#' @param age_col Column in the sample annotation holding chronological age.
#' @param group Grouping column, required for `"within_group"`.
#' @param clocks Optional clock names. Naming them means every one must be
#'   valid, and a pace or log-hazard clock raises. Leaving it `NULL` means "the
#'   ones this makes sense for" and quietly excludes the others.
#'
#' @section Which convention a paper used:
#' Often not stated, and the three disagree by several years on the same data.
#' The convention is recorded in the returned frame's `method` attribute.
#'
#' @return A data frame of samples by clocks.
#' @examples
#' \dontrun{
#' acceleration(res)
#' acceleration(res, method = "within_group", group = "dataset")
#' }
#' @export
acceleration <- function(x, method = c("residual", "absolute", "within_group"),
                         age_col = "age", group = NULL, clocks = NULL) {
  method <- match.arg(method)
  out <- py_do(fa()$acceleration(
    x$py, age_col = age_col, method = method, group = or_none(group),
    clocks = if (is.null(clocks)) reticulate::py_none() else reticulate::r_to_py(as.list(clocks))))
  df <- as_df(out)
  attr(df, "method") <- method
  df
}

#' Associate clock scores with an outcome
#'
#' Ordinary least squares of each clock on an outcome, adjusted for covariates,
#' with Benjamini-Hochberg correction across clocks.
#'
#' @param x A `falcon_result`.
#' @param outcome Column in the sample annotation.
#' @param covariates Character vector of adjustment columns.
#' @param clocks Optional clock names.
#' @return A data frame with `beta`, `se`, `t`, `p` and `q`, sorted by p.
#' @examples
#' \dontrun{
#' associate(res, outcome = "bmi", covariates = c("age", "sex"))
#' }
#' @export
associate <- function(x, outcome, covariates = c("age", "sex"), clocks = NULL) {
  as_df(py_do(fa()$associate(
    x$py, outcome = outcome, covariates = reticulate::r_to_py(as.list(covariates)),
    clocks = if (is.null(clocks)) reticulate::py_none() else reticulate::r_to_py(as.list(clocks)))))
}

#' Univariable Cox hazard ratio per clock
#'
#' Breslow-tied partial likelihood by Newton-Raphson. Deliberately minimal:
#' competing risks and time-varying covariates belong in a survival package, and
#' pretending otherwise would be worse than saying so.
#'
#' @param x A `falcon_result`.
#' @param time_col,event_col Columns in the sample annotation.
#' @param clocks Optional clock names.
#' @return A data frame with `hr`, its 95% interval, `p` and `q`.
#' @examples
#' \dontrun{
#' cox_hazard(res, time_col = "permth_exm", event_col = "mortstat")
#' }
#' @export
cox_hazard <- function(x, time_col, event_col, clocks = NULL) {
  as_df(py_do(fa()$cox_hazard(
    x$py, time_col = time_col, event_col = event_col,
    clocks = if (is.null(clocks)) reticulate::py_none() else reticulate::r_to_py(as.list(clocks)))))
}

#' Between-clock agreement
#'
#' @param x A `falcon_result`.
#' @param method `"spearman"` (the default) or `"pearson"`. Spearman because
#'   two clocks on different scales -- years and a log-hazard -- have no
#'   meaningful Pearson correlation but a perfectly meaningful rank one, and
#'   mixing scales is the normal case rather than the exception.
#' @return A correlation matrix.
#' @examples
#' \dontrun{
#' round(agreement(res), 2)
#' }
#' @export
agreement <- function(x, method = c("spearman", "pearson")) {
  as.matrix(as_df(py_do(fa()$agreement(x$py, method = match.arg(method)))))
}

#' Intraclass correlation, ICC(2,1)
#'
#' Two-way random effects, absolute agreement, single measure -- the variant
#' that answers "would a repeat measurement of this person give the same
#' number". ICC(3,1) assumes the raters are the only ones of interest and
#' reports a higher number for the same data; papers rarely say which they used.
#'
#' @param values A data frame with one row per measurement.
#' @param subject_col,value_col Column names.
#' @return A single numeric.
#' @examples
#' \dontrun{
#' icc(replicates, "subject", "horvath2013")
#' }
#' @export
icc <- function(values, subject_col, value_col) {
  py_do(reticulate::py_to_r(
    fa()$icc(as_pandas(values), subject_col, value_col)))
}

#' The AA1 and AA2 benchmark
#'
#' Tests a clock the way the field now expects: does it show higher age
#' acceleration in people with an aging-accelerating condition than in their own
#' controls? Median absolute error against chronological age does not answer
#' that -- a perfect chronological oracle would score best on it and be useless.
#'
#' **AA2**, for a dataset with controls: is the condition group's acceleration
#' higher than its controls'? One-sided Mann-Whitney, BH corrected.
#'
#' **AA1**, for a dataset without controls: is it above zero? One-sided
#' Wilcoxon signed-rank.
#'
#' **MedE**, the median signed error on healthy controls, discounts the AA1
#' credit: `total = AA2 + AA1 * (1 - max(0, MedE) / MedAE)`. Without it a clock
#' that simply over-predicts everybody sweeps AA1, because every group looks
#' accelerated when the baseline is wrong.
#'
#' @param x A `falcon_result`, usually from [combine()] across studies.
#' @param condition_col,control Column naming the condition, and the value that
#'   marks a control.
#' @param dataset_col Column naming the study, when several are combined.
#' @param age_col Chronological age column.
#' @param alpha FDR threshold.
#' @return A list with `summary` (per clock) and `per_dataset` (per comparison).
#' @examples
#' \dontrun{
#' b <- run_benchmark(res, condition_col = "condition", control = "HC",
#'                    dataset_col = "dataset")
#' b$summary
#' }
#' @export
run_benchmark <- function(x, condition_col = "condition", control = "HC",
                          dataset_col = NULL, age_col = "age", alpha = 0.05) {
  b <- py_do(fa()$run_benchmark(
    x$py, condition_col = condition_col, control = control,
    dataset_col = or_none(dataset_col), age_col = age_col, alpha = alpha))
  structure(list(summary = as_df(b$summary_table),
                 per_dataset = as_df(b$per_dataset, row_names = FALSE)),
            class = "falcon_benchmark")
}

#' @param x A `falcon_benchmark`.
#' @param ... Ignored.
#' @rdname run_benchmark
#' @export
print.falcon_benchmark <- function(x, ...) {
  cat("FALCONAge benchmark:", nrow(x$summary), "clocks,",
      nrow(x$per_dataset), "comparisons,",
      sum(x$per_dataset$significant), "significant\n\n")
  print(x$summary)
  invisible(x)
}

# =============================================================================
# Clinical references
# =============================================================================

#' Fit a Klemera-Doubal reference
#'
#' KDM has no fixed coefficients. Each biomarker is regressed on chronological
#' age in a reference cohort and the panel is inverted to a maximum-likelihood
#' age, so the same person scored against NHANES III and against a hospital
#' cohort gets two different numbers, both correct. The manifest records which
#' reference was used.
#'
#' @param reference A data frame with the markers and an age column.
#' @param markers Character vector of column names.
#' @param age_col Age column name.
#' @return A reference object to pass to [score()].
#' @examples
#' \dontrun{
#' ref <- fit_kdm(nhanes3, markers = c("albumin", "creatinine", "glucose"))
#' score(d, clocks = "kdm", reference = ref)
#' }
#' @export
fit_kdm <- function(reference, markers, age_col = "age") {
  cl <- reticulate::import("falconage.models.clinical", convert = FALSE)
  py_do(cl$fit_kdm(as_pandas(reference), reticulate::r_to_py(as.list(markers)),
                   age_col = age_col))
}

#' Fit a homeostatic dysregulation reference
#'
#' The reference should be the healthy young subset, not the whole cohort.
#' Fitting the centre on everybody makes the average unhealthy person the
#' definition of normal, which is a different measurement with the same units.
#'
#' @param reference A data frame of the healthy reference population.
#' @param markers Character vector of column names.
#' @return A reference object to pass to [score()].
#' @examples
#' \dontrun{
#' ref <- fit_hd(nhanes3_hdtrain, markers = markers)
#' score(d, clocks = "hd", reference = ref)
#' }
#' @export
fit_hd <- function(reference, markers) {
  cl <- reticulate::import("falconage.models.clinical", convert = FALSE)
  py_do(cl$fit_hd(as_pandas(reference), reticulate::r_to_py(as.list(markers))))
}
