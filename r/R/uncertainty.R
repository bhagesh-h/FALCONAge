# =============================================================================
# Uncertainty, study design, and the things that decide whether a score moved
# =============================================================================
# Thin wrappers over the same Python core the rest of the package uses, so the
# numbers are identical by construction rather than by agreement.

#' Technical standard error on each score
#'
#' How much of a score is the assay rather than the person. Every
#' implementation of every aging clock, including this one before v1.1,
#' reported a point estimate for a quantity whose technical replicates differ
#' by up to nine years (Nat Aging 2022, s43587-022-00248-2).
#'
#' For a linear clock the propagation is one line of algebra: with per-probe
#' measurement variance `s^2 (1 - ICC)`, the score's variance is the weighted
#' sum `sum_j w_j^2 s_j^2 (1 - ICC_j)`, carried through the clock's output
#' transform by the delta method.
#'
#' @param x A `falcon_result`.
#' @param data The `falcon_data` that was scored. Required for the per-probe
#'   path: the spread of each feature has to come from the matrix, not from the
#'   scores.
#' @param source `"auto"` (per-probe where possible, per-clock otherwise),
#'   `"probe"`, or `"clock"`.
#'
#' @section What this is not:
#' Measurement error, not prediction error. A clock can be perfectly repeatable
#' and still be a poor estimate of anything; see [conformal_interval()] for that
#' question. It also says nothing about biological variability -- the same
#' person sampled a fortnight later is a different measurement of a different
#' thing.
#'
#' @section An imputed feature widens the interval:
#' Deliberately, and it is the part most likely to look wrong. An imputed probe
#' is the cohort mean, so it carries no information about the sample in front of
#' you and contributes its whole between-sample variance. Treating it as well
#' measured would make worse data produce a narrower interval.
#'
#' @return A list with `se` (samples by clocks), `diagnostics` (per clock: how
#'   many probes had a published ICC, how many fell back to the median, and the
#'   ICC this cohort implies) and `refused` (clocks with no usable source).
#' @examples
#' \dontrun{
#' u <- technical_se(res, d)
#' u$diagnostics
#' }
#' @export
technical_se <- function(x, data = NULL, source = c("auto", "probe", "clock")) {
  source <- match.arg(source)
  out <- py_do(fa()$technical_se(x$py, if (is.null(data)) reticulate::py_none() else data$py,
                                 source = source))
  list(se = as_df(out$se), diagnostics = as_df(out$diagnostics),
       refused = reticulate::py_to_r(out$refused),
       source = reticulate::py_to_r(out$source))
}


#' Per-probe reliability from your own technical replicates
#'
#' Preferred over the bundled table whenever it is available, because it is this
#' laboratory's noise on this platform rather than a published cohort's.
#' A one-way random-effects single-measurement ICC is the right model when the
#' repeated measurements are interchangeable array positions rather than named
#' assessors.
#'
#' Negative values are kept rather than clipped: a negative ICC means the
#' within-subject spread exceeded the between-subject spread, which is a real
#' and reportable state of affairs for a probe that measures nothing.
#'
#' @param data A `falcon_data` containing replicates.
#' @param subject_col Column naming the subject each sample came from.
#' @return A named numeric vector of ICCs, one per feature.
#' @export
icc_from_replicates <- function(data, subject_col) {
  out <- py_do(fa()$icc_from_replicates(data$py, subject_col))
  stats::setNames(as.numeric(reticulate::py_to_r(out$values)),
                  as.character(reticulate::py_to_r(out$index$tolist())))
}


#' Scores with their measurement interval
#'
#' @param x A `falcon_result`.
#' @param data The `falcon_data` that was scored.
#' @param level Coverage, default 0.95.
#' @return A data frame with one row per sample per clock: value, se, lo, hi.
#' @export
interval <- function(x, data = NULL, level = 0.95) {
  as_df(py_do(fa()$interval(x$py,
                            if (is.null(data)) reticulate::py_none() else data$py,
                            level = level)))
}


#' Distribution-free prediction intervals against chronological age
#'
#' A different question from [technical_se()], with a larger answer. Technical
#' error asks what a repeat of the same DNA would do; this asks how far the
#' number is likely to be from the truth.
#'
#' Split conformal: the half-width is a quantile of the absolute residual on a
#' calibration set of healthy blood samples with known ages, so on any sample
#' exchangeable with that cohort the interval covers the truth at the stated
#' rate, with no distributional assumption and a finite-sample guarantee.
#'
#' @section The limit, which is not small:
#' Coverage holds for data *exchangeable with the calibration cohort* -- public
#' blood data, adult, overwhelmingly of European ancestry. On a paediatric or
#' non-European cohort the guarantee does not transfer, which is why every row
#' carries `exchangeable = FALSE`: nothing here can verify it, so nothing here
#' implies it.
#'
#' @param x A `falcon_result`.
#' @param level Coverage. One of 0.80, 0.90, 0.95.
#' @param clocks Optional clock names; only age-scale clocks are calibrated.
#' @return A data frame with value, lo, hi, half_width, median_bias and mae.
#' @export
conformal_interval <- function(x, level = 0.90, clocks = NULL) {
  as_df(py_do(fa()$conformal_interval(
    x$py, level = level,
    clocks = if (is.null(clocks)) reticulate::py_none() else reticulate::r_to_py(as.list(clocks)))))
}


#' How many samples to see an effect
#'
#' The first thing a laboratory needs, and it is needed before any array is run.
#' Two independent groups, two-sided.
#'
#' Reliability is part of the answer: the SD a user measures already contains
#' the assay's noise, so splitting it out with the clock's test-retest ICC says
#' how much of the sample size is buying signal and how much is averaging out
#' the instrument. That is the arithmetic behind the finding that the original
#' clocks need 3-16 replicates per condition where their principal-component
#' versions need 1-2.
#'
#' @param clock Clock name.
#' @param effect The difference worth detecting, in the clock's own unit. No
#'   default: a power calculation with an assumed effect size is a way of
#'   writing down an assumption without noticing.
#' @param sd Population SD of the score. Measured from `result` when given.
#' @param result A scored pilot. Supplies `sd`, and -- if [technical_se()] has
#'   been called on it -- a measured ICC for this laboratory.
#' @param icc Override the reliability figure.
#' @param alpha,power Significance and target power.
#' @param replicates Assay each sample this many times and average.
#' @return A list with `n_per_group`, `n_total`, the reliability used and where
#'   it came from, and the n a perfectly repeatable assay would need.
#' @examples
#' \dontrun{
#' power_n("horvath2013", effect = 1, sd = 5, icc = 0.9)
#' }
#' @export
power_n <- function(clock, effect, sd = NULL, result = NULL, icc = NULL,
                    alpha = 0.05, power = 0.80, replicates = 1L) {
  out <- py_do(fa()$power(clock, effect = effect, sd = or_none(sd),
                          result = if (is.null(result)) reticulate::py_none() else result$py,
                          icc = or_none(icc), alpha = alpha, power = power,
                          replicates = as.integer(replicates)))
  structure(list(
    clock = clock, effect = effect,
    n_per_group = as.integer(reticulate::py_to_r(out$n_per_group)),
    n_total = as.integer(reticulate::py_to_r(out$n_total)),
    sd = reticulate::py_to_r(out$sd),
    icc = reticulate::py_to_r(out$icc),
    icc_source = reticulate::py_to_r(out$icc_source),
    n_if_perfectly_measured = reticulate::py_to_r(out$n_if_perfectly_measured),
    assumptions = reticulate::py_to_r(out$assumptions)
  ), class = "falcon_power")
}

#' @export
print.falcon_power <- function(x, ...) {
  cat(sprintf("%s: n = %d per group to see %g at %d%% power\n",
              x$clock, x$n_per_group, x$effect, 80L))
  cat(sprintf("  sd %.4g (%s)\n", x$sd, x$assumptions))
  if (is.null(x$icc)) {
    cat("  reliability not established; n is not adjusted for measurement error\n")
  } else {
    cat(sprintf("  technical ICC %.3f (%s)\n", x$icc, x$icc_source))
    if (!is.null(x$n_if_perfectly_measured)) {
      cat(sprintf("  %d of those samples exist only to average out the assay\n",
                  x$n_total - 2L * as.integer(x$n_if_perfectly_measured)))
    }
  }
  invisible(x)
}


#' Does a group difference hold up across clocks?
#'
#' Implements the decision rule from *When to Trust Epigenetic Clocks*
#' (PMC11526921). Re-analysing six intervention datasets, the authors found that
#' in five of them exactly one clock reached significance -- a first-generation
#' clock every time -- and four of those five lost it under multiple-testing
#' correction. In no case did the principal-component version of the same clock
#' corroborate the finding. Their conclusion, stated plainly: a single
#' significant clock after an intervention is likely a false positive.
#'
#' Each clock is tested on its acceleration residual where that is a legal
#' operation for its scale, and on the raw score where it is not. A pace of
#' aging has no residual to take.
#'
#' @param x A `falcon_result`.
#' @param group_col Column with exactly two levels.
#' @param reference Which level is the comparison group.
#' @param alpha Significance level.
#' @return A list with `verdict` (`supported`, `unsupported`, `inconclusive`),
#'   `why` -- which always carries the counts it was computed from -- and the
#'   per-clock `table`.
#' @export
consensus <- function(x, group_col, reference = NULL, alpha = 0.05) {
  out <- py_do(fa()$consensus(x$py, group_col, reference = or_none(reference),
                              alpha = alpha))
  structure(list(verdict = reticulate::py_to_r(out$verdict),
                 why = reticulate::py_to_r(out$why),
                 table = as_df(out$table),
                 n_tests = as.integer(reticulate::py_to_r(out$n_tests))),
            class = "falcon_consensus")
}

#' @export
print.falcon_consensus <- function(x, ...) {
  cat(sprintf("verdict: %s\n  %s\n", x$verdict, x$why))
  invisible(x)
}


# =============================================================================
# Batch correction that does not move a result you already reported
# =============================================================================

#' Fit a frozen batch-correction reference
#'
#' ComBat estimates its parameters from every sample it is given at once, so
#' adding a plate and re-running changes every previously corrected value --
#' measured at up to 2.20 years of shift in already-reported epigenetic ages
#' (PMC12495439). Freezing the global parameters and the empirical-Bayes priors
#' on a reference cohort removes that: every later batch is standardised against
#' the frozen values and gets only its own effects estimated.
#'
#' The reference is an artefact you keep and version-control, like a coefficient
#' file. That is the whole design; without it this is ComBat with extra steps.
#'
#' @param data A `falcon_data` for the reference cohort.
#' @param batch_col Column naming the batch.
#' @param covariates Columns whose effect should be preserved rather than
#'   removed, typically age and sex.
#' @param protect Columns checked for being nested inside batch. A confounded
#'   design is refused, because correcting it removes the effect along with the
#'   artefact and returns a clean-looking null.
#' @return A `falcon_batch_reference`, with a digest for the run manifest.
#' @export
fit_batch_reference <- function(data, batch_col, covariates = character(0),
                                protect = c("condition", "group")) {
  py <- py_do(fa()$fit_batch_reference(
    data$py, batch_col = batch_col,
    covariates = reticulate::r_to_py(as.list(covariates)),
    protect = reticulate::r_to_py(as.list(protect))))
  structure(list(py = py, digest = reticulate::py_to_r(py$digest)),
            class = "falcon_batch_reference")
}

#' @export
print.falcon_batch_reference <- function(x, ...) {
  cat(sprintf("<falcon_batch_reference %s>\n", substr(x$digest, 1, 12)))
  invisible(x)
}

#' Apply a frozen batch-correction reference
#'
#' @param data A `falcon_data` to correct.
#' @param reference A `falcon_batch_reference` from [fit_batch_reference()].
#' @param batch_col Column naming the batch.
#' @return A corrected `falcon_data`, recording the reference's digest.
#' @export
apply_batch_reference <- function(data, reference, batch_col) {
  py <- py_do(fa()$apply_batch_reference(data$py, reference$py, batch_col = batch_col))
  new_falcon_data(py)
}
