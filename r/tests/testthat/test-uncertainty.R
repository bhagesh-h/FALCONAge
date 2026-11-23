# =============================================================================
# Uncertainty, study design and batch, from R
# =============================================================================
#
# The R side is a bridge, not a second implementation, so these do not re-derive
# the arithmetic -- the Python suite does that. What is worth asserting here is
# that the bridge marshals every new return type correctly and that the R
# numbers are the same bits as the Python ones.
# =============================================================================

skip_if_no_python <- function() {
  testthat::skip_if_not(falconage_available(), "Python core not available")
}

synthetic <- function(n = 24L) {
  fa <- reticulate::import("falconage")
  reg <- fa$registry$load()
  feats <- sort(unique(unlist(lapply(
    c("horvath2013", "hannum", "dnamphenoage"),
    function(cid) as.character(reticulate::py_to_r(reg$feature_ids(cid)))))))
  set.seed(20260809)
  age <- seq(22, 82, length.out = n)
  base <- stats::runif(length(feats), 0.15, 0.85)
  drift <- stats::rnorm(length(feats), 0, 0.003)
  X <- outer(rep(1, n), base) + outer(age - 50, drift) +
    matrix(stats::rnorm(n * length(feats), 0, 0.01), n)
  X <- pmin(pmax(X, 0.001), 0.999)
  dimnames(X) <- list(sprintf("S%03d", seq_len(n)), feats)
  falcon_data(X, obs = data.frame(age = age, tissue = "whole blood",
                                  arm = rep(c("ctrl", "treat"), length.out = n),
                                  person = sprintf("P%02d", (seq_len(n) - 1L) %% (n %/% 2L)),
                                  visit = rep(c("v1", "v2"), each = n %/% 2L),
                                  row.names = rownames(X)))
}

test_that("technical_se crosses the bridge with its diagnostics", {
  skip_if_no_python()
  d <- synthetic()
  res <- score(d, clocks = c("horvath2013", "hannum"), min_coverage = 0)
  u <- technical_se(res, d)

  expect_equal(nrow(u$se), 24L)
  expect_setequal(colnames(u$se), c("horvath2013", "hannum"))
  expect_true(all(u$se > 0))
  expect_true(all(c("n_icc_published", "implied_cohort_icc", "se_over_sd")
                  %in% colnames(u$diagnostics)))
  # Provenance travels with the interval, or the interval is worth less than
  # nothing.
  expect_true(nzchar(u$source$sha256))
})

test_that("the R standard errors are the same bits as the Python ones", {
  skip_if_no_python()
  d <- synthetic()
  res <- score(d, clocks = "hannum", min_coverage = 0)
  r_se <- as.numeric(technical_se(res, d)$se[["hannum"]])

  fa <- reticulate::import("falconage")
  py_se <- as.numeric(reticulate::py_to_r(
    fa$technical_se(res$py, d$py)$se)[["hannum"]])
  expect_identical(r_se, py_se)
})

test_that("power_n reports the reliability split", {
  skip_if_no_python()
  p <- power_n("horvath2013", effect = 1, sd = 5, icc = 0.9)
  expect_s3_class(p, "falcon_power")
  expect_gt(p$n_per_group, 0)
  expect_lt(p$n_if_perfectly_measured, p$n_per_group)
  expect_equal(p$icc, 0.9)
})

test_that("power_n refuses to default the standard deviation", {
  skip_if_no_python()
  expect_error(power_n("horvath2013", effect = 1), "sd")
})

test_that("consensus returns a verdict that carries its counts", {
  skip_if_no_python()
  d <- synthetic()
  res <- score(d, clocks = "compatible")
  cons <- consensus(res, "arm", reference = "ctrl")
  expect_s3_class(cons, "falcon_consensus")
  expect_true(cons$verdict %in% c("supported", "unsupported", "inconclusive"))
  expect_match(cons$why, "Bonferroni")
  expect_true(all(c("p", "q_bh", "p_bonferroni") %in% colnames(cons$table)))
})

test_that("a paired consensus is the Python one, bit for bit", {
  skip_if_no_python()
  d <- synthetic()
  res <- score(d, clocks = c("horvath2013", "hannum", "dnamphenoage"), min_coverage = 0)
  cons <- consensus(res, "visit", reference = "v1", design = "paired",
                    subject_col = "person")
  expect_true(all(c("mdc95", "below_mdc", "n_subjects") %in% colnames(cons$table)))
  expect_true(all(cons$table$n_subjects == 12))
  py <- reticulate::import("falconage")$consensus(
    res$py, "visit", reference = "v1", design = "paired", subject_col = "person")
  expect_identical(cons$table$p, py$table$p)
  expect_error(consensus(res, "visit", design = "paired"), "subject_col")
})
