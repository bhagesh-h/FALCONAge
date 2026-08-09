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

test_that("a frozen batch reference leaves an earlier plate untouched", {
  skip_if_no_python()
  set.seed(7)
  n_per <- 20L
  feats <- sprintf("cg%08d", seq_len(300))
  mk <- function(tag, shift) {
    age <- stats::runif(n_per, 25, 80)
    X <- 0.5 + outer(age - 50, rep(0.002, length(feats))) +
      matrix(stats::rnorm(n_per * length(feats), 0, 0.03), n_per) + shift
    dimnames(X) <- list(sprintf("%s%03d", tag, seq_len(n_per)), feats)
    list(X = X, obs = data.frame(age = age, plate = tag,
                                 row.names = rownames(X)))
  }
  a <- mk("A", 0)
  b <- mk("B", 0.08)
  d_a <- falcon_data(a$X, obs = a$obs)
  d_all <- falcon_data(rbind(a$X, b$X), obs = rbind(a$obs, b$obs))

  ref <- fit_batch_reference(d_a, "plate", covariates = "age")
  expect_s3_class(ref, "falcon_batch_reference")
  expect_equal(nchar(ref$digest), 64L)

  only_a <- apply_batch_reference(d_a, ref, "plate")
  with_b <- apply_batch_reference(d_all, ref, "plate")

  f1 <- reticulate::py_to_r(only_a$py$X)
  f2 <- reticulate::py_to_r(with_b$py$X)
  # Positional, then checked. reticulate does not always carry a pandas index
  # through as row names, and a name-based subset that silently matches nothing
  # would make this test pass on an empty comparison.
  keep <- seq_len(nrow(f1))
  expect_equal(nrow(f1), n_per)
  expect_equal(nrow(f2), 2L * n_per)
  a1 <- unname(as.matrix(f1))
  a2 <- unname(as.matrix(f2[keep, colnames(f1), drop = FALSE]))
  # Bit equality of the values, not of the names: subsetting a data frame in R
  # invents row labels, and the claim is about the numbers.
  expect_identical(a1, a2)
})
