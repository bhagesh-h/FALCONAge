# Building, adding and stacking clocks from R call the same Python core.

skip_if_not(FALCONAge::falconage_available(), "Python core not available")

demo <- function(n = 40L) FALCONAge:::new_falcon_data(FALCONAge:::fa()$demo_data(n = n))

test_that("stack_clocks returns weights, metrics and the single-clock comparison", {
  res <- score(demo(), clocks = c("horvath2013", "hannum", "dnamphenoage"))
  st <- stack_clocks(res, target = "age", method = "nnls")
  expect_s3_class(st, "falcon_stack")
  expect_true(all(st$weights >= 0))
  expect_true(all(c("mae", "r") %in% names(st$metrics)))
  expect_true("plain mean" %in% rownames(st$comparison))
})

test_that("a stack written as a clock is added and scores like the stack", {
  d <- demo()
  res <- score(d, clocks = c("hannum", "dnamphenoage"))
  st <- stack_clocks(res, target = "age", method = "ridge")
  path <- clock_to_file(st, "rstack", tempdir())
  id <- add_clock(path, replace = TRUE)
  again <- as.data.frame(score(d, clocks = id))[[id]]
  full <- FALCONAge:::as_series(st$py$predict(res$py))
  expect_equal(unname(again), unname(full), tolerance = 1e-6)
})

test_that("a plain mean across scale types is refused", {
  res <- score(demo(), clocks = c("horvath2013", "dunedinpoam38"))
  expect_error(stack_clocks(res, method = "mean"), "no unit")
})
