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

test_that("the coverage map has a track and a coverage for every methylation clock", {
  skip_if_not_installed("ggplot2")
  p <- plot_clock_coverage(demo(10L))
  expect_s3_class(p, "ggplot")
  tab <- attr(p, "table")
  expect_true("horvath2013" %in% rownames(tab))
  expect_true(all(as.numeric(tab$coverage) == 100))
})

test_that("figures are drawn without a title and saved under it with a .txt", {
  skip_if_not(falconage_available())
  skip_if_not_installed("ggplot2")
  p <- plot_clock_coverage(platform = "450K")
  t <- figure_text(p)
  expect_equal(t$title, "Clock coverage across the genome")
  expect_true(any(grepl("element_blank", class(p$theme$plot.title))))
  out <- tempfile()
  png <- save_figure(p, out, width = 12, height = 9)
  expect_equal(basename(png), "clock_coverage_across_the_genome.png")
  txt <- readLines(file.path(out, "clock_coverage_across_the_genome.txt"))
  expect_equal(txt[1], "Clock coverage across the genome")
})
