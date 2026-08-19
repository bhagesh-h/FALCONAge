# =============================================================================
# The R surface against the real public corpus
# =============================================================================
#
# The clinical clocks live here rather than in the Python suite for a concrete
# reason: NHANES III and IV are published as .rda, which R reads natively and
# Python does not. This is the one place the R side does something the Python
# side cannot, and it is a file-format accident rather than a design split --
# the arithmetic still happens in Python.
# =============================================================================

corpus <- normalizePath(file.path("..", "..", "..", "test", "data"), mustWork = FALSE)

skip_unless_corpus <- function() {
  testthat::skip_if_not(falconage_available(), "Python core not available")
  testthat::skip_if_not(file.exists(file.path(corpus, "checksums.sha256")),
                        "test corpus absent; see test/data/README.md")
}

test_that("NHANES loads and the three clinical clocks run on it", {
  skip_unless_corpus()

  e <- new.env()
  load(file.path(corpus, "clinical", "NHANES3.rda"), envir = e)
  nh <- get(ls(e)[1], envir = e)
  expect_gt(nrow(nh), 1000)

  # BioAge's own column names, converted to FALCONAge's canonical markers and
  # to the units Levine's coefficients expect.
  map <- c(albumin = "albumin_gL", creatinine = "creat_umol", glucose = "glucose_mmol",
           crp = "lncrp", lymphocyte_percent = "lymph",
           mean_cell_volume = "mcv", red_cell_distribution_width = "rdw",
           alkaline_phosphatase = "alp", white_blood_cell_count = "wbc",
           age = "age")
  have <- map[map %in% names(nh)]
  testthat::skip_if(length(have) < 8,
                    paste("NHANES3 columns differ from expectation:",
                          paste(utils::head(names(nh), 20), collapse = ", ")))

  df <- nh[, unname(have), drop = FALSE]
  names(df) <- names(have)
  # BioAge stores CRP already logged; PhenoAge takes the raw value in mg/dL.
  if ("crp" %in% names(df)) df$crp <- exp(df$crp)
  df <- df[stats::complete.cases(df), , drop = FALSE]
  rownames(df) <- as.character(seq_len(nrow(df)))
  expect_gt(nrow(df), 500)

  d <- falcon_data(df, modality = "clinical_chemistry",
                   units = list(albumin = "g/L", creatinine = "umol/L",
                                glucose = "mmol/L", crp = "mg/dL",
                                lymphocyte_percent = "%", mean_cell_volume = "fL",
                                red_cell_distribution_width = "%",
                                alkaline_phosphatase = "U/L",
                                white_blood_cell_count = "10^3/uL", age = "years"))

  res <- score(d, clocks = "phenoage")
  s <- as.data.frame(res)
  expect_true(all(is.finite(s$phenoage)))
  # PhenoAge is dominated by its age term, as published.
  expect_gt(stats::cor(s$phenoage, df$age), 0.85)

  markers <- setdiff(names(df), "age")
  ref <- fit_kdm(df, markers)
  kres <- score(d, clocks = "kdm", reference = ref)
  expect_gt(stats::cor(as.data.frame(kres)$kdm, df$age), 0.5)

  href <- fit_hd(df[df$age < 35, , drop = FALSE], markers)
  hres <- score(d, clocks = "hd", reference = href)
  hv <- as.data.frame(hres)$hd
  expect_true(all(hv >= 0), info = "a Mahalanobis distance is never negative")
})

test_that("a GEO series matrix reads and scores from R", {
  skip_unless_corpus()
  d <- prepare(read_series_matrix(
    file.path(corpus, "gestational", "GSE66459_series_matrix.txt.gz")))
  expect_equal(nrow(obs(d)), 22L)
  expect_true("gestational_age_days" %in% names(obs(d)))

  # GSE66459 is umbilical cord blood. Knight was trained on it; leecontrol was
  # trained on placenta, and asking for it by name is a refusal --
  # an explicit request is never silently dropped.
  expect_error(score(d, clocks = c("knight", "leecontrol")), "category error")

  res <- score(d, clocks = "knight")
  s <- as.data.frame(res)
  expect_true(all(s$knight > 25 & s$knight < 50))
})

test_that("EPIC v2 suffixes are collapsed on the R side too", {
  skip_unless_corpus()
  raw <- read_series_matrix(file.path(corpus, "epicv2", "GSE330325_series_matrix.txt.gz"))
  prepped <- prepare(raw)
  res <- score(prepped, clocks = "horvath2013")
  expect_true(all(is.finite(as.data.frame(res)$horvath2013)))
})
