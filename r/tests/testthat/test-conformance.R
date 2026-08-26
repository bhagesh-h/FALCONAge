# =============================================================================
# R/Python conformance
# =============================================================================
#
# The claim the whole architecture rests on: an R result and a Python result are
# the same bits, not two implementations that agree to six decimals. These tests
# assert bit equality (tolerance exactly zero), because approximate agreement is
# what a second implementation would give and the point of the bridge is to have
# only one.
# =============================================================================

skip_if_no_python <- function() {
  testthat::skip_if_not(falconage_available(), "Python core not available")
}

test_that("the module imports and reports a coherent configuration", {
  skip_if_no_python()
  cfg <- falconage_config()
  expect_s3_class(cfg, "falcon_config")
  expect_equal(cfg$n_clocks, 175L)
  expect_equal(cfg$falconage, as.character(utils::packageVersion("FALCONAge")))
  tiers <- cfg$clocks_by_availability
  expect_equal(tiers$bundled + tiers$untraced + tiers$licensed, 175L)
})

test_that("the registry browses from R", {
  skip_if_no_python()
  a <- list_clocks(tier = "bundled")
  expect_gte(nrow(a), 20)
  expect_true(all(a$availability == "bundled"))
  expect_true("horvath2013" %in% rownames(a))

  scaffolds <- list_clocks(tier = "licensed")
  expect_equal(nrow(scaffolds), 40L)
})

test_that("a scaffold clock says why and names an alternative", {
  skip_if_no_python()
  out <- capture.output(clock_info("grimage2"))
  txt <- paste(out, collapse = "\n")
  expect_match(txt, "scaffold-only")
  expect_match(txt, "Obtain them from")
  expect_match(txt, "Open alternatives")
})

test_that("scores from R are bit-identical to scores from Python", {
  skip_if_no_python()

  # Build the same synthetic dataset on both sides from the same seed, score it
  # in each language, and compare with tolerance exactly zero.
  py <- reticulate::import_builtins(convert = FALSE)
  reticulate::py_run_string("
import numpy as np, pandas as pd, falconage as fa
_reg = fa.registry.load()
_feats = sorted(set(_reg.feature_ids('horvath2013')) | set(_reg.feature_ids('hannum')))
_rng = np.random.default_rng(4242)
_n = 12
_age = np.linspace(25, 75, _n)
_base = _rng.uniform(0.2, 0.8, size=len(_feats))
_X = np.clip(_base[None, :] + _rng.normal(0, 0.02, size=(_n, len(_feats))), 0.001, 0.999)
_ids = [f'S{i:03d}' for i in range(_n)]
_df = pd.DataFrame(_X, index=_ids, columns=_feats)
_obs = pd.DataFrame({'age': _age}, index=_ids)
_data = fa.FalconData(X=_df, obs=_obs, modality='dna_methylation', platform='450K')
_py_scores = fa.score(_data, clocks=['horvath2013', 'hannum']).scores
")
  py_scores <- FALCONAge:::as_df(reticulate::py_eval("_py_scores", convert = FALSE))

  # Same data, but assembled and scored through the R surface.
  betas <- FALCONAge:::as_df(reticulate::py_eval("_df", convert = FALSE))
  pheno <- FALCONAge:::as_df(reticulate::py_eval("_obs", convert = FALSE))
  d <- falcon_data(betas, obs = pheno, modality = "dna_methylation", platform = "450K")
  r_scores <- as.data.frame(score(d, clocks = c("horvath2013", "hannum")))

  expect_equal(rownames(r_scores), rownames(py_scores))
  expect_equal(colnames(r_scores), colnames(py_scores))
  for (cl in colnames(py_scores)) {
    expect_identical(r_scores[[cl]], py_scores[[cl]],
                     info = paste(cl, "must be bit-identical, not merely close"))
  }
})

test_that("the manifest records the coefficient digest and the R caller", {
  skip_if_no_python()
  reticulate::py_run_string("
import numpy as np, pandas as pd, falconage as fa
_f = list(fa.registry.load().feature_ids('hannum'))
_d2 = pd.DataFrame(np.full((4, len(_f)), 0.5), index=list('abcd'), columns=_f)
")
  betas <- FALCONAge:::as_df(reticulate::py_eval("_d2", convert = FALSE))
  d <- falcon_data(betas, modality = "dna_methylation")
  res <- score(d, clocks = "hannum")
  m <- manifest(res)

  expect_equal(m$caller, "R", info = "an R-originated run is distinguishable")
  expect_equal(nchar(m$weights$hannum$sha256), 64L)
  expect_equal(m$dtype, "float64")
})

test_that("scale types are enforced from R too", {
  skip_if_no_python()
  reticulate::py_run_string("
import numpy as np, pandas as pd, falconage as fa
_f3 = list(fa.registry.load().feature_ids('dunedinpoam38'))
_d3 = pd.DataFrame(np.full((6, len(_f3)), 0.5), index=[f'x{i}' for i in range(6)], columns=_f3)
_o3 = pd.DataFrame({'age': [30, 40, 50, 60, 70, 80]}, index=_d3.index)
")
  d <- falcon_data(FALCONAge:::as_df(reticulate::py_eval("_d3", convert = FALSE)),
                   obs = FALCONAge:::as_df(reticulate::py_eval("_o3", convert = FALSE)),
                   modality = "dna_methylation")
  res <- score(d, clocks = "dunedinpoam38")
  expect_error(acceleration(res, clocks = "dunedinpoam38"), "already a rate")
})

test_that("a Python error arrives as an R condition with its message intact", {
  skip_if_no_python()
  reticulate::py_run_string("
import numpy as np, pandas as pd, falconage as fa
_f4 = list(fa.registry.load().feature_ids('hannum'))[:5]
_d4 = pd.DataFrame(np.full((3, 5), 0.5), index=list('abc'), columns=_f4)
")
  d <- falcon_data(FALCONAge:::as_df(reticulate::py_eval("_d4", convert = FALSE)),
                   modality = "dna_methylation")
  # The message must survive the crossing -- it is where the remedy lives.
  expect_error(score(d, clocks = "hannum"), "below the")
  expect_error(score(d, clocks = "grimage2"), "scaffold-only")
})
