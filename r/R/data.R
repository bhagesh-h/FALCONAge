# =============================================================================
# Readers and the falcon_data class
# =============================================================================

#' Read a beta matrix
#'
#' @param path CSV, TSV or parquet. Gzip is handled by extension.
#' @param samples_are `"auto"`, `"rows"` or `"columns"`. Auto decides from which
#'   axis carries probe-shaped identifiers, which is reliable because `cg` ids
#'   are unmistakable -- guessing from the shape alone gets a 1000-sample 450K
#'   matrix right and a 500,000-probe cohort wrong.
#' @param obs Optional per-sample annotation, row names matching the samples.
#'
#' @return A [falcon_data].
#' @examples
#' \dontrun{
#' d <- read_betas("betas.csv")
#' d <- read_betas("betas.parquet", obs = pheno)
#' }
#' @export
read_betas <- function(path, samples_are = "auto", obs = NULL) {
  py <- py_do(fa()$read_betas(
    path.expand(path), samples_are = samples_are,
    obs = if (is.null(obs)) reticulate::py_none() else as_pandas(obs)))
  new_falcon_data(py)
}

#' Read a GEO series matrix
#'
#' Metadata and beta values in one gzipped file, which is what roughly 60% of
#' GEO methylation series publish and nothing else. The characteristics block
#' becomes the sample annotation, with GEO's own key names kept verbatim -- an
#' `age` column that silently turned out to be `age at diagnosis` is worse than
#' no column.
#'
#' Some series carry metadata only and put the values in a supplementary file.
#' That is read too, and flagged, rather than treated as an error.
#'
#' @param path Path to a `*_series_matrix.txt.gz`.
#' @return A [falcon_data].
#' @examples
#' \dontrun{
#' d <- read_series_matrix("GSE66459_series_matrix.txt.gz")
#' head(obs(d))
#' }
#' @export
read_series_matrix <- function(path) {
  new_falcon_data(py_do(fa()$read_series_matrix(path.expand(path))))
}

#' Read a clinical chemistry table
#'
#' @param path CSV, TSV or parquet.
#' @param units Named list mapping marker to unit, e.g.
#'   `list(albumin = "g/L", creatinine = "umol/L")`. Optional here so that a
#'   file can be read to look at it; the clinical clocks require it and raise
#'   with the exact list to supply if it is missing.
#' @return A [falcon_data].
#' @examples
#' \dontrun{
#' d <- read_clinical("nhanes.csv",
#'                    units = list(albumin = "g/L", creatinine = "umol/L",
#'                                 glucose = "mmol/L", crp = "mg/dL"))
#' }
#' @export
read_clinical <- function(path, units = NULL) {
  new_falcon_data(py_do(fa()$read_clinical(
    path.expand(path),
    units = if (is.null(units)) reticulate::py_none() else reticulate::r_to_py(units))))
}

#' Read RRBS coverage files
#'
#' @param paths Character vector of per-sample site files.
#' @param min_coverage Minimum read depth for a site to be kept. A ratio from
#'   four reads and one from four hundred are not the same measurement, and a
#'   clock handed both without distinction reports sequencing depth as biology.
#' @return A [falcon_data] with `modality = "rrbs"`.
#' @examples
#' \dontrun{
#' d <- read_rrbs_dir(list.files("mouse", full.names = TRUE), min_coverage = 5)
#' }
#' @export
read_rrbs_dir <- function(paths, min_coverage = 5L) {
  new_falcon_data(py_do(fa()$read_rrbs_dir(
    reticulate::r_to_py(as.list(path.expand(paths))),
    min_coverage = as.integer(min_coverage))))
}

#' Build a dataset from an R matrix or data frame
#'
#' @param x Samples in rows, features in columns. Row names are the sample ids
#'   and are load-bearing: every join downstream is on them.
#' @param obs Per-sample annotation with matching row names.
#' @param modality `"dna_methylation"`, `"clinical_chemistry"` or `"rrbs"`.
#' @param platform Optional, e.g. `"450K"`. Detected from the probe identifiers
#'   when omitted.
#' @param species Which organism the samples came from. Checked, not assumed:
#'   the mammalian array carries 96% of Horvath2013's CpGs, so a zebra scores at
#'   high coverage and returns a confident number from a clock fitted on people.
#' @param units For clinical chemistry, a named list of marker to unit.
#' @return A [falcon_data].
#' @examples
#' \dontrun{
#' d <- falcon_data(betas, obs = pheno, modality = "dna_methylation")
#' }
#' @export
falcon_data <- function(x, obs = NULL, modality = "dna_methylation",
                        platform = NULL, species = "Homo sapiens", units = NULL) {
  core <- reticulate::import("falconage.core", convert = FALSE)
  # A pandas proxy passes straight through. Users who already have one -- from
  # reticulate, or from another package that returns Python objects -- should
  # not have to round-trip it through R just to hand it back.
  X <- if (inherits(x, "python.builtin.object")) x else as_pandas(as.data.frame(x))
  py <- py_do(core$FalconData(
    X = X,
    obs = if (is.null(obs)) reticulate::py_none()
          else if (inherits(obs, "python.builtin.object")) obs else as_pandas(obs),
    modality = modality,
    units = if (is.null(units)) reticulate::r_to_py(list()) else reticulate::r_to_py(units),
    platform = or_none(platform),
    species = species))
  new_falcon_data(py)
}

new_falcon_data <- function(py) {
  structure(list(py = py), class = "falcon_data")
}

#' @param x A `falcon_data`.
#' @param ... Ignored.
#' @rdname falcon_data
#' @export
print.falcon_data <- function(x, ...) {
  cat(reticulate::py_str(x$py), "\n")
  invisible(x)
}

#' @param object A `falcon_data`.
#' @param ... Ignored.
#' @rdname falcon_data
#' @export
summary.falcon_data <- function(object, ...) as_series(object$py$summary())

#' Sample annotation
#'
#' @param x A `falcon_data` or `falcon_result`.
#' @return A data frame, one row per sample.
#' @examples
#' \dontrun{
#' head(obs(d))
#' }
#' @export
obs <- function(x) UseMethod("obs")

#' @rdname obs
#' @export
obs.falcon_data <- function(x) as_df(x$py$obs)

#' @rdname obs
#' @export
obs.falcon_result <- function(x) as_df(x$py$obs)

#' Standard methylation preprocessing
#'
#' Harmonises probe identifiers, collapses EPIC v2 replicate suffixes, clips to
#' the beta range, and identifies the platform.
#'
#' The EPIC v2 step is not optional in effect. Illumina renamed `cg00000029` to
#' `cg00000029_TC21`, and every clock in the registry matches on the bare
#' identifier -- so without aggregation an EPIC v2 dataset overlaps almost
#' nothing, the imputation step fills everything, and the clock returns a
#' confident number computed from imputed values rather than an error.
#'
#' @param data A [falcon_data].
#' @param aggregate_epicv2 Collapse EPIC v2 replicate probes.
#' @param clip Clip values into `[0, 1]`.
#' @return A [falcon_data].
#' @examples
#' \dontrun{
#' d <- prepare(read_series_matrix("GSE330325_series_matrix.txt.gz"))
#' }
#' @export
prepare <- function(data, aggregate_epicv2 = TRUE, clip = TRUE) {
  new_falcon_data(py_do(fa()$prepare(data$py, aggregate_epicv2 = aggregate_epicv2,
                                     clip = clip)))
}

#' Quality control before scoring
#'
#' Reports rather than fixes. A sample that is 40% missing may be a failed array
#' or may be a 27K matrix aligned against an EPIC feature space, and the right
#' response differs -- so this says what it sees and leaves the decision where
#' it belongs.
#'
#' @param data A [falcon_data].
#' @return A list with `summary` (a named vector) and `warnings` (character).
#' @examples
#' \dontrun{
#' r <- qc(d)
#' r$summary
#' r$warnings
#' }
#' @export
qc <- function(data) {
  rep <- py_do(fa()$qc(data$py))
  list(summary  = as_series(rep$summary()),
       warnings = as.character(unlist_na(reticulate::py_to_r(rep$warnings))),
       per_sample = as_df(rep$per_sample))
}

#' Write a dataset to the interchange format both languages read
#'
#' @param data A [falcon_data].
#' @param path Destination `.h5ad`.
#' @return The path, invisibly.
#' @examples
#' \dontrun{
#' write_h5ad(d, "prepared.h5ad")
#' }
#' @export
write_h5ad <- function(data, path) {
  py_do(data$py$write_h5ad(path.expand(path)))
  invisible(path)
}

#' @rdname write_h5ad
#' @export
read_h5ad <- function(path) {
  core <- reticulate::import("falconage.core", convert = FALSE)
  new_falcon_data(py_do(core$FalconData$read_h5ad(path.expand(path))))
}
