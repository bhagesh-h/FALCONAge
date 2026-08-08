# =============================================================================
# Fetching public data
# =============================================================================

#' Download public data by accession
#'
#' Dispatches on the shape of the accession: `GSE*`/`GSM*` to GEO, `E-MTAB-*` to
#' ArrayExpress, a DOI to Zenodo, `owner/name` to Hugging Face, or a full https
#' URL fetched directly.
#'
#' Credentialed archives -- dbGaP, EGA, Synapse, UK Biobank -- are documented
#' rather than automated. The access agreement is between you and the archive,
#' and a tool that made it one function call would be inviting people to breach
#' it. Once the files are local, read them with [read_betas()] or
#' [read_series_matrix()].
#'
#' @param accession An accession or URL.
#' @param want For GEO series: `"matrix"` (the default, metadata and values in
#'   one file), `"suppl"`, or `"both"`.
#' @param dry_run List what would be fetched, and how much, without fetching.
#'   Worth using first on a large series: a supplementary directory can be
#'   several gigabytes.
#'
#' @return A list with `files` (paths), `samples` (a data frame, when the source
#'   provides one) and `notes`.
#' @examples
#' \dontrun{
#' d <- download("GSE182991")
#' d$files
#' head(d$samples)
#'
#' download("GSE182991", want = "suppl", dry_run = TRUE)
#' }
#' @export
download <- function(accession, want = NULL, dry_run = FALSE) {
  args <- list(accession, dry_run = dry_run)
  if (!is.null(want)) args$want <- want
  res <- py_do(do.call(fa()$download, args))
  samples <- tryCatch({
    s <- res$samples
    if (inherits(s, "python.builtin.object") &&
        !identical(reticulate::py_to_r(reticulate::py_bool(s)), FALSE)) as_df(s) else NULL
  }, error = function(e) NULL)
  list(
    accession = reticulate::py_to_r(res$accession),
    source    = reticulate::py_to_r(res$source),
    files     = as.character(unlist_na(reticulate::py_to_r(
                  reticulate::import_builtins(convert = FALSE)$list(
                    reticulate::import_builtins(convert = FALSE)$map(
                      reticulate::import_builtins(convert = FALSE)$str, res$files))))),
    samples   = samples,
    notes     = as.character(unlist_na(reticulate::py_to_r(res$notes)))
  )
}

#' Inspect or clear the download cache
#'
#' The cache is content-addressed by URL, so two accessions that share a file
#' share the copy.
#'
#' @param confirm Required for `clear_cache()`. Without it the function reports
#'   how much would be deleted and stops -- a cache holding a re-downloadable
#'   gigabyte is still an hour of somebody's time.
#' @return `cache_info()` a data frame; `clear_cache()` the bytes freed.
#' @examples
#' \dontrun{
#' cache_info()
#' clear_cache(confirm = TRUE)
#' }
#' @export
cache_info <- function() {
  dl <- reticulate::import("falconage.download", convert = FALSE)
  as_df(py_do(dl$cache_info()), row_names = FALSE)
}

#' @rdname cache_info
#' @export
clear_cache <- function(confirm = FALSE) {
  dl <- reticulate::import("falconage.download", convert = FALSE)
  invisible(reticulate::py_to_r(py_do(dl$clear_cache(confirm = confirm))))
}
