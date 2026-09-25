# =============================================================================
# Scoring and the falcon_result class
# =============================================================================

#' Score a dataset against one or more clocks
#'
#' @param data A `falcon_data`.
#' @param clocks `"compatible"` scores everything this dataset can support and
#'   reports the rest as skipped with a reason. `"all"` attempts every clock of
#'   the right modality and fails loudly on the ones that cannot run. A
#'   character vector names them explicitly, and then every name must work -- an
#'   explicit request is never silently dropped.
#' @param device `"auto"`, `"cpu"`, `"cuda"` or `"mps"`. Naming a device that is
#'   not present is an error rather than a silent downgrade: a run that was
#'   asked for a GPU and quietly used a CPU looks like a very slow success.
#'   `"auto"` is CPU even where CUDA exists, which is measured rather than
#'   cautious: on the clocks that ship, the transfer costs more than the dot
#'   product it feeds. A named device is not granted to every clock either --
#'   the three clinical formulas compute in numpy whatever is asked, because
#'   nine markers are not worth a kernel launch. `manifest(res)$compute` records
#'   what each clock actually used.
#' @param dtype `NULL` (float64), `"float64"` or `"float32"`. Clocks flagged
#'   `requires_fp64` in the registry override a float32 request, with a warning.
#' @param imputation How to fill a clock feature the data does not carry.
#'   `"reference"` uses the value the clock's authors published where one
#'   exists; `"mean"` uses the column mean; `"none"` refuses, so the coverage
#'   check fails loudly instead of being papered over. Zero is never used --
#'   in beta space it means completely unmethylated, which is a real and extreme
#'   measurement. The mitotic mean, percentile and transmission clocks ignore
#'   it: their published statistic is taken over the features present, so absent
#'   ones are left out and the coverage record reads `"excluded"`.
#' @param min_coverage Fraction of a clock's features that must be present.
#' @param reference A reference fitted with [fit_kdm()] or [fit_hd()], for the
#'   two clinical clocks that have no fixed coefficients.
#'
#' @return A `falcon_result`: an S3 object wrapping the scores, the per-clock
#'   coverage and the run manifest.
#' @aliases falcon_result
#' @examples
#' \dontrun{
#' res <- score(d, clocks = "compatible")
#' res <- score(d, clocks = c("horvath2013", "dnamphenoage"), device = "cuda")
#' }
#' @export
score <- function(data, clocks = "compatible", device = "auto", dtype = NULL,
                  imputation = c("reference", "mean", "none"),
                  min_coverage = 0.8, reference = NULL) {
  imputation <- match.arg(imputation)
  py <- py_do(fa()$score(
    data$py,
    clocks = if (length(clocks) == 1L && clocks %in% c("compatible", "all"))
      clocks else reticulate::r_to_py(as.list(clocks)),
    device = device, dtype = or_none(dtype), imputation = imputation,
    min_coverage = min_coverage, reference = or_none(reference),
    caller = "R"))
  new_falcon_result(py)
}

new_falcon_result <- function(py) structure(list(py = py), class = "falcon_result")

#' @param x A `falcon_result`.
#' @param ... Ignored.
#' @rdname score
#' @export
print.falcon_result <- function(x, ...) {
  cat(reticulate::py_str(x$py), "\n")
  invisible(x)
}

#' @param object A `falcon_result`.
#' @param ... Ignored.
#' @rdname score
#' @export
summary.falcon_result <- function(object, ...) as_df(object$py$summary())

#' Scores as a data frame
#'
#' @param x A `falcon_result`.
#' @param row.names Ignored; sample ids are always the row names.
#' @param optional Ignored.
#' @param form `"wide"` (samples x clocks) or `"long"` (one row per sample per
#'   clock, carrying the scale and the provenance).
#' @param ... Ignored.
#'
#' @section Why long form carries the scale:
#' It stops a reader averaging a mortality log-hazard with an age in years
#' because both were numbers in a column called `value`.
#'
#' @return A data frame.
#' @examples
#' \dontrun{
#' as.data.frame(res)
#' as.data.frame(res, form = "long")
#' }
#' @export
as.data.frame.falcon_result <- function(x, row.names = NULL, optional = FALSE,
                                        form = c("wide", "long"), ...) {
  form <- match.arg(form)
  if (form == "wide") as_df(x$py$wide()) else as_df(x$py$long(), row_names = FALSE)
}

#' Per-clock coverage and skip reasons
#'
#' @param x A `falcon_result`.
#' @return A data frame.
#' @examples
#' \dontrun{
#' coverage(res)
#' }
#' @export
coverage <- function(x) UseMethod("coverage")

#' @rdname coverage
#' @export
coverage.falcon_result <- function(x) as_df(x$py$qc(), row_names = FALSE)

#' The run manifest
#'
#' Versions, device, dtype, the SHA-256 of every coefficient file used, per-clock
#' coverage, and every warning raised. Two runs reporting the same score either
#' used the same coefficients or the manifest says they did not.
#'
#' @param x A `falcon_result`.
#' @return A named list.
#' @examples
#' \dontrun{
#' m <- manifest(res)
#' m$weights$horvath2013$sha256
#' }
#' @export
manifest <- function(x) UseMethod("manifest")

#' @rdname manifest
#' @export
manifest.falcon_result <- function(x) reticulate::py_to_r(x$py$manifest$to_dict())

#' Write the standard results layout
#'
#' `scores.csv`, `scores_wide.csv`, `qc.csv` and `run_manifest.json`.
#'
#' @param x A `falcon_result`.
#' @param outdir Destination directory, created if absent.
#' @return A named character vector of what was written, invisibly.
#' @examples
#' \dontrun{
#' write_results(res, "results/")
#' }
#' @export
write_results <- function(x, outdir) {
  out <- reticulate::py_to_r(py_do(x$py$write(path.expand(outdir))))
  invisible(vapply(out, as.character, character(1)))
}

#' Combine per-dataset results for a benchmark across studies
#'
#' Datasets are scored separately and combined afterwards, never merged before
#' scoring. A 27K study and an EPIC study have different probe spaces, and
#' feature coverage is a property of a dataset -- merged, a clock that covers
#' 99% of one study and 40% of another reports a single meaningless average, and
#' the AA2 test compares cases in the well-covered study against controls in the
#' badly-covered one.
#'
#' @param results A list of `falcon_result` objects.
#' @param keys Optional dataset names, one per result.
#' @return A `falcon_result`.
#' @examples
#' \dontrun{
#' res <- combine(list(r1, r2, r3), keys = c("GSE1", "GSE2", "GSE3"))
#' }
#' @export
combine <- function(results, keys = NULL) {
  new_falcon_result(py_do(fa()$combine(
    reticulate::r_to_py(lapply(results, `[[`, "py")),
    keys = if (is.null(keys)) reticulate::py_none() else reticulate::r_to_py(as.list(keys)))))
}

#' Write a self-contained HTML report
#'
#' One file: inlined CSS, base64 figures, the tables embedded. A report that
#' references `figures/ba_vs_ca.png` stops working the moment somebody emails
#' it, which is the only thing anybody does with a report.
#'
#' @param x A `falcon_result`.
#' @param path Destination `.html`.
#' @param age_col Column in the sample annotation holding chronological age.
#' @param group Optional grouping column for the figures.
#' @param title Page title.
#' @return The path, invisibly.
#' @examples
#' \dontrun{
#' report(res, "report.html", group = "condition")
#' }
#' @export
report <- function(x, path, age_col = "age", group = NULL,
                   title = "FALCONAge report") {
  rp <- reticulate::import("falconage.report", convert = FALSE)
  py_do(rp$write_report(x$py, path.expand(path), age_col = age_col,
                        group = or_none(group), title = title))
  invisible(path)
}

#' Run the whole analysis and write every deliverable to a directory
#'
#' Quality control, scoring, technical standard error, conformal intervals, age
#' acceleration, a consensus test when `group` names a column, interpretation
#' and evidence tables, figures and the one-page report: the directory
#' `falconage report` writes. With `quarto = TRUE` it also writes
#' `falconage_report.qmd`, with every file placed under the step of the
#' analysis that produced it, and renders it to one self-contained
#' `falconage_report.html` when Quarto is on the path.
#'
#' @param data A `falcon_data`, already prepared.
#' @param outdir Output directory.
#' @param clocks `"compatible"`, `"all"`, or a character vector of clock ids.
#' @param group Optional column of the sample annotation to compare groups by.
#' @param level Coverage of the conformal intervals.
#' @param min_coverage Fraction of a clock's features that must be present.
#' @param figures Draw the figures.
#' @param quarto Also write the step-ordered Quarto report.
#' @param render With `quarto = TRUE`, render it when Quarto is available.
#' @return The paths of the main files, as a named character vector, invisibly.
#' @examples
#' \dontrun{
#' run_report(d, "results", group = "condition", quarto = TRUE)
#' }
#' @export
run_report <- function(data, outdir, clocks = "compatible", group = NULL,
                       level = 0.90, min_coverage = 0.8, figures = TRUE,
                       quarto = FALSE, render = TRUE) {
  rp <- reticulate::import("falconage.report", convert = FALSE)
  out <- py_do(rp$run_report(
    data$py, path.expand(outdir),
    clocks = if (length(clocks) == 1L && clocks %in% c("compatible", "all"))
      clocks else reticulate::r_to_py(as.list(clocks)),
    group_col = or_none(group), level = level, min_coverage = min_coverage,
    figures = figures, quarto = quarto, render = render))
  paths <- reticulate::py_to_r(out)
  invisible(vapply(paths, function(p) if (is.character(p)) p else
    reticulate::py_str(p), character(1)))
}

#' Write the step-ordered Quarto report for an output directory
#'
#' Every file in `outdir` is placed under the step of the analysis that
#' produced it, in step order, with its description; a file the report does
#' not recognise is listed in the last step, so nothing is left out. Rendering
#' embeds every figure and table, so the HTML is one file that references
#' nothing. Without Quarto on the path, the `.qmd` is written and the error
#' names the command that renders it.
#'
#' @param outdir A directory written by [run_report()] or `falconage report`.
#' @param x Optional `falcon_result`, to add what each scored category of clock
#'   means to the scores step.
#' @param title Page title.
#' @param render Render to HTML as well as writing the source.
#' @return The path of the `.html` (or the `.qmd` without rendering), invisibly.
#' @examples
#' \dontrun{
#' quarto_report("results")
#' }
#' @export
quarto_report <- function(outdir, x = NULL, title = "FALCONAge report",
                          render = TRUE) {
  rp <- reticulate::import("falconage.report", convert = FALSE)
  p <- py_do(rp$write_quarto_report(path.expand(outdir),
                                    if (is.null(x)) reticulate::py_none() else x$py,
                                    title = title, render = render))
  invisible(reticulate::py_str(p))
}
