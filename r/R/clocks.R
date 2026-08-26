# =============================================================================
# Browsing the registry
# =============================================================================

#' List the clock catalogue
#'
#' @param tier `"bundled"`, `"untraced"` or `"licensed"`, or `NULL` for all.
#'   The retired `"A"`, `"B"`, `"C"` are still accepted. Bundled ships with
#'   coefficients and runs offline; untraced is catalogued but has no traced
#'   primary source yet; licensed is implemented but its coefficients are
#'   research-use-only.
#' @param data_type `"dna_methylation"` or `"clinical_chemistry"`.
#' @param generation `"first"`, `"second"`, `"pace"`, `"causal"`, `"mitotic"`,
#'   `"system"` or `"other"`.
#' @param untraced Only clocks with no established primary source.
#' @param search Substring match over id, name, what it predicts, and citation.
#'
#' @return A data frame, one row per clock.
#' @examples
#' \dontrun{
#' list_clocks(tier = "bundled")
#' list_clocks(search = "mortality")
#' list_clocks(tier = "licensed")   # the ones needing author permission
#' }
#' @export
list_clocks <- function(tier = NULL, data_type = NULL, generation = NULL,
                        untraced = FALSE, search = NULL) {
  reg <- py_do(fa()$registry$load())
  df <- as_df(reg$summary())
  if (!is.null(tier)) {
    # The letters are what every script written before the rename says, so
    # they are translated rather than silently matching nothing.
    tier <- switch(as.character(tier), A = "bundled", B = "untraced",
                   C = "licensed", tier)
    df <- df[df$availability == tier, , drop = FALSE]
  }
  if (!is.null(data_type))  df <- df[df$data_type == data_type, , drop = FALSE]
  if (!is.null(generation)) df <- df[df$generation == generation, , drop = FALSE]
  if (isTRUE(untraced))     df <- df[!df$traced, , drop = FALSE]
  if (!is.null(search)) {
    hits <- vapply(reticulate::py_to_r(reg$search(search)),
                   function(c) reticulate::py_to_r(c$id), character(1))
    df <- df[rownames(df) %in% hits, , drop = FALSE]
  }
  df
}

#' Everything the registry knows about one clock
#'
#' For a licensed clock this also prints why its coefficients are not distributed,
#' where to obtain them, and which open clocks answer the same question.
#'
#' @param clock_id A clock identifier.
#' @return A named list, invisibly. Printed for reading.
#' @examples
#' \dontrun{
#' clock_info("horvath2013")
#' clock_info("grimage2")
#' }
#' @export
clock_info <- function(clock_id) {
  reg <- py_do(fa()$registry$load())
  c <- py_do(reg$get(clock_id))
  cs <- c$coefficient_source
  bi <- reticulate::import_builtins(convert = FALSE)
  g <- function(x) reticulate::py_to_r(x)
  # legal_operations is a Python set, and a set proxy has no ordering for R to
  # sort by; sorted() on the Python side turns it into a list first.
  gset <- function(x) as.character(unlist_na(g(bi$sorted(x))))

  out <- list(
    id = g(c$id), name = g(c$name), year = g(c$year), species = g(c$species),
    data_type = g(c$data_type), generation = g(c$generation),
    predicts = unlist(g(c$predicts)), unit = unlist(g(c$unit)),
    scale_type = g(c$scale_type), legal_operations = gset(c$legal_operations),
    platform = unlist(g(c$platform)), tissue = unlist(g(c$tissue)),
    n_features = g(c$n_features), availability = g(c$availability),
    provenance = g(cs$provenance), primary_source_traced = g(cs$primary_source_traced),
    citation = g(c$citation), doi = g(c$doi), notes = g(c$notes))

  cat(out$id, " -- ", out$name, "\n", sep = "")
  cat("  year         ", out$year, "\n")
  cat("  species      ", out$species, "\n")
  cat("  predicts     ", paste(out$predicts, collapse = ", "),
      " (", paste(out$unit, collapse = ", "), ")\n", sep = "")
  cat("  scale type   ", out$scale_type, "\n")
  cat("  legal ops    ", paste(out$legal_operations, collapse = ", "), "\n")
  cat("  platform     ", paste(out$platform, collapse = ", "), "\n")
  cat("  features     ", out$n_features %||% "unknown", "\n")
  cat("  availability ", out$availability, "\n", sep = "")
  cat("  provenance   ", out$provenance, "\n")
  cat("  traced       ", out$primary_source_traced, "\n")
  if (identical(out$availability, "licensed")) {
    cat("\n", reticulate::py_to_r(reg$unavailable_message(clock_id)), "\n", sep = "")
  }
  if (nzchar(out$notes)) cat("\n  ", out$notes, "\n", sep = "")
  cat("\n  ", out$citation, " ", out$doi, "\n", sep = "")
  invisible(out)
}

#' Cite a clock
#'
#' @param clock_id A clock identifier.
#' @param style `"plain"` or `"bibtex"`.
#' @return A character string.
#' @examples
#' \dontrun{
#' cite_clock("horvath2013")
#' cat(cite_clock("dnamphenoage", "bibtex"))
#' }
#' @export
cite_clock <- function(clock_id, style = c("plain", "bibtex")) {
  reg <- py_do(fa()$registry$load())
  reticulate::py_to_r(reg$get(clock_id)$cite(match.arg(style)))
}

#' Supply a coefficient file for a clock FALCONAge does not distribute
#'
#' Twenty-eight clocks ship as scaffolds: the model, the feature list, the
#' preprocess and postprocess chain, the expected shapes -- everything except
#' the numbers, which are research-use-only. Once you hold a licensed file, this
#' registers it.
#'
#' Registration validates the file against the scaffold and rejects a mismatch
#' with the discrepancy named, which also makes it a way to check a coefficient
#' set somebody handed you. The digest goes into the run manifest as
#' `user_supplied`, so a result computed from a licensed copy is distinguishable
#' from one computed from a redistributed set.
#'
#' @param clock_id A clock identifier, e.g. `"grimage2"`.
#' @param path A CSV with `feature_id,coefficient` columns.
#' @param sha256 Optional expected digest; a mismatch is an error.
#' @return The file's SHA-256, invisibly.
#' @examples
#' \dontrun{
#' register_local_weights("grimage2", "~/licensed/grimage2_coefs.csv")
#' score(d, clocks = "grimage2")
#' }
#' @export
register_local_weights <- function(clock_id, path, sha256 = NULL) {
  reg <- py_do(fa()$registry$load())
  invisible(reticulate::py_to_r(py_do(
    reg$register_local_weights(clock_id, path.expand(path), or_none(sha256)))))
}

#' Which clocks this dataset can actually be scored on
#'
#' Compatibility is coverage, not platform. A clock trained on 450K runs
#' perfectly well on EPIC data that carries its probes, and fails on 450K data
#' filtered down to 20,000 probes -- only the feature list can answer it.
#'
#' @param data A `falcon_data`.
#' @param min_coverage Fraction of a clock's features that must be present.
#' @return A character vector of clock ids.
#' @examples
#' \dontrun{
#' compatible_clocks(d)
#' }
#' @export
compatible_clocks <- function(data, min_coverage = 0.8) {
  reg <- py_do(fa()$registry$load())
  cs <- reticulate::py_to_r(reg$compatible_with(data$py, min_coverage = min_coverage))
  vapply(cs, function(c) reticulate::py_to_r(c$id), character(1))
}


#' What each clock has lost on this dataset, before scoring
#'
#' One row per clock: how many of its features are present, and -- where the
#' coefficients are available -- how much of the model's total weight those
#' present features carry, plus the heaviest probes that are missing.
#'
#' @section Why both numbers:
#' A count treats every probe as interchangeable, and an elastic net's weights
#' are nothing like uniform. "92% of probes present" covers both "the missing
#' 8% are negligible" and "the missing 8% carry a third of the model". EPIC v2
#' dropped probes that several first-generation clocks lean on, which is why
#' those clocks shift on v2 arrays while the principal-component versions
#' barely move -- the same probe loss, very different consequences.
#'
#' Run this on an array you have not used before, before `score()`. It answers
#' "will this dataset support these clocks" without producing a number anyone
#' can quote.
#'
#' @param x A `falcon_data`.
#' @param clocks `"all"`, `"scoreable"` for the ones whose coefficients are
#'   available, or a character vector of clock names.
#' @param top How many of the heaviest absent features to name per clock.
#' @return A data frame, worst mass coverage first. `mass_coverage` is `NA` for
#'   a clock whose coefficients are not available -- the weights are what the
#'   column is computed from.
#' @examples
#' \dontrun{
#' probe_loss(d, clocks = "scoreable")
#' }
#' @export
probe_loss <- function(x, clocks = "all", top = 3L) {
  cl <- if (length(clocks) == 1L && clocks %in% c("all", "scoreable")) {
    clocks
  } else {
    reticulate::r_to_py(as.list(clocks))
  }
  as_df(py_do(fa()$probe_loss(x$py, clocks = cl, top = as.integer(top))))
}
