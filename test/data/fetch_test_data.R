#!/usr/bin/env Rscript
# =============================================================================
# Fetch the FALCONAge public test corpus described in datasets.yaml
# =============================================================================
#
# The R half of a matched pair. fetch_test_data.py is the same program, reads
# the same manifest, and must produce a BYTE-IDENTICAL checksums.sha256 -- that
# equality is the corpus-level form of the R/Python conformance gate in
# the design notes, and Dockerfile.testdata asserts it at build.
#
# WHY THIS IS NOT A RETICULATE WRAPPER, WHEN THE PACKAGE ITSELF IS. FALCONAge
# proper wraps one Python numerical core from R, because bit-identical clock
# scores are worth more than language independence. This script is the opposite
# case: it exists to produce the fixtures the package is tested against, so it
# must not depend on the package, on Python, or on the bridge between them. An
# R user with no Python gets the corpus by running this file.
#
# Dependencies: yaml, jsonlite, curl, digest. All four are on CRAN, all four
# install as binaries, none needs the package.
#
# Usage:
#   Rscript fetch_test_data.R --dry-run
#   Rscript fetch_test_data.R
#   Rscript fetch_test_data.R --groups bench,mouse
#   Rscript fetch_test_data.R --verify
#   Rscript fetch_test_data.R --self-test
# =============================================================================

suppressPackageStartupMessages({
  library(yaml)
  library(jsonlite)
  library(curl)
  library(digest)
})

VERSION <- "1.0.0"
USER_AGENT <- paste0("FALCONAge-testdata/", VERSION,
                     " (+https://github.com/bhagesh-h/FALCONAge)")

# -----------------------------------------------------------------------------
# arguments
# -----------------------------------------------------------------------------
# Hand-parsed rather than optparse'd. Four packages is already the whole
# dependency budget for a script whose job is to work when nothing else does,
# and the flag set is small enough that a fifth package would cost more than it
# saves. The flags mirror fetch_test_data.py exactly, including the defaults.
parse_args <- function(argv) {
  opts <- list(
    manifest   = NULL,
    out        = NULL,
    groups     = "default",
    max_bytes  = NA_real_,
    dry_run    = FALSE,
    verify     = FALSE,
    self_test  = FALSE,
    force      = FALSE,
    quiet      = FALSE
  )
  i <- 1L
  take <- function() {
    if (i >= length(argv)) stop("missing value after ", argv[[i]], call. = FALSE)
    argv[[i + 1L]]
  }
  while (i <= length(argv)) {
    a <- argv[[i]]
    switch(
      a,
      "--manifest"  = { opts$manifest <- take();            i <- i + 2L },
      "--out"       = { opts$out <- take();                 i <- i + 2L },
      "--groups"    = { opts$groups <- take();              i <- i + 2L },
      "--max-bytes" = { opts$max_bytes <- as.numeric(take()); i <- i + 2L },
      "--dry-run"   = { opts$dry_run <- TRUE;               i <- i + 1L },
      "--verify"    = { opts$verify <- TRUE;                i <- i + 1L },
      "--self-test" = { opts$self_test <- TRUE;             i <- i + 1L },
      "--force"     = { opts$force <- TRUE;                 i <- i + 1L },
      "--quiet"     = { opts$quiet <- TRUE;                 i <- i + 1L },
      "--version"   = { cat("fetch_test_data.R", VERSION, "\n"); quit(status = 0L) },
      {
        if (a %in% c("-h", "--help")) { usage(); quit(status = 0L) }
        stop("unknown argument: ", a, call. = FALSE)
      }
    )
  }
  opts
}

usage <- function() {
  cat(
    "Fetch the FALCONAge public test corpus.\n\n",
    "  --manifest PATH   datasets.yaml (default: alongside this script)\n",
    "  --out DIR         destination (default: alongside the manifest)\n",
    "  --groups SPEC     'default', 'all', or a comma-separated list of ids\n",
    "  --max-bytes N     override the ceiling declared in the manifest\n",
    "  --dry-run         show the plan and stop\n",
    "  --verify          re-check files already on disk\n",
    "  --self-test       check the manifest, touch nothing\n",
    "  --force           refetch files that are already present\n",
    "  --quiet           no per-file progress\n",
    sep = ""
  )
}

script_dir <- function() {
  ca <- commandArgs(trailingOnly = FALSE)
  m <- grep("^--file=", ca, value = TRUE)
  if (length(m)) return(dirname(normalizePath(sub("^--file=", "", m[[1]]))))
  getwd()
}

# -----------------------------------------------------------------------------
# manifest
# -----------------------------------------------------------------------------
load_manifest <- function(path) {
  raw <- readBin(path, "raw", file.info(path)$size)
  doc <- yaml::yaml.load(rawToChar(raw))

  if (!identical(as.integer(doc$schema_version), 1L)) {
    stop(path, ": schema_version is not 1; this fetcher only understands schema 1",
         call. = FALSE)
  }

  entries <- list()
  for (g in doc$groups) {
    for (f in g$files) {
      entries[[length(entries) + 1L]] <- list(
        path   = f$path,
        url    = f$url,
        bytes  = as.numeric(f$bytes),
        # `sha256` is omitted, never null, when the publisher gives us none --
        # so that PyYAML and this reader agree on what absence looks like.
        sha256 = if (is.null(f$sha256)) NA_character_ else as.character(f$sha256),
        source = f$source,
        group  = g$id,
        note   = if (is.null(f$note)) "" else f$note
      )
    }
  }

  list(
    schema_version        = as.integer(doc$schema_version),
    budget_bytes          = as.numeric(doc$budget_bytes),
    expected_total_bytes  = as.numeric(doc$expected_total_bytes),
    expected_total_files  = as.integer(doc$expected_total_files),
    sources               = doc$sources,
    groups                = doc$groups,
    entries               = entries,
    # Digest of the file, not of the parse: two implementations that parse
    # differently still agree here, which is what makes it a provenance field
    # rather than a checksum of an opinion.
    digest                = digest::digest(raw, algo = "sha256", serialize = FALSE)
  )
}

group_ids <- function(m) vapply(m$groups, function(g) g$id, character(1))

default_group_ids <- function(m) {
  keep <- vapply(m$groups, function(g) isTRUE(g$default), logical(1))
  group_ids(m)[keep]
}

select_entries <- function(m, spec) {
  all_ids <- group_ids(m)
  wanted <- if (identical(spec, "default")) {
    default_group_ids(m)
  } else if (identical(spec, "all")) {
    all_ids
  } else {
    w <- trimws(strsplit(spec, ",", fixed = TRUE)[[1]])
    w <- w[nzchar(w)]
    unknown <- setdiff(w, all_ids)
    if (length(unknown)) {
      stop("unknown group(s): ", paste(unknown, collapse = ", "),
           "\navailable: ", paste(all_ids, collapse = ", "), call. = FALSE)
    }
    w
  }
  Filter(function(e) e$group %in% wanted, m$entries)
}

# -----------------------------------------------------------------------------
# hashing and formatting
# -----------------------------------------------------------------------------
sha256_file <- function(path) digest::digest(path, algo = "sha256", file = TRUE)

human <- function(n) {
  units <- c("B", "kB", "MB", "GB")
  i <- 1L
  while (abs(n) >= 1000 && i < length(units)) { n <- n / 1000; i <- i + 1L }
  if (i == 1L) sprintf("%.0f B", n) else sprintf("%.1f %s", n, units[[i]])
}

# -----------------------------------------------------------------------------
# transfer
# -----------------------------------------------------------------------------
# curl::multi_download does resumption, retries and progress natively, which is
# the whole reason the `curl` package is a dependency: base R's download.file()
# cannot resume a partial transfer portably, and a 94 MB parquet restarting from
# zero on a dropped connection is how a five-minute fetch becomes an hour.
download_entry <- function(e, dest, quiet = FALSE) {
  part <- paste0(dest, ".part")
  dir.create(dirname(dest), recursive = TRUE, showWarnings = FALSE)

  # The ... of multi_download are handle OPTIONS, not a handle. Passing a
  # built handle here fails with "Unknown option: handle", which looks like a
  # network error in the log and is not one.
  res <- curl::multi_download(
    urls      = e$url,
    destfiles = part,
    resume    = TRUE,
    progress  = !quiet,
    timeout   = 3600,
    useragent = USER_AGENT
  )

  # success can be TRUE for a 404: curl succeeded in transferring the server's
  # error page. The status code has to be checked separately or a 9 kB "not
  # found" document lands on disk under the name of a 94 MB parquet.
  status <- res$status_code[[1]]
  if (!isTRUE(res$success[[1]]) || is.na(status) || status < 200 || status >= 300) {
    unlink(part)
    stop(e$url, ": ",
         if (!is.na(res$error[[1]])) res$error[[1]] else paste("HTTP", status),
         call. = FALSE)
  }

  actual <- file.info(part)$size
  dg <- sha256_file(part)

  if (!is.na(e$sha256) && !identical(dg, e$sha256)) {
    unlink(part)
    stop(e$path, ": SHA-256 mismatch\n",
         "  expected ", e$sha256, "\n",
         "  got      ", dg, "\n",
         "  The publisher gives a digest for this file, so this is a hard error: ",
         "the bytes are not the bytes the manifest describes.", call. = FALSE)
  }

  if (!identical(as.numeric(actual), e$bytes)) {
    # A warning, not an error. GEO publishes no digests and a submitter can
    # replace a supplementary file in place; the honest response is to say the
    # manifest is stale, not to refuse to work.
    message("    note: ", e$path, " is ", human(actual), ", manifest says ",
            human(e$bytes), " -- the source has changed since 2026-08-07")
  }

  # Rename only after the checks pass, so a file that exists at its final name
  # is a file that was verified. There is no state in which a truncated
  # download looks complete.
  if (!file.rename(part, dest)) {
    file.copy(part, dest, overwrite = TRUE)
    unlink(part)
  }
  invisible(dg)
}

# -----------------------------------------------------------------------------
# outputs
# -----------------------------------------------------------------------------
# WHY method = "radix" AND A BINARY CONNECTION. This file is the conformance
# artefact and must match the Python one byte for byte. R's default sort() is
# locale-collated (in some locales "GSE1" and "gse1" interleave); radix is C
# byte order, which is what Python's sorted() does on ASCII. And a text
# connection on Windows silently rewrites "\n" as "\r\n", which would make the
# two implementations differ in every single line.
write_checksums <- function(out, records) {
  paths <- sort(names(records), method = "radix")
  text <- paste0(unlist(records[paths]), "  ", paths, "\n", collapse = "")
  con <- file(file.path(out, "checksums.sha256"), open = "wb")
  on.exit(close(con))
  writeBin(charToRaw(text), con)
  invisible(NULL)
}

read_checksums <- function(out) {
  f <- file.path(out, "checksums.sha256")
  if (!file.exists(f)) return(list())
  lines <- readLines(f, warn = FALSE)
  lines <- lines[nzchar(trimws(lines))]
  records <- list()
  for (l in lines) {
    parts <- strsplit(l, "  ", fixed = TRUE)[[1]]
    records[[paste(parts[-1], collapse = "  ")]] <- parts[[1]]
  }
  records
}

# Informational, unlike checksums.sha256: the two implementations may differ in
# whitespace here and that is fine. No timestamp, deliberately -- a provenance
# file that changes on every run cannot be diffed against the last one, which is
# the only thing anybody ever wants to do with it.
write_provenance <- function(out, m, entries, records) {
  entries <- entries[order(vapply(entries, function(e) e$path, character(1)),
                           method = "radix")]
  files <- lapply(
    Filter(function(e) !is.null(records[[e$path]]), entries),
    function(e) list(
      bytes_expected     = e$bytes,
      checksum_authority = if (is.na(e$sha256)) "trust-on-first-use" else "publisher",
      group              = e$group,
      path               = e$path,
      sha256_expected    = if (is.na(e$sha256)) NULL else e$sha256,
      sha256_observed    = records[[e$path]],
      source             = e$source,
      url                = e$url
    )
  )
  src_ids <- sort(names(m$sources), method = "radix")
  sources <- lapply(src_ids, function(k) {
    s <- m$sources[[k]]
    list(citation = s$citation, homepage = s$homepage,
         licence = s$licence, name = s$name)
  })
  names(sources) <- src_ids

  doc <- list(
    files           = files,
    manifest_sha256 = m$digest,
    schema_version  = m$schema_version,
    sources         = sources
  )
  # sub() rather than paste0(): jsonlite's pretty printer already ends with a
  # newline in some versions and not others. Normalising to exactly one is what
  # makes this file byte-identical to the Python one rather than nearly so.
  text <- sub("\n*$", "\n", jsonlite::toJSON(doc, pretty = 2, auto_unbox = TRUE,
                                             null = "null", digits = NA))
  con <- file(file.path(out, "provenance.json"), open = "wb")
  on.exit(close(con))
  writeBin(charToRaw(text), con)
  invisible(NULL)
}

# -----------------------------------------------------------------------------
# commands
# -----------------------------------------------------------------------------
cmd_plan <- function(m, entries, out, max_bytes) {
  gids <- unique(vapply(entries, function(e) e$group, character(1)))
  have <- vapply(entries, function(e) file.exists(file.path(out, e$path)), logical(1))
  selected <- sum(vapply(entries, function(e) e$bytes, numeric(1)))
  remaining <- sum(vapply(entries[!have], function(e) e$bytes, numeric(1)))

  cat(sprintf("manifest    %s...  schema %d\n", substr(m$digest, 1, 16), m$schema_version))
  cat(sprintf("destination %s\n\n", out))
  cat(sprintf("%-14s%6s%12s   %s\n", "group", "files", "size", "what it is for"))
  cat(strrep("-", 78), "\n", sep = "")
  for (gid in gids) {
    es <- Filter(function(e) e$group == gid, entries)
    meta <- Filter(function(g) g$id == gid, m$groups)[[1]]
    cat(sprintf("%-14s%6d%12s   %s\n", gid, length(es),
                human(sum(vapply(es, function(e) e$bytes, numeric(1)))), meta$title))
  }
  cat(strrep("-", 78), "\n", sep = "")
  cat(sprintf("%-14s%6d%12s\n", "selected", length(entries), human(selected)))
  if (any(have)) {
    cat(sprintf("%-14s%6d%12s\n", "already here", sum(have), human(selected - remaining)))
    cat(sprintf("%-14s%6d%12s\n", "to download", sum(!have), human(remaining)))
  }
  cat(sprintf("%-14s%6s%12s   %.0f%% used\n", "ceiling", "", human(max_bytes),
              100 * selected / max_bytes))

  if (selected > max_bytes) {
    cat(sprintf("\nREFUSED: the selection is %s, the ceiling is %s.\n",
                human(selected), human(max_bytes)))
    cat(sprintf("         Over by %s. Narrow --groups, or raise --max-bytes if you mean it.\n",
                human(selected - max_bytes)))
    return(1L)
  }
  0L
}

cmd_fetch <- function(m, entries, out, max_bytes, force, quiet) {
  total <- sum(vapply(entries, function(e) e$bytes, numeric(1)))
  if (total > max_bytes) {
    message(sprintf("REFUSED: selection is %s, ceiling is %s", human(total), human(max_bytes)))
    return(1L)
  }

  dir.create(out, recursive = TRUE, showWarnings = FALSE)
  records <- read_checksums(out)
  fetched <- 0L; skipped <- 0L

  for (i in seq_along(entries)) {
    e <- entries[[i]]
    dest <- file.path(out, e$path)

    if (file.exists(dest) && !force) {
      # Present is not the same as correct: a half-written file from an
      # interrupted earlier run must not be silently accepted as done.
      if (!is.na(e$sha256)) {
        dg <- if (!is.null(records[[e$path]])) records[[e$path]] else sha256_file(dest)
        if (!identical(dg, e$sha256)) {
          cat(sprintf("[%d/%d] %s: on disk but wrong digest, refetching\n",
                      i, length(entries), e$path))
          unlink(dest)
        } else {
          records[[e$path]] <- dg; skipped <- skipped + 1L; next
        }
      } else {
        if (is.null(records[[e$path]])) records[[e$path]] <- sha256_file(dest)
        skipped <- skipped + 1L; next
      }
    }

    cat(sprintf("[%d/%d] %s  (%s, %s)\n", i, length(entries), e$path,
                human(e$bytes), e$source))
    download_entry(e, dest, quiet = quiet)
    records[[e$path]] <- sha256_file(dest)
    fetched <- fetched + 1L
  }

  write_checksums(out, records)
  write_provenance(out, m, entries, records)

  present <- Filter(function(e) file.exists(file.path(out, e$path)), entries)
  on_disk <- sum(vapply(present, function(e) file.info(file.path(out, e$path))$size, numeric(1)))
  cat(sprintf("\nfetched %d, already present %d, %s in %s\n",
              fetched, skipped, human(on_disk), out))
  cat(sprintf("wrote %s and %s\n", file.path(out, "checksums.sha256"),
              file.path(out, "provenance.json")))
  0L
}

cmd_verify <- function(m, entries, out) {
  recorded <- read_checksums(out)
  if (!length(recorded)) {
    message("no checksums.sha256 in ", out, " -- nothing has been fetched here")
    return(1L)
  }
  missing <- character(0); bad <- character(0); ok <- 0L

  for (e in entries) {
    dest <- file.path(out, e$path)
    if (!file.exists(dest)) { missing <- c(missing, e$path); next }
    dg <- sha256_file(dest)

    # Published digest beats recorded digest. A recorded one only says the file
    # has not changed since it was fetched, which is worth much less if what was
    # fetched was wrong.
    expected <- if (!is.na(e$sha256)) e$sha256 else recorded[[e$path]]
    if (is.null(expected)) {
      bad <- c(bad, paste0(e$path, ": no digest to check against"))
    } else if (!identical(dg, expected)) {
      authority <- if (!is.na(e$sha256)) "publisher" else "first fetch"
      bad <- c(bad, paste0(e$path, ": differs from the ", authority, " digest"))
    } else ok <- ok + 1L
  }

  cat(sprintf("verified %d/%d files against %s\n", ok, length(entries),
              file.path(out, "checksums.sha256")))
  for (x in missing) cat("  MISSING  ", x, "\n", sep = "")
  for (x in bad)     cat("  BAD      ", x, "\n", sep = "")
  if (!length(missing) && !length(bad)) 0L else 1L
}

cmd_self_test <- function(m) {
  problems <- character(0)
  n <- length(m$entries)
  total <- sum(vapply(m$entries, function(e) e$bytes, numeric(1)))

  if (n != m$expected_total_files) {
    problems <- c(problems, sprintf("file count is %d, manifest declares %d",
                                    n, m$expected_total_files))
  }
  if (total != m$expected_total_bytes) {
    problems <- c(problems, sprintf("byte total is %.0f, manifest declares %.0f (off by %.0f)",
                                    total, m$expected_total_bytes,
                                    total - m$expected_total_bytes))
  }
  if (total > m$budget_bytes) {
    problems <- c(problems, sprintf("whole corpus is %s, over the %s ceiling",
                                    human(total), human(m$budget_bytes)))
  }

  seen <- character(0)
  for (e in m$entries) {
    if (e$path %in% seen) problems <- c(problems, paste0("duplicate destination path: ", e$path))
    seen <- c(seen, e$path)
    if (!(e$source %in% names(m$sources))) {
      problems <- c(problems, sprintf("%s: source '%s' is not declared under sources:",
                                      e$path, e$source))
    }
    if (!startsWith(e$url, "https://")) {
      problems <- c(problems, paste0(e$path, ": url is not https"))
    }
    if (!is.na(e$sha256) && nchar(e$sha256) != 64L) {
      problems <- c(problems, sprintf("%s: sha256 is %d characters, not 64",
                                      e$path, nchar(e$sha256)))
    }
    if (grepl("^([A-Za-z]:)?[/\\\\]", e$path) || ".." %in% strsplit(e$path, "/")[[1]]) {
      problems <- c(problems, paste0(e$path, ": destination escapes the output directory"))
    }
  }

  for (g in m$groups) {
    if (!is.null(g$bytes)) {
      actual <- sum(vapply(Filter(function(e) e$group == g$id, m$entries),
                           function(e) e$bytes, numeric(1)))
      if (as.numeric(g$bytes) != actual) {
        problems <- c(problems, sprintf("group %s: declares %.0f bytes, files sum to %.0f",
                                        g$id, as.numeric(g$bytes), actual))
      }
    }
  }

  with_digest <- sum(vapply(m$entries, function(e) !is.na(e$sha256), logical(1)))
  cat(sprintf("manifest %s...\n", substr(m$digest, 1, 16)))
  cat(sprintf("  %d groups, %d files, %s (%.0f%% of the ceiling)\n",
              length(m$groups), n, human(total), 100 * total / m$budget_bytes))
  cat(sprintf("  %d files carry a publisher digest, %d are trust-on-first-use\n",
              with_digest, n - with_digest))

  if (length(problems)) {
    cat("\n")
    for (p in problems) cat("  PROBLEM  ", p, "\n", sep = "")
    return(1L)
  }
  cat("  self-test passed\n")
  0L
}

# -----------------------------------------------------------------------------
main <- function(argv = commandArgs(trailingOnly = TRUE)) {
  here <- script_dir()
  opts <- parse_args(argv)
  if (is.null(opts$manifest)) opts$manifest <- file.path(here, "datasets.yaml")
  if (!file.exists(opts$manifest)) stop("manifest not found: ", opts$manifest, call. = FALSE)
  if (is.null(opts$out)) opts$out <- dirname(normalizePath(opts$manifest))

  m <- load_manifest(opts$manifest)
  if (opts$self_test) return(cmd_self_test(m))

  entries <- select_entries(m, opts$groups)
  if (!length(entries)) stop("--groups ", opts$groups, " selected nothing", call. = FALSE)

  dir.create(opts$out, recursive = TRUE, showWarnings = FALSE)
  out <- normalizePath(opts$out, winslash = "/", mustWork = TRUE)
  max_bytes <- if (is.na(opts$max_bytes)) m$budget_bytes else opts$max_bytes

  if (opts$verify)  return(cmd_verify(m, entries, out))
  if (opts$dry_run) return(cmd_plan(m, entries, out, max_bytes))

  rc <- cmd_plan(m, entries, out, max_bytes)
  if (rc != 0L) return(rc)
  cat("\n")
  cmd_fetch(m, entries, out, max_bytes, opts$force, opts$quiet)
}

if (sys.nframe() == 0L || identical(environment(), globalenv())) {
  status <- tryCatch(main(), error = function(e) {
    message("\nerror: ", conditionMessage(e))
    1L
  })
  quit(status = as.integer(status))
}
