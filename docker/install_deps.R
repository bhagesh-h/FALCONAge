#!/usr/bin/env Rscript
# =============================================================================
# Install everything FALCONAge declares, from the declaration itself
# =============================================================================
#
# WHY IT PARSES DESCRIPTION INSTEAD OF CARRYING ITS OWN LIST: a second list is a
# second thing to forget. If this file named the packages directly, adding an
# Import would build an image that installs one set and loads another, and the
# failure would surface at run time, inside the container, in the middle of an
# analysis. Reading Depends/Imports/Suggests out of DESCRIPTION makes that
# impossible by construction -- there is one declaration and this script obeys
# it.
#
# WHY BiocManager AND NOT install.packages(): a methylation package reaches
# Bioconductor sooner or later (minfi, sesame, ExperimentHub for the manifests),
# and install.packages() cannot see it. BiocManager::install() resolves CRAN and
# Bioconductor together and honours the pinned BIOC_VERSION the image sets, so a
# rebuild months from now gets the same release rather than whatever is current.
#
# Usage:
#   Rscript install_deps.R                    # DESCRIPTION in the working dir
#   Rscript install_deps.R path/to/DESCRIPTION
#
# Environment:
#   R_REPOS       CRAN mirror; the images point it at a dated Posit snapshot so
#                 the resolution is fixed and the packages arrive as binaries
#   BIOC_VERSION  Bioconductor release to pin
# =============================================================================

args <- commandArgs(trailingOnly = TRUE)
desc_path <- if (length(args)) args[[1]] else "DESCRIPTION"
if (!file.exists(desc_path) && file.exists(file.path(desc_path, "DESCRIPTION"))) {
  desc_path <- file.path(desc_path, "DESCRIPTION")
}
if (!file.exists(desc_path)) {
  stop("DESCRIPTION not found at: ", desc_path, call. = FALSE)
}

repos <- Sys.getenv("R_REPOS", "https://cloud.r-project.org")

# Posit's package manager serves precompiled Linux binaries, but only to a
# client whose User-Agent says which distribution and R version it is; to
# anything else it serves source, and R's default agent says neither. Without
# this line the "binary snapshot" in the Dockerfiles is a source snapshot: the
# R layer takes half an hour instead of two minutes and fails on the first
# package needing a -dev header nobody installed, which is how `fs` brought
# down testthat and rmarkdown with it.
options(HTTPUserAgent = sprintf(
  "R/%s R (%s)", getRversion(),
  paste(getRversion(), R.version["platform"], R.version["arch"], R.version["os"])))

options(
  repos = c(CRAN = repos),
  # A warning that scrolls past is a package that is missing at run time. The
  # explicit check at the bottom is the real guard, but failing loudly on the
  # way there makes the build log readable.
  warn = 1,
  Ncpus = max(1L, parallel::detectCores(logical = FALSE))
)

dcf <- read.dcf(desc_path)

# Suggests is installed too, deliberately. Those packages are optional to the
# PACKAGE but not to this image: without the CLI option parser there is no
# command line, without knitr the vignettes do not build, and without testthat
# the build-time self-check cannot run. An image that can only exercise half the
# package is not reproducible in any useful sense.
fields <- c("Depends", "Imports", "LinkingTo", "Suggests")
raw <- unlist(lapply(intersect(fields, colnames(dcf)), function(f) dcf[1L, f]))

pkgs <- unlist(strsplit(paste(raw, collapse = ","), ","))
pkgs <- trimws(sub("\\(.*", "", pkgs))                 # drop version constraints
pkgs <- pkgs[nzchar(pkgs)]
pkgs <- setdiff(pkgs, c("R", rownames(installed.packages(priority = "base"))))
pkgs <- unique(pkgs)

if (!length(pkgs)) {
  message("DESCRIPTION declares no external dependencies; nothing to install")
  quit(status = 0L)
}

if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager")
}

# The Bioconductor pin is advisory, and it has to be. A Bioconductor release is
# tied to an R minor version, and the CRAN apt repository carries only the
# current R -- so a pin that was correct when it was written becomes impossible
# the day R moves, and BiocManager refuses with "requires R version 4.4; use
# version = '3.22'". Failing the build there would mean a Dockerfile comment
# going stale takes the image down, for a pin that only matters once a
# Bioconductor dependency is actually declared. FALCONAge declares none today.
bioc <- Sys.getenv("BIOC_VERSION", "")
if (nzchar(bioc)) {
  ok <- tryCatch({
    BiocManager::install(version = bioc, ask = FALSE, update = FALSE)
    TRUE
  }, error = function(e) {
    message("could not pin Bioconductor ", bioc, " on R ",
            getRversion(), ": ", conditionMessage(e))
    FALSE
  })
  message(if (ok) paste("pinned Bioconductor", bioc)
          else paste("continuing with Bioconductor", as.character(BiocManager::version())))
}

message("installing ", length(pkgs), " package(s) declared in ", desc_path, ":")
message("  ", paste(sort(pkgs), collapse = ", "))
message("from ", repos)

BiocManager::install(pkgs, ask = FALSE, update = FALSE)

# reticulate needs saying twice. R CMD INSTALL succeeds with reticulate present
# and no Python behind it, and the failure then arrives at the first .onLoad in
# a user's session rather than here. Checking that the namespace loads (not just
# that the directory exists) is what catches a half-installed build.
missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1L), quietly = TRUE)]
if (length(missing)) {
  message("\nFAILED to install: ", paste(sort(missing), collapse = ", "))
  quit(status = 1L)
}

message("\nall ", length(pkgs), " declared dependencies present")
