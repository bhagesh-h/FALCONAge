# =============================================================================
# Load-time setup
# =============================================================================
#
# Nothing here touches Python. `library(FALCONAge)` has to succeed on a machine
# that has none -- which is the state every machine is in before
# falconage_install() runs -- so the module handle is resolved lazily in fa().
# =============================================================================

.onLoad <- function(libname, pkgname) {
  op <- options()
  defaults <- list(
    # Mirrors FALCONAGE_VERBOSE on the Python side, so a script translated
    # between the languages behaves the same way.
    falconage.verbose = Sys.getenv("FALCONAGE_VERBOSE", "inform"),
    falconage.device  = Sys.getenv("FALCONAGE_DEVICE", "auto")
  )
  toset <- !(names(defaults) %in% names(op))
  if (any(toset)) options(defaults[toset])

  declare_python_requirement()
  invisible()
}

# -----------------------------------------------------------------------------
# Declaring the Python dependency instead of installing it
# -----------------------------------------------------------------------------
# reticulate 1.41 added py_require(), which records what a package needs and
# lets reticulate build a uv-managed ephemeral environment on first use. The
# effect for a reader is that `library(FALCONAge); score(...)` works on a
# machine with no Python at all, and falconage_install() stops being a step
# somebody has to be told about before anything runs.
#
# WHY falconage_install() STAYS. The two answer different questions.
# py_require() resolves whatever satisfies the constraint at the moment it is
# first used, which is right for trying the package and wrong for an analysis
# somebody has to reproduce in a year. falconage_install() pins the core to the
# same tag the R package was built from, so the two halves cannot drift. An
# ephemeral environment is the convenient default; a pinned one is the
# defensible record.
#
# Wrapped because it must not break older reticulate: the function does not
# exist before 1.41, and a package that fails to load on account of a
# convenience is worse than one without the convenience.
declare_python_requirement <- function() {
  if (!requireNamespace("reticulate", quietly = TRUE)) return(invisible(FALSE))
  if (!("py_require" %in% getNamespaceExports("reticulate"))) return(invisible(FALSE))

  # An explicitly configured interpreter wins. Somebody who has run
  # falconage_install(), or set RETICULATE_PYTHON, has already answered this
  # question and should not have a second environment built behind them.
  if (nzchar(Sys.getenv("RETICULATE_PYTHON"))) return(invisible(FALSE))
  if (isTRUE(getOption("falconage.no_py_require", FALSE))) return(invisible(FALSE))

  spec <- sprintf(
    "falconage[methylation,plot,anndata] @ git+%s@%s#subdirectory=python",
    "https://github.com/bhagesh-h/FALCONAge.git",
    falconage_python_ref()
  )
  tryCatch(
    reticulate::py_require(spec),
    error = function(e) invisible(FALSE)
  )
  invisible(TRUE)
}

# The git ref the Python core is taken from. A released R package asks for the
# matching tag; a development build tracks main, because during development
# there is no tag carrying the change being tested.
falconage_python_ref <- function() {
  v <- as.character(utils::packageVersion("FALCONAge"))
  if (grepl("^[0-9]+\\.[0-9]+\\.[0-9]+$", v)) paste0("v", v) else "main"
}

.onAttach <- function(libname, pkgname) {
  if (!falconage_available()) {
    packageStartupMessage(
      "FALCONAge: the Python core is not resolved yet.\n",
      "  Nothing to do if you are on reticulate >= 1.41 -- it will be built on\n",
      "  first use. For a pinned, reproducible environment instead:\n",
      "    falconage_install()   (one-off, a few minutes)")
  }
}

# Silences R CMD check on ggplot2's .data pronoun without importing rlang.
utils::globalVariables(".data")
