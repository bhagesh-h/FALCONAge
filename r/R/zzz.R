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
  invisible()
}

.onAttach <- function(libname, pkgname) {
  if (!falconage_available()) {
    packageStartupMessage(
      "FALCONAge: the Python core is not available yet.\n",
      "  falconage_install()   sets it up (one-off, a few minutes)")
  }
}

# Silences R CMD check on ggplot2's .data pronoun without importing rlang.
utils::globalVariables(".data")
