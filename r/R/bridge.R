# =============================================================================
# The reticulate bridge
# =============================================================================
#
# Everything that crosses into Python goes through here, for three reasons that
# are all about failure rather than about calling convention:
#
#   1. A missing interpreter must produce one clear sentence about how to get
#      one, not a reticulate traceback about a module that could not be found.
#   2. A Python exception must arrive as an R condition whose message is the
#      Python message -- those messages carry the remedy (which open clock to
#      use instead, which units dict to pass), and losing them to a generic
#      "Error in py_call_impl" throws away the most useful part of the package.
#   3. Marshalling has exactly one implementation. A data frame converted two
#      different ways in two functions is how row names stop matching.
# =============================================================================

.falconage <- new.env(parent = emptyenv())

#' Is the Python core importable?
#'
#' @return `TRUE` when `falconage` can be imported in the configured
#'   interpreter, `FALSE` otherwise. Never raises, so it is safe in a
#'   conditional or a `skip_if_not()`.
#' @examples
#' falconage_available()
#' @export
falconage_available <- function() {
  isTRUE(tryCatch({
    fa()
    TRUE
  }, error = function(e) FALSE))
}

#' The Python module handle
#'
#' Imported once and cached. Delayed rather than loaded in `.onLoad` so that
#' `library(FALCONAge)` works on a machine with no Python at all -- which is the
#' state every machine is in before [falconage_install()] runs.
#'
#' @return The `falconage` Python module.
#' @keywords internal
fa <- function() {
  if (!is.null(.falconage$mod)) return(.falconage$mod)

  if (!reticulate::py_available(initialize = TRUE)) {
    stop(no_python_message(), call. = FALSE)
  }
  if (!reticulate::py_module_available("falconage")) {
    stop(no_module_message(), call. = FALSE)
  }
  .falconage$mod <- reticulate::import("falconage", delay_load = FALSE, convert = FALSE)
  .falconage$mod
}

no_python_message <- function() {
  paste0(
    "FALCONAge needs a Python interpreter and could not start one.\n\n",
    "  falconage_install()   creates a managed environment with everything needed\n\n",
    "  If you manage Python yourself, point reticulate at it before loading:\n",
    "    Sys.setenv(RETICULATE_PYTHON = '/path/to/python')\n",
    "  and install the core there with:\n",
    "    ", pip_spec()
  )
}

no_module_message <- function() {
  paste0(
    "Python is available but the 'falconage' module is not installed in it.\n\n",
    "  interpreter: ", tryCatch(reticulate::py_config()$python, error = function(e) "unknown"),
    "\n\n  falconage_install()   installs it into a managed environment\n",
    "  or, into the interpreter above:\n    ", pip_spec()
  )
}

#' The pip specifier for the Python core
#'
#' FALCONAge is not on PyPI, so `pip install falconage` finds nothing (or, worse,
#' finds some unrelated name). The core lives in the `python/` subdirectory of
#' the GitHub repository, which is what the `#subdirectory=` fragment selects.
#'
#' The ref is pinned to `v<Version>` from DESCRIPTION rather than left on
#' `main`, so an R package built from a tag installs the Python core from the
#' same tag. Those two halves have to agree: the R package asserts bit equality
#' against the core, and a mismatched pair fails that assertion in a way that
#' looks like a numerical bug rather than a version skew.
#'
#' @return A single string suitable for `pip install`.
#' @keywords internal
pip_spec <- function() {
  v <- as.character(utils::packageVersion("FALCONAge"))
  sprintf(
    "pip install 'falconage @ git+https://github.com/bhagesh-h/FALCONAge.git@v%s#subdirectory=python'",
    v)
}

#' Run a call against the Python core, translating its errors
#'
#' A Python exception becomes an R error whose message is the Python message,
#' verbatim, with the class name attached. Those messages are written to be read
#' -- `WeightsUnavailableError` names an open alternative clock, and
#' `UnitsNotDeclaredError` prints the exact `units=` list to supply -- so
#' replacing them with an R-flavoured summary would throw away the useful part.
#'
#' @param expr An expression calling into the Python module.
#' @return Whatever `expr` returns.
#' @keywords internal
py_do <- function(expr) {
  withCallingHandlers(
    tryCatch(force(expr), error = function(e) {
      msg <- conditionMessage(e)
      cls <- sub("^([A-Za-z]*Error).*", "\\1", strsplit(msg, "\n")[[1]][1])
      # reticulate prefixes the Python traceback; the readable message is the
      # part after it, and keeping the whole thing buries the remedy.
      msg <- sub("(?s)^.*?(?=[A-Za-z]*Error: )", "", msg, perl = TRUE)
      stop(structure(
        class = c(paste0("falconage_", tolower(cls)), "falconage_error",
                  "error", "condition"),
        list(message = msg, call = NULL)
      ))
    }),
    warning = function(w) {
      # Python warnings surface as R warnings rather than being swallowed;
      # coverage and species warnings are the ones users most need to see.
      invokeRestart("muffleWarning")
    }
  )
}

# -- marshalling --------------------------------------------------------------

#' Convert a Python pandas object to an R data frame
#'
#' Goes through plain Python lists rather than letting reticulate convert the
#' DataFrame directly. That is deliberate: reticulate's pandas conversion is
#' registered against a class name that has changed between releases (1.42
#' reports `pandas.DataFrame` where earlier versions reported
#' `pandas.core.frame.DataFrame`), so `py_to_r()` on a DataFrame silently
#' returns the proxy untouched on some installations and a data frame on
#' others. A package whose return type depends on the user's reticulate version
#' is not usable, and the failure is quiet -- `nrow()` on the proxy gives
#' `NULL`, not an error.
#'
#' `to_dict("list")` and `index.tolist()` are stable across every pandas and
#' reticulate version in use, at the cost of one extra copy.
#'
#' @param x A pandas DataFrame or Series proxy.
#' @param row_names Keep the pandas index as row names.
#' @return A data frame.
#' @keywords internal
as_df <- function(x, row_names = TRUE) {
  # Series first: one column, and to_dict() on a Series is keyed by index.
  if (inherits(x, "pandas.Series") ||
      any(grepl("pandas\\.core\\.series", class(x)))) {
    v <- unlist_na(reticulate::py_to_r(x$tolist()))
    out <- data.frame(value = v, stringsAsFactors = FALSE)
    if (row_names) rownames(out) <- as.character(
      reticulate::py_to_r(x$index$astype("str")$tolist()))
    return(out)
  }

  cols <- as.character(reticulate::py_to_r(x$columns$astype("str")$tolist()))
  idx  <- as.character(reticulate::py_to_r(x$index$astype("str")$tolist()))
  raw  <- reticulate::py_to_r(x$to_dict("list"))

  if (!length(cols)) {
    return(data.frame(row.names = if (row_names) idx else NULL))
  }
  out <- as.data.frame(lapply(cols, function(cn) unlist_na(raw[[cn]])),
                       stringsAsFactors = FALSE, optional = TRUE)
  names(out) <- cols
  if (row_names && length(idx) == nrow(out)) rownames(out) <- idx
  out
}

#' Flatten a Python list to an atomic vector, turning None into NA
#'
#' `unlist()` drops NULLs rather than preserving position, which would silently
#' shorten a column and misalign every row after the first missing value.
#'
#' @param v A list from `py_to_r()`.
#' @return An atomic vector of the same length.
#' @keywords internal
unlist_na <- function(v) {
  if (is.null(v)) return(logical(0))
  if (!is.list(v)) return(v)
  v[vapply(v, is.null, logical(1))] <- NA
  unlist(v, use.names = FALSE)
}

#' Convert an R data frame to a pandas DataFrame, keeping row names as the index
#'
#' Row names matter here in a way they usually do not in R: they are the sample
#' identifiers, and every join downstream is on them. reticulate's default
#' conversion drops them.
#'
#' @param df A data frame.
#' @return A pandas DataFrame proxy.
#' @keywords internal
as_pandas <- function(df) {
  pd <- reticulate::import("pandas", convert = FALSE)
  df <- as.data.frame(df, stringsAsFactors = FALSE)
  idx <- rownames(df)
  # Column-wise, for the mirror of the reason as_df() goes through to_dict():
  # r_to_py() on a whole data frame produces a DataFrame on some reticulate
  # versions and a dict on others. A list of columns converts the same way
  # everywhere, and factors become characters rather than integer codes.
  cols <- lapply(df, function(v) if (is.factor(v)) as.character(v) else v)
  out <- pd$DataFrame(reticulate::r_to_py(cols))
  if (!is.null(idx)) out$index <- reticulate::r_to_py(as.character(idx))
  out
}

#' Coerce NULL to the Python None reticulate expects for an absent argument
#' @param x Any value.
#' @return `x`, or Python `None`.
#' @keywords internal
or_none <- function(x) if (is.null(x)) reticulate::py_none() else x

#' Convert a pandas Series to a named R vector
#'
#' Same reasoning as [as_df()]: reticulate's own Series conversion is
#' version-dependent, and a summary that comes back as an opaque environment on
#' one machine and a vector on another is not a summary.
#'
#' @param x A pandas Series proxy.
#' @return A named vector.
#' @keywords internal
as_series <- function(x) {
  v <- unlist_na(reticulate::py_to_r(x$tolist()))
  names(v) <- as.character(reticulate::py_to_r(x$index$astype("str")$tolist()))
  v
}
