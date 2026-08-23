#' Install the Python core FALCONAge computes with
#'
#' Creates a managed virtual environment and installs the `falconage` Python
#' package into it. Run once per machine; [falconage_config()] reports what it
#' resolved to.
#'
#' @section Why an environment of its own:
#' The alternative -- installing into whatever interpreter reticulate finds --
#' works until the day something else in that interpreter upgrades numpy, and
#' then the same script gives a different answer in the seventh decimal with no
#' change to anything the user can see. A named environment makes the
#' dependency set a property of the package rather than of the machine.
#'
#' @section Where the core comes from:
#' FALCONAge is not on PyPI, so this installs from the GitHub repository. The
#' Python package sits in the `python/` subdirectory, which the
#' `#subdirectory=python` fragment selects, and the ref is pinned to the tag
#' matching this R package's version rather than left on `main`. Those two
#' halves have to agree: the R suite asserts bit equality against the core, and
#' a mismatched pair fails that assertion in a way that reads as a numerical bug
#' rather than as version skew.
#'
#' @param envname Environment name. The default keeps it out of the way of any
#'   other reticulate project on the machine.
#' @param method `"auto"`, `"virtualenv"` or `"conda"`, passed to reticulate.
#' @param extras Optional extras to install alongside the core.
#'   `"methylation"` adds the parquet reader; `"plot"` adds matplotlib;
#'   `"anndata"` adds the `.h5ad` reader. Note that `"gpu"` is deliberately
#'   *not* in this list -- see `gpu` below.
#' @param gpu Install CUDA torch as well. Off by default, and not merely out of
#'   caution: on the clocks that ship today the GPU is slower than the CPU
#'   (0.58 s against 3.74 s at 16,384 samples on an RTX 4060), because a linear
#'   clock over a few thousand features is too small a matrix multiplication to
#'   pay for the transfer. It earns its place on the PC clocks and on neural
#'   architectures, none of which are bundled yet.
#' @param cuda CUDA version for the torch wheel index, e.g. `"cu124"`. The
#'   `gpu` extra alone is not enough: pip's default index serves a CUDA build on
#'   Linux and a CPU-only build on Windows under the same name, so the wheel has
#'   to be requested from PyTorch's own index explicitly.
#' @param version Git ref to install from. A version number such as `"1.0.0"`
#'   becomes the tag `v1.0.0`; anything else -- `"main"`, a branch, a commit
#'   SHA -- is used verbatim. Defaults to the version of this R package.
#' @param ... Passed to [reticulate::py_install()].
#'
#' @return Invisibly, the resolved configuration from [falconage_config()].
#' @examples
#' \dontrun{
#' falconage_install()
#' falconage_install(extras = c("plot", "anndata"))
#' falconage_install(gpu = TRUE, cuda = "cu124")
#' falconage_install(version = "main")   # track the development branch
#' }
#' @export
falconage_install <- function(envname = "r-falconage",
                              method = c("auto", "virtualenv", "conda"),
                              extras = c("methylation", "plot", "anndata"),
                              gpu = FALSE, cuda = "cu124",
                              version = NULL, ...) {
  method <- match.arg(method)
  version <- version %||% as.character(utils::packageVersion("FALCONAge"))
  # A bare version number names a release tag; anything else is already a ref.
  ref <- if (grepl("^[0-9]+\\.[0-9]+", version)) paste0("v", version) else version

  # torch first, and from PyTorch's own index. Pip's default index serves a
  # CUDA build of torch on Linux and a CPU-only build on Windows under the same
  # version, so a plain `falconage[gpu]` silently gives half the machines an
  # environment where device="cuda" cannot resolve.
  if (isTRUE(gpu)) {
    idx <- paste0("https://download.pytorch.org/whl/", cuda)
    message("installing torch from ", idx)
    reticulate::py_install("torch", envname = envname, method = method, pip = TRUE,
                           pip_options = c("--index-url", idx), ...)
  }

  spec <- sprintf(
    "falconage%s @ git+https://github.com/bhagesh-h/FALCONAge.git@%s#subdirectory=python",
    if (length(extras)) sprintf("[%s]", paste(extras, collapse = ",")) else "", ref)

  message("installing ", spec, " into '", envname, "'")
  reticulate::py_install(spec, envname = envname, method = method, pip = TRUE, ...)

  .falconage$mod <- NULL   # force a re-import against the new environment
  invisible(falconage_config())
}

#' What this installation resolved to
#'
#' Versions, available compute devices, and the size and version of the clock
#' registry. The same information `falconage config` prints on the command line
#' and `falconage.config()` returns in Python -- one function, three surfaces,
#' so a bug report from any of them is comparable.
#'
#' @return A named list, invisibly when printing.
#' @examples
#' \dontrun{
#' falconage_config()
#' }
#' @export
falconage_config <- function() {
  cfg <- reticulate::py_to_r(py_do(fa()$config()))
  structure(cfg, class = c("falcon_config", "list"))
}

#' @param x A `falcon_config`.
#' @param ... Ignored.
#' @rdname falconage_config
#' @export
print.falcon_config <- function(x, ...) {
  cat("FALCONAge", x$falconage, "\n")
  cat("  registry     ", x$registry_version, " (", x$n_clocks, " clocks)\n", sep = "")
  t <- x$clocks_by_availability
  cat("  availability ", t$bundled, " bundled - ", t$untraced, " untraced - ",
      t$licensed, " licensed\n", sep = "")
  cat("  numpy        ", x$numpy, "\n", sep = "")
  cat("  torch        ", x$torch %||% "not installed (the CPU path needs none)", "\n", sep = "")
  cat("  devices      ", paste(unlist(x$devices), collapse = ", "), "\n", sep = "")
  if (!is.null(x$cuda_devices)) {
    cat("  cuda         ", x$cuda_version, ": ",
        paste(unlist(x$cuda_devices), collapse = ", "), "\n", sep = "")
  }
  cat("  python       ", tryCatch(reticulate::py_config()$python,
                                  error = function(e) "unknown"), "\n", sep = "")
  invisible(x)
}

`%||%` <- function(a, b) if (is.null(a)) b else a
