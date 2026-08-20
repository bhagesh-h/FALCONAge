#' @keywords internal
#'
#' @section Why this package is a wrapper:
#' Two implementations of the same clock disagree. Not in the first decimal --
#' in the fourth, because one of them centred before scaling and the other did
#' not, or because one filled an absent probe with a cohort mean and the other
#' with the value the clock's authors published. Those disagreements are the
#' field's reproducibility problem in miniature, and a package that shipped an
#' R port alongside a Python one would be adding to it.
#'
#' So there is one numerical core, written in Python, and this package calls it
#' through \pkg{reticulate}. An R result and a Python result are the same bits.
#' What R gets is not a translation but an idiom: data frames in, data frames
#' out, S3 classes with \code{print}, \code{summary} and \code{plot} methods,
#' and \pkg{ggplot2} rendering of the same numbers matplotlib draws on the other
#' side.
#'
#' @section Getting started:
#' \preformatted{
#' library(FALCONAge)
#' falconage_install()          # one-off: creates the managed Python env
#' falconage_config()           # what resolved: versions, devices, registry
#'
#' data <- read_betas("betas.csv")
#' res  <- score(data, clocks = "compatible")
#' summary(res)
#' acceleration(res)
#' }
#'
#' @section What ships:
#' 161 catalogued clocks. 34 carry coefficients inside the package and run
#' offline; 28 are scaffolds whose coefficients are research-use-only and are
#' not ours to distribute -- \code{list_clocks(tier = "C")} names each one, why,
#' and where to obtain a file. The rest are catalogued with metadata and await a
#' traced extractor.
#'
#' @section Scales, and what may be done with them:
#' Every clock carries a \code{scale_type}, and it governs which downstream
#' operations are defined. Age acceleration is a residual against chronological
#' age: meaningful for a clock that returns years, undefined for one that
#' returns a mortality log-hazard, and actively misleading for a pace of aging,
#' which is already a rate. \code{acceleration()} refuses rather than computing,
#' because a wrong number in a column of numbers is invisible.
#'
"_PACKAGE"

## usethis namespace: start
## usethis namespace: end
NULL
