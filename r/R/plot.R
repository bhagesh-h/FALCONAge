# =============================================================================
# Figures, drawn in R from the same specification
# =============================================================================
#
# The Python side returns (figure, data) from every plot function; R takes the
# data half and draws it with ggplot2. Two reasons: an R user should get an
# object they can add layers to, not a PNG; and shipping matplotlib output into
# an R session would mean two rendering engines to keep visually consistent.
#
# What makes them consistent anyway is that neither side chooses anything. Every
# colour, every font size, and every title, subtitle and one-line description
# comes from colorscheme.yaml -- the same file, read at call time by both. A
# colour changed there changes both languages; a caption reworded there is
# reworded in both. There is no second copy to drift.
# =============================================================================

.scheme <- new.env(parent = emptyenv())

#' The shared visual specification
#'
#' Reads `colorscheme.yaml` -- the single file that decides how every FALCONAge
#' figure looks and what it says, in both languages.
#'
#' Searched for in three places, in order: the `path` argument, the
#' `FALCONAGE_COLORSCHEME` environment variable, then the copy that ships inside
#' the Python core. That order lets you restyle a whole report without touching
#' the installation.
#'
#' @param path Optional path to a colour scheme file.
#' @param reload Force a re-read, after editing the file in a live session.
#' @return A nested list: `palette`, `theme`, `plots`.
#' @examples
#' \dontrun{
#' sch <- falcon_scheme()
#' sch$palette$categorical
#' sch$plots$ba_vs_ca$description
#' }
#' @export
falcon_scheme <- function(path = NULL, reload = FALSE) {
  if (!reload && !is.null(.scheme$doc) && is.null(path)) return(.scheme$doc)
  p <- path %||% Sys.getenv("FALCONAGE_COLORSCHEME", "")
  if (!nzchar(p)) {
    # The packaged copy lives beside the Python plot module, so the two
    # languages cannot end up reading different files.
    p <- tryCatch({
      m <- reticulate::import("falconage.plot.spec", convert = TRUE)
      m$load()[["_path"]]
    }, error = function(e) "")
  }
  if (!nzchar(p) || !file.exists(p)) {
    stop("colour scheme not found. Set FALCONAGE_COLORSCHEME to a copy of ",
         "colorscheme.yaml, or install the Python core with falconage_install().",
         call. = FALSE)
  }
  doc <- yaml::read_yaml(p)
  if (!identical(as.integer(doc$schema_version), 1L)) {
    stop(p, ": schema_version is not 1", call. = FALSE)
  }
  .scheme$doc <- doc
  doc
}

#' The shared categorical palette
#'
#' Okabe-Ito, colour-blind safe, ordered so the first two entries -- what a
#' two-group comparison uses -- are the most separable pair in the set.
#'
#' @param which `"categorical"`, `"sequential"`, or a name under `semantic`.
#' @return A character vector of hex colours, or one colour for a semantic role.
#' @examples
#' \dontrun{
#' falcon_palette()
#' falcon_palette("case")
#' }
#' @export
falcon_palette <- function(which = "categorical") {
  p <- falcon_scheme()$palette
  if (which %in% names(p)) return(unlist(p[[which]], use.names = FALSE))
  if (which %in% names(p$semantic)) return(p$semantic[[which]])
  stop("unknown palette or role: ", which, call. = FALSE)
}

.sem <- function(role) falcon_scheme()$palette$semantic[[role]]
.thm <- function(key) falcon_scheme()$theme[[key]]

.plat_col <- function(x) {
  m <- falcon_scheme()$palette$platform
  vapply(as.character(x), function(v) {
    if (!is.null(m[[v]])) m[[v]] else m[["unknown"]]
  }, character(1))
}

# Assign colours to group levels, with control arms pinned to the semantic
# control colour so a reader comparing two panels never has to re-read a legend.
.group_cols <- function(levels) {
  pal <- falcon_palette()
  ctrl <- c("hc", "control", "healthy", "ctrl", "normal", "none")
  out <- character(0); i <- 1L
  for (lv in as.character(levels)) {
    if (tolower(lv) %in% ctrl) {
      out[lv] <- .sem("control")
    } else {
      out[lv] <- pal[[((i - 1L) %% length(pal)) + 1L]]; i <- i + 1L
    }
  }
  out
}

# Fill the {} fields of a plot's title / subtitle / description. Missing fields
# become "?" rather than erroring: losing a whole figure to a formatting gap is
# a worse outcome than a subtitle with a hole in it.
.txt <- function(plot, ...) {
  spec <- falcon_scheme()$plots[[plot]]
  if (is.null(spec)) stop("no text defined for plot '", plot, "'", call. = FALSE)
  f <- list(...)
  fill <- function(s) {
    if (is.null(s) || !nzchar(s)) return("")
    for (k in names(f)) s <- gsub(paste0("{", k, "}"), as.character(f[[k]]), s, fixed = TRUE)
    gsub("\\{[a-z_0-9]+\\}", "?", s)
  }
  list(title = fill(spec$title), subtitle = fill(spec$subtitle),
       caption = fill(gsub("\\s+", " ", trimws(spec$description %||% ""))),
       xlab = fill(spec$xlab), ylab = fill(spec$ylab))
}

need_ggplot <- function() {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("this plot needs ggplot2: install.packages('ggplot2')\n",
         "  Every plot function also returns its data, so you can draw it ",
         "another way without the dependency.", call. = FALSE)
  }
}

#' The shared ggplot2 theme
#'
#' Reads its sizes from `colorscheme.yaml`, so a change there moves both
#' languages together.
#'
#' @return A ggplot2 theme.
#' @examples
#' \dontrun{
#' ggplot2::ggplot(df) + falcon_theme()
#' }
#' @export
falcon_theme <- function() {
  need_ggplot()
  ggplot2::theme_minimal(base_size = .thm("base_size")) +
    ggplot2::theme(
      panel.grid.minor = ggplot2::element_blank(),
      panel.grid.major = ggplot2::element_line(linewidth = 0.3,
                                               colour = grDevices::grey(0.85)),
      plot.title = ggplot2::element_text(size = .thm("title_size"), hjust = 0),
      plot.subtitle = ggplot2::element_text(size = .thm("subtitle_size"),
                                            colour = "#555555", hjust = 0),
      plot.caption = ggplot2::element_text(size = .thm("caption_size"),
                                           colour = "#666666", hjust = 0),
      plot.caption.position = "plot",
      plot.title.position = "plot",
      legend.title = ggplot2::element_blank())
}

.labs <- function(t) {
  ggplot2::labs(title = t$title, subtitle = t$subtitle, caption = t$caption,
                x = if (nzchar(t$xlab)) t$xlab else NULL,
                y = if (nzchar(t$ylab)) t$ylab else NULL)
}

.refline <- function(dir = "h", at = 0) {
  f <- if (dir == "h") ggplot2::geom_hline else ggplot2::geom_vline
  args <- list(colour = .sem("reference"), linewidth = .thm("line_width") * 0.4,
               linetype = .thm("reference_line"))
  do.call(f, c(stats::setNames(list(at), if (dir == "h") "yintercept" else "xintercept"), args))
}

.unit_of <- function(x, clock) {
  u <- clock_info_quiet(clock)$unit
  if (length(u)) paste(u, collapse = ", ") else "score"
}

clock_info_quiet <- function(clock) {
  reg <- py_do(fa()$registry$load())
  c <- py_do(reg$get(clock))
  list(unit = unlist(reticulate::py_to_r(c$unit)),
       scale_type = reticulate::py_to_r(c$scale_type))
}

# =============================================================================
# per-clock accuracy and calibration
# =============================================================================

#' Predicted against chronological age
#'
#' With the identity line, not a regression line. A regression line always looks
#' like a good fit; the identity line is what exposes a clock running five years
#' high on everybody, which is what MedE measures and what quietly wins AA1 in a
#' benchmark that does not discount for it.
#'
#' @param x A `falcon_result`.
#' @param clock A clock id.
#' @param age_col Chronological age column in the sample annotation.
#' @param group Optional grouping column.
#' @return A ggplot.
#' @examples
#' \dontrun{
#' plot_ba_vs_ca(res, "horvath2013", group = "condition")
#' }
#' @export
plot_ba_vs_ca <- function(x, clock, age_col = "age", group = NULL) {
  need_ggplot()
  o <- obs(x)
  d <- data.frame(chronological = suppressWarnings(as.numeric(o[[age_col]])),
                  predicted = as.data.frame(x)[[clock]])
  if (!is.null(group)) d$group <- as.character(o[[group]])
  d <- d[stats::complete.cases(d[, c("chronological", "predicted")]), , drop = FALSE]

  t <- .txt("ba_vs_ca", clock = clock, n = nrow(d),
            r = sprintf("%.2f", stats::cor(d$chronological, d$predicted)),
            medae = sprintf("%.2f", stats::median(abs(d$predicted - d$chronological))),
            unit = .unit_of(x, clock))

  p <- ggplot2::ggplot(d, ggplot2::aes(x = .data$chronological, y = .data$predicted)) +
    ggplot2::geom_abline(slope = 1, intercept = 0, linetype = .thm("reference_line"),
                         colour = .sem("reference")) +
    .labs(t) + falcon_theme()
  if (!is.null(group)) {
    p + ggplot2::geom_point(ggplot2::aes(colour = .data$group),
                            size = .thm("point_size"), alpha = .thm("point_alpha")) +
      ggplot2::scale_colour_manual(values = .group_cols(sort(unique(d$group))))
  } else {
    p + ggplot2::geom_point(size = .thm("point_size"), alpha = .thm("point_alpha"),
                            colour = falcon_palette()[[1]])
  }
}

#' Bland-Altman agreement across the age range
#'
#' A correlation coefficient cannot show that a clock's error depends on age.
#' This can, and age-dependent error is the failure mode that makes a single
#' MedAE meaningless.
#'
#' @inheritParams plot_ba_vs_ca
#' @return A ggplot.
#' @examples
#' \dontrun{
#' plot_bland_altman(res, "horvath2013")
#' }
#' @export
plot_bland_altman <- function(x, clock, age_col = "age") {
  need_ggplot()
  age <- suppressWarnings(as.numeric(obs(x)[[age_col]]))
  y <- as.data.frame(x)[[clock]]
  d <- data.frame(mean = (age + y) / 2, diff = y - age)
  d <- d[stats::complete.cases(d), , drop = FALSE]
  bias <- mean(d$diff); sd_ <- stats::sd(d$diff)
  lo <- bias - 1.96 * sd_; hi <- bias + 1.96 * sd_

  ggplot2::ggplot(d, ggplot2::aes(x = .data$mean, y = .data$diff)) +
    .refline("h", 0) +
    ggplot2::geom_hline(yintercept = bias, colour = .sem("case")) +
    ggplot2::geom_hline(yintercept = c(lo, hi), colour = .sem("case"), linetype = "dotted") +
    ggplot2::geom_point(size = .thm("point_size"), alpha = .thm("point_alpha"),
                        colour = falcon_palette()[[1]]) +
    .labs(.txt("bland_altman", clock = clock, bias = sprintf("%.2f", bias),
               lo = sprintf("%.1f", lo), hi = sprintf("%.1f", hi))) +
    falcon_theme()
}

#' Residual against chronological age
#'
#' The slope is the diagnostic. A negative one means the clock over-ages the
#' young and under-ages the old -- regression to the mean, the most common
#' artefact in this field and the one most often reported as a finding.
#'
#' @inheritParams plot_ba_vs_ca
#' @return A ggplot.
#' @examples
#' \dontrun{
#' plot_calibration(res, "horvath2013")
#' }
#' @export
plot_calibration <- function(x, clock, age_col = "age") {
  need_ggplot()
  age <- suppressWarnings(as.numeric(obs(x)[[age_col]]))
  y <- as.data.frame(x)[[clock]]
  ok <- stats::complete.cases(cbind(age, y))
  fit <- stats::lm(y[ok] ~ age[ok])
  d <- data.frame(chronological = age[ok], residual = stats::resid(fit))
  slope <- stats::coef(stats::lm(residual ~ chronological, d))[[2]]

  ggplot2::ggplot(d, ggplot2::aes(x = .data$chronological, y = .data$residual)) +
    .refline("h", 0) +
    ggplot2::geom_point(size = .thm("point_size"), alpha = .thm("point_alpha"),
                        colour = falcon_palette()[[1]]) +
    ggplot2::geom_smooth(method = "lm", formula = y ~ x, se = FALSE,
                         colour = .sem("case"), linewidth = .thm("line_width") * 0.5) +
    .labs(.txt("calibration", clock = clock, slope = sprintf("%+.3f", slope))) +
    falcon_theme()
}

# =============================================================================
# age acceleration
# =============================================================================

#' Age acceleration by group
#'
#' The case/control workhorse: box for the median and interquartile range,
#' points for every sample, because a box on twelve samples hides how few.
#'
#' @param acc Output of [acceleration()].
#' @param clock A clock id.
#' @param obs Sample annotation, from [obs()].
#' @param group Grouping column.
#' @return A ggplot.
#' @examples
#' \dontrun{
#' plot_acceleration_by_group(acceleration(res), "horvath2013", obs(res), "condition")
#' }
#' @export
plot_acceleration_by_group <- function(acc, clock, obs, group) {
  need_ggplot()
  d <- data.frame(acceleration = acc[[clock]], group = as.character(obs[[group]]))
  d <- d[stats::complete.cases(d), , drop = FALSE]
  lv <- sort(unique(d$group))
  n <- table(d$group)
  d$label <- paste0(d$group, "\nn=", as.integer(n[d$group]))

  ggplot2::ggplot(d, ggplot2::aes(x = .data$label, y = .data$acceleration,
                                  fill = .data$group, colour = .data$group)) +
    .refline("h", 0) +
    ggplot2::geom_boxplot(alpha = 0.35, outlier.shape = NA, width = 0.55) +
    ggplot2::geom_jitter(width = 0.13, size = .thm("point_size") * 0.9,
                         alpha = .thm("point_alpha")) +
    ggplot2::scale_fill_manual(values = .group_cols(lv), guide = "none") +
    ggplot2::scale_colour_manual(values = .group_cols(lv), guide = "none") +
    .labs(.txt("acceleration_by_group", clock = clock,
               method = attr(acc, "method") %||% "residual",
               groups = paste(lv, collapse = " vs "))) +
    falcon_theme()
}

#' Distribution of age acceleration
#'
#' Counted, not smoothed. A kernel density on a dozen cases invents structure
#' that is not in the data, and the figure is usually read before anyone checks
#' the sample size.
#'
#' @inheritParams plot_acceleration_by_group
#' @param group Optional grouping column.
#' @return A ggplot.
#' @examples
#' \dontrun{
#' plot_acceleration(acceleration(res), "horvath2013", obs(res), "condition")
#' }
#' @export
plot_acceleration <- function(acc, clock, obs = NULL, group = NULL) {
  need_ggplot()
  d <- data.frame(acceleration = acc[[clock]])
  if (!is.null(group) && !is.null(obs)) d$group <- as.character(obs[[group]])
  d <- d[stats::complete.cases(d), , drop = FALSE]

  p <- ggplot2::ggplot(d, ggplot2::aes(x = .data$acceleration)) +
    .refline("v", 0) +
    .labs(.txt("acceleration_density", clock = clock, n = nrow(d),
               method = attr(acc, "method") %||% "residual")) +
    falcon_theme()
  if (!is.null(d$group)) {
    p + ggplot2::geom_histogram(ggplot2::aes(fill = .data$group), bins = 24,
                                alpha = 0.55, position = "identity") +
      ggplot2::scale_fill_manual(values = .group_cols(sort(unique(d$group))))
  } else {
    p + ggplot2::geom_histogram(bins = 24, fill = falcon_palette()[[1]], alpha = 0.8)
  }
}

#' Age acceleration across clocks and samples
#'
#' Z-scored per clock, because the clocks are on wildly different scales;
#' without it one clock's variance dominates the colour map.
#'
#' @param acc Output of [acceleration()].
#' @return A ggplot.
#' @examples
#' \dontrun{
#' plot_acceleration_heatmap(acceleration(res))
#' }
#' @export
plot_acceleration_heatmap <- function(acc) {
  need_ggplot()
  z <- scale(as.matrix(acc))
  d <- data.frame(sample = rep(rownames(acc), times = ncol(acc)),
                  clock = rep(colnames(acc), each = nrow(acc)),
                  z = as.vector(z))
  dv <- falcon_scheme()$palette$diverging
  lim <- stats::quantile(abs(d$z), 0.98, na.rm = TRUE)

  ggplot2::ggplot(d, ggplot2::aes(x = .data$sample, y = .data$clock, fill = .data$z)) +
    ggplot2::geom_tile() +
    ggplot2::scale_fill_gradient2(low = dv$low, mid = dv$mid, high = dv$high,
                                  midpoint = 0, limits = c(-lim, lim), oob = scales_squish) +
    .labs(.txt("acceleration_heatmap", n_clocks = ncol(acc), n_samples = nrow(acc))) +
    falcon_theme() +
    ggplot2::theme(axis.text.x = ggplot2::element_blank(),
                   panel.grid = ggplot2::element_blank())
}

# `scales` is not a declared dependency; ggplot2 re-exports what is needed here.
scales_squish <- function(x, range = c(0, 1), only.finite = TRUE) {
  force(range)
  x[x < range[1] & is.finite(x)] <- range[1]
  x[x > range[2] & is.finite(x)] <- range[2]
  x
}

#' Forest plot of benchmark effect sizes
#'
#' The interval is what makes this honest. An eight-versus-eight comparison with
#' a twenty-year point estimate has an interval wide enough to say so, and a bar
#' chart of the point estimates alone would not.
#'
#' @param bench Output of [run_benchmark()].
#' @param top Keep only the largest effects in each direction.
#' @return A ggplot.
#' @examples
#' \dontrun{
#' plot_forest(run_benchmark(res))
#' }
#' @export
plot_forest <- function(bench, top = NULL) {
  need_ggplot()
  d <- bench$per_dataset
  se <- sqrt(1 / pmax(d$n_case, 1) + 1 / pmax(d$n_control, 1)) *
    stats::ave(abs(d$delta), d$clock, FUN = function(v) max(v, na.rm = TRUE)) * 0.8
  d$lo <- d$delta - 1.96 * se
  d$hi <- d$delta + 1.96 * se
  # The separator is a middle dot, written as an escape rather than as the character
  # itself: R CMD check rejects non-ASCII bytes in R source, and an escape
  # survives every locale a check machine might run under. Renders identically.
  d$label <- paste(d$clock, d$dataset, d$condition, sep = " \u00b7 ")
  d <- d[order(d$delta), , drop = FALSE]
  if (!is.null(top) && nrow(d) > top) {
    d <- rbind(utils::head(d, top %/% 2), utils::tail(d, top - top %/% 2))
  }
  d$label <- factor(d$label, levels = d$label)

  ggplot2::ggplot(d, ggplot2::aes(x = .data$delta, y = .data$label,
                                  colour = .data$significant)) +
    .refline("v", 0) +
    ggplot2::geom_errorbarh(ggplot2::aes(xmin = .data$lo, xmax = .data$hi), height = 0) +
    ggplot2::geom_point(size = .thm("point_size")) +
    ggplot2::scale_colour_manual(values = c("FALSE" = .sem("neutral"),
                                            "TRUE" = .sem("case")), guide = "none") +
    .labs(.txt("forest", n = nrow(d))) +
    falcon_theme()
}

# =============================================================================
# cross-clock
# =============================================================================

#' Agreement between clocks
#'
#' Rank correlation, not Pearson: clocks on different scales have no meaningful
#' linear correlation. Blocks of agreement are usually shared training cohorts
#' rather than shared biology, which is what [plot_clock_chord()] can confirm.
#'
#' @param x A `falcon_result`.
#' @param method `"spearman"` or `"pearson"`.
#' @param cluster Order rows and columns by hierarchical clustering.
#' @return A ggplot.
#' @examples
#' \dontrun{
#' plot_agreement(res)
#' }
#' @export
plot_agreement <- function(x, method = c("spearman", "pearson"), cluster = TRUE) {
  need_ggplot()
  method <- match.arg(method)
  m <- agreement(x, method)
  if (isTRUE(cluster) && nrow(m) > 2) {
    ord <- stats::hclust(stats::as.dist(1 - m), method = "average")$order
    m <- m[ord, ord, drop = FALSE]
  }
  d <- data.frame(a = factor(rep(rownames(m), times = ncol(m)), levels = rownames(m)),
                  b = factor(rep(colnames(m), each = nrow(m)), levels = colnames(m)),
                  r = as.vector(m))
  dv <- falcon_scheme()$palette$diverging

  ggplot2::ggplot(d, ggplot2::aes(x = .data$a, y = .data$b, fill = .data$r)) +
    ggplot2::geom_tile() +
    ggplot2::scale_fill_gradient2(low = dv$low, mid = dv$mid, high = dv$high,
                                  midpoint = 0, limits = c(-1, 1)) +
    .labs(.txt("clock_corr", n_clocks = nrow(m), n = nrow(as.data.frame(x)))) +
    falcon_theme() +
    ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 90, hjust = 1, vjust = 0.5),
                   panel.grid = ggplot2::element_blank())
}

#' Samples embedded in clock space
#'
#' PC1 is almost always chronological age; that is expected and is not a
#' finding. Structure on PC2 that tracks a plate and not a phenotype is a batch
#' effect, and this is where it shows up.
#'
#' @param x A `falcon_result`.
#' @param colour_by Column in the sample annotation.
#' @return A ggplot.
#' @examples
#' \dontrun{
#' plot_clock_pca(res, colour_by = "condition")
#' }
#' @export
plot_clock_pca <- function(x, colour_by = NULL) {
  need_ggplot()
  s <- scale(as.matrix(as.data.frame(x)))
  s[!is.finite(s)] <- 0
  pc <- stats::prcomp(s, center = TRUE, scale. = FALSE)
  var <- 100 * pc$sdev^2 / sum(pc$sdev^2)
  d <- data.frame(d1 = pc$x[, 1], d2 = pc$x[, 2])

  t <- .txt("clock_pca", pc1 = sprintf("%.0f", var[1]), pc2 = sprintf("%.0f", var[2]))
  if (is.null(colour_by)) {
    return(ggplot2::ggplot(d, ggplot2::aes(x = .data$d1, y = .data$d2)) +
             ggplot2::geom_point(size = .thm("point_size"),
                                 colour = falcon_palette()[[1]]) +
             .labs(t) + falcon_theme())
  }

  v <- obs(x)[[colour_by]]
  numeric_scale <- is.numeric(v)
  d$colour <- if (numeric_scale) v else as.character(v)
  p <- ggplot2::ggplot(d, ggplot2::aes(x = .data$d1, y = .data$d2,
                                       colour = .data$colour)) +
    ggplot2::geom_point(size = .thm("point_size"), alpha = .thm("point_alpha")) +
    .labs(t) + falcon_theme()
  if (numeric_scale) {
    p + ggplot2::scale_colour_viridis_c(name = colour_by)
  } else {
    p + ggplot2::scale_colour_manual(values = .group_cols(sort(unique(d$colour))))
  }
}

#' Circos chord diagram of CpG sharing between clocks
#'
#' The correlation heatmap says two clocks agree. This says how much of that
#' agreement is built in: chord width is the number of CpGs the two clocks have
#' literally in common. A pair with a thick chord and a high correlation has
#' told you much less than a pair with a high correlation and no chord.
#'
#' Follows the circos convention of the single-cell aging literature: an outer
#' ring of colour-coded entities, chords in the interior weighted by the
#' strength of the relationship.
#'
#' @param x A `falcon_result`.
#' @param min_shared Minimum shared CpGs for a chord to be drawn.
#' @param max_clocks Cap on ring size, for legibility.
#' @return A ggplot.
#' @examples
#' \dontrun{
#' plot_clock_chord(res)
#' }
#' @export
plot_clock_chord <- function(x, min_shared = 5L, max_clocks = 24L) {
  need_ggplot()
  reg <- py_do(fa()$registry$load())
  clocks <- colnames(as.data.frame(x))
  clocks <- Filter(function(c) isTRUE(reticulate::py_to_r(reg$has_coefficients(c))), clocks)
  clocks <- utils::head(clocks, max_clocks)
  feats <- lapply(clocks, function(c) as.character(unlist_na(
    reticulate::py_to_r(reg$feature_ids(c)))))
  names(feats) <- clocks

  n <- length(clocks)
  ang <- stats::setNames(2 * pi * (seq_len(n) - 1) / n, clocks)
  pal <- falcon_palette()

  ring <- do.call(rbind, lapply(seq_len(n), function(i) {
    a <- seq(ang[i] - pi / n * 0.86, ang[i] + pi / n * 0.86, length.out = 24)
    data.frame(x = cos(a), y = sin(a), clock = clocks[i])
  }))
  lab <- data.frame(x = 1.13 * cos(ang), y = 1.13 * sin(ang), clock = clocks)

  chords <- list(); shared <- data.frame()
  for (i in seq_len(n - 1L)) for (j in (i + 1L):n) {
    k <- length(intersect(feats[[i]], feats[[j]]))
    if (k < min_shared) next
    shared <- rbind(shared, data.frame(a = clocks[i], b = clocks[j], shared = k))
    p0 <- c(cos(ang[i]), sin(ang[i])); p1 <- c(cos(ang[j]), sin(ang[j]))
    sep <- abs(((ang[i] - ang[j] + pi) %% (2 * pi)) - pi) / pi
    ctrl <- (p0 + p1) / 2 * (1 - 0.85 * sep)
    tt <- seq(0, 1, length.out = 60)
    chords[[length(chords) + 1L]] <- data.frame(
      x = (1 - tt)^2 * p0[1] + 2 * (1 - tt) * tt * ctrl[1] + tt^2 * p1[1],
      y = (1 - tt)^2 * p0[2] + 2 * (1 - tt) * tt * ctrl[2] + tt^2 * p1[2],
      id = length(chords) + 1L, clock = clocks[i], shared = k)
  }
  cd <- if (length(chords)) do.call(rbind, chords) else
    data.frame(x = numeric(0), y = numeric(0), id = integer(0),
               clock = character(0), shared = numeric(0))

  p <- ggplot2::ggplot()
  if (nrow(cd)) {
    wmax <- max(cd$shared)
    p <- p + ggplot2::geom_path(
      data = cd, ggplot2::aes(x = .data$x, y = .data$y, group = .data$id,
                              colour = .data$clock,
                              linewidth = .data$shared, alpha = .data$shared)) +
      ggplot2::scale_linewidth(range = c(0.15, 1.6), guide = "none") +
      ggplot2::scale_alpha(range = c(0.18, 0.68), guide = "none")
  }
  p +
    ggplot2::geom_path(data = ring, ggplot2::aes(x = .data$x, y = .data$y,
                                                 group = .data$clock,
                                                 colour = .data$clock),
                       linewidth = 2.4, lineend = "butt") +
    ggplot2::geom_text(data = lab, ggplot2::aes(x = .data$x, y = .data$y,
                                                label = .data$clock),
                       size = .thm("caption_size") / 3, colour = "#444444") +
    ggplot2::scale_colour_manual(values = stats::setNames(
      pal[((seq_len(n) - 1L) %% length(pal)) + 1L], clocks), guide = "none") +
    ggplot2::coord_fixed(xlim = c(-1.3, 1.3), ylim = c(-1.3, 1.3)) +
    .labs(.txt("clock_chord", n_clocks = n, n_chords = nrow(shared),
               min_shared = min_shared)) +
    falcon_theme() +
    ggplot2::theme(axis.text = ggplot2::element_blank(),
                   panel.grid = ggplot2::element_blank())
}

#' Multi-clock radar profile
#'
#' Z-scored within clock against the cohort, because the axes are otherwise in
#' incompatible units and the polygon would be a picture of the scales rather
#' than of the samples.
#'
#' @param x A `falcon_result`.
#' @param group Optional grouping column; one polygon per level.
#' @param max_clocks Cap on the number of axes.
#' @return A ggplot in polar coordinates.
#' @examples
#' \dontrun{
#' plot_clock_radar(res, group = "condition")
#' }
#' @export
plot_clock_radar <- function(x, group = NULL, max_clocks = 12L) {
  need_ggplot()
  s <- as.data.frame(x)
  cols <- utils::head(colnames(s), max_clocks)
  if (length(cols) < 3L)
    stop("clock_radar: a polygon needs at least three axes", call. = FALSE)
  z <- as.data.frame(scale(as.matrix(s[, cols, drop = FALSE])))

  if (!is.null(group)) {
    g <- as.character(obs(x)[[group]])
    prof <- do.call(rbind, lapply(split(z, g), function(sub)
      apply(sub, 2, stats::median, na.rm = TRUE)))
    label <- paste(nrow(prof), "groups by", group)
  } else {
    prof <- matrix(apply(z, 2, stats::median, na.rm = TRUE), nrow = 1,
                   dimnames = list("all samples", cols))
    label <- "cohort median"
  }

  d <- data.frame(group = rep(rownames(prof), times = ncol(prof)),
                  clock = factor(rep(colnames(prof), each = nrow(prof)), levels = cols),
                  value = as.vector(prof))
  # Repeat the first axis so the polygon closes.
  d <- rbind(d, transform(d[d$clock == cols[1], ], clock = factor(cols[1], levels = cols)))

  n <- length(cols)
  rmax <- max(abs(prof), na.rm = TRUE) * 1.15
  if (!is.finite(rmax) || rmax == 0) rmax <- 1

  # Spike labels, matching the Python figure. `coord_polar` writes its x tick
  # labels tangentially, and past about eight axes they run into each other, so
  # they are switched off and each clock name is drawn along its own radius
  # instead -- flipped through 180 degrees on the left half so none of them
  # reads upside down.
  theta <- 360 * (seq_len(n) - 1L) / n          # clockwise from twelve o'clock
  scr <- (90 - theta) %% 360                    # the same direction on screen
  flip <- scr > 90 & scr < 270
  lab <- data.frame(clock = factor(cols, levels = cols),
                    value = rmax * 1.06,
                    angle = ifelse(flip, scr - 180, scr),
                    hjust = ifelse(flip, 1, 0))
  # Radial room for the longest name at its outward extent. Without it the
  # panel ends at rmax and ggplot clips the labels away at the boundary.
  headroom <- rmax * (0.10 + 0.052 * max(nchar(cols)))

  # Legend along the bottom rather than beside the panel. On the right it has
  # to share the margin with the spikes pointing that way, and with a dozen
  # study groups it wins -- which is how it came to sit on top of them.
  ncol_leg <- min(6L, nrow(prof))

  ggplot2::ggplot(d, ggplot2::aes(x = .data$clock, y = .data$value,
                                  group = .data$group, colour = .data$group,
                                  fill = .data$group)) +
    ggplot2::geom_hline(yintercept = 0, colour = .sem("reference"),
                        linetype = .thm("reference_line")) +
    ggplot2::geom_polygon(alpha = 0.14, linewidth = .thm("line_width") * 0.6) +
    ggplot2::geom_text(data = lab,
                       ggplot2::aes(x = .data$clock, y = .data$value,
                                    label = .data$clock, angle = .data$angle,
                                    hjust = .data$hjust),
                       inherit.aes = FALSE, size = (.thm("caption_size") - 0.5) / 3,
                       colour = "#444444") +
    ggplot2::scale_y_continuous(limits = c(-rmax, rmax + headroom)) +
    ggplot2::coord_polar() +
    ggplot2::scale_colour_manual(values = .group_cols(rownames(prof))) +
    ggplot2::scale_fill_manual(values = .group_cols(rownames(prof)), guide = "none") +
    ggplot2::guides(colour = ggplot2::guide_legend(ncol = ncol_leg)) +
    .labs(.txt("clock_radar", n_clocks = n, label = label)) +
    falcon_theme() +
    ggplot2::theme(axis.text.x = ggplot2::element_blank(),
                   axis.title = ggplot2::element_blank(),
                   legend.position = "bottom",
                   legend.box.margin = ggplot2::margin(t = 6))
}

# =============================================================================
# quality control
# =============================================================================

#' Per-clock feature coverage
#'
#' The plot to read before the scores. Below the dashed line a clock is mostly
#' scoring imputed values, which is why it was refused.
#'
#' @param x A `falcon_result`.
#' @param floor Coverage threshold, matching the one used at score time.
#' @return A ggplot.
#' @examples
#' \dontrun{
#' plot_coverage(res)
#' }
#' @export
plot_coverage <- function(x, floor = 0.8) {
  need_ggplot()
  d <- coverage(x)
  d <- d[!is.na(d$coverage), c("clock", "coverage"), drop = FALSE]
  d$clock <- factor(d$clock, levels = d$clock[order(d$coverage)])
  d$band <- cut(d$coverage, c(-Inf, floor, 0.95, Inf),
                labels = c("below floor", "marginal", "usable"))

  ggplot2::ggplot(d, ggplot2::aes(x = .data$coverage, y = .data$clock,
                                  fill = .data$band)) +
    ggplot2::geom_col() +
    .refline("v", floor) +
    ggplot2::scale_fill_manual(values = c("below floor" = .sem("fail"),
                                          "marginal" = .sem("warn"),
                                          "usable" = .sem("pass"))) +
    ggplot2::xlim(0, 1) +
    .labs(.txt("coverage_bar", n_clocks = nrow(d), floor = round(floor * 100),
               platform = "mixed")) +
    falcon_theme()
}

#' Score distribution by platform or by study
#'
#' A shift between platforms in the same tissue is a technical effect, and it is
#' the reason cross-study comparisons need the platform as a covariate.
#' Between-study spread is usually larger than the within-study effect being
#' tested, which is why AA2 compares cases with their own controls.
#'
#' @param x A `falcon_result`.
#' @param clock A clock id.
#' @param col Column in the sample annotation to split by.
#' @return A ggplot.
#' @examples
#' \dontrun{
#' plot_by_platform(res, "horvath2013")
#' plot_by_study(res, "horvath2013")
#' }
#' @export
plot_by_platform <- function(x, clock, col = "platform") {
  .split_box(x, clock, col, "platform_comparison", "n_platforms", TRUE)
}

#' @rdname plot_by_platform
#' @export
plot_by_study <- function(x, clock, col = "dataset") {
  .split_box(x, clock, col, "study_comparison", "n_studies", FALSE)
}

.split_box <- function(x, clock, col, plot_name, count_key, by_platform) {
  need_ggplot()
  d <- data.frame(value = as.data.frame(x)[[clock]],
                  split = as.character(obs(x)[[col]]))
  d <- d[stats::complete.cases(d), , drop = FALSE]
  lv <- sort(unique(d$split))
  cols <- if (by_platform) .plat_col(lv) else .group_cols(lv)
  names(cols) <- lv
  n <- table(d$split)
  d$label <- paste0(d$split, "\nn=", as.integer(n[d$split]))

  fields <- list(clock = clock, n = nrow(d), unit = .unit_of(x, clock))
  fields[[count_key]] <- length(lv)

  ggplot2::ggplot(d, ggplot2::aes(x = .data$label, y = .data$value,
                                  fill = .data$split, colour = .data$split)) +
    ggplot2::geom_boxplot(alpha = 0.4, outlier.shape = NA, width = 0.6) +
    ggplot2::geom_jitter(width = 0.14, size = .thm("point_size") * 0.8, alpha = 0.8) +
    ggplot2::scale_fill_manual(values = cols, guide = "none") +
    ggplot2::scale_colour_manual(values = cols, guide = "none") +
    .labs(do.call(.txt, c(list(plot_name), fields))) +
    falcon_theme()
}

# =============================================================================
# benchmark
# =============================================================================

#' AA2 and AA1 counts per clock
#'
#' @param bench Output of [run_benchmark()].
#' @return A ggplot.
#' @examples
#' \dontrun{
#' plot_benchmark(run_benchmark(res))
#' }
#' @export
plot_benchmark <- function(bench) {
  need_ggplot()
  s <- bench$summary
  d <- data.frame(clock = rep(rownames(s), 2),
                  test = rep(c("AA2 (vs own controls)", "AA1 (vs zero)"),
                             each = nrow(s)),
                  n = c(s$AA2, s$AA1))
  d$clock <- factor(d$clock, levels = rev(rownames(s)))

  ggplot2::ggplot(d, ggplot2::aes(x = .data$n, y = .data$clock, fill = .data$test)) +
    ggplot2::geom_col(position = ggplot2::position_dodge(width = 0.75), width = 0.7) +
    ggplot2::scale_fill_manual(values = falcon_palette()[1:2]) +
    .labs(.txt("benchmark_bars", n_clocks = nrow(s),
               n_datasets = length(unique(bench$per_dataset$dataset)), alpha = 0.05)) +
    falcon_theme()
}

#' Error against bias on healthy controls
#'
#' Neither axis is a ranking, and that is the point of drawing them together. A
#' clock at the origin predicts chronological age well, which is the one thing a
#' useful clock does not have to do.
#'
#' @param bench Output of [run_benchmark()].
#' @return A ggplot.
#' @examples
#' \dontrun{
#' plot_benchmark_error_bias(run_benchmark(res))
#' }
#' @export
plot_benchmark_error_bias <- function(bench) {
  need_ggplot()
  s <- bench$summary
  s$clock <- rownames(s)

  ggplot2::ggplot(s, ggplot2::aes(x = .data$MedAE, y = .data$MedE)) +
    .refline("h", 0) +
    ggplot2::geom_point(ggplot2::aes(size = .data$total), colour = falcon_palette()[[1]],
                        alpha = .thm("point_alpha")) +
    ggplot2::geom_text(ggplot2::aes(label = .data$clock), size = 2.3, hjust = -0.12,
                       colour = "#555555") +
    ggplot2::scale_size(range = c(1.5, 6), guide = "none") +
    .labs(.txt("benchmark_error_bias", n_clocks = nrow(s))) +
    falcon_theme()
}

#' Effect size per clock and dataset
#'
#' Read the columns. A dataset blank across every clock is either a condition
#' the clocks cannot see or a cohort too small to show it.
#'
#' @param bench Output of [run_benchmark()].
#' @return A ggplot.
#' @examples
#' \dontrun{
#' plot_benchmark_heatmap(run_benchmark(res))
#' }
#' @export
plot_benchmark_heatmap <- function(bench) {
  need_ggplot()
  d <- bench$per_dataset
  dv <- falcon_scheme()$palette$diverging
  lim <- stats::quantile(abs(d$delta), 0.95, na.rm = TRUE)

  ggplot2::ggplot(d, ggplot2::aes(x = .data$dataset, y = .data$clock,
                                  fill = .data$delta)) +
    ggplot2::geom_tile() +
    ggplot2::geom_tile(data = d[d$significant, , drop = FALSE],
                       colour = "black", linewidth = 0.5, fill = NA) +
    ggplot2::scale_fill_gradient2(low = dv$low, mid = dv$mid, high = dv$high,
                                  midpoint = 0, limits = c(-lim, lim),
                                  oob = scales_squish, name = "delta") +
    .labs(.txt("benchmark_heatmap", n_clocks = length(unique(d$clock)),
               n_datasets = length(unique(d$dataset)))) +
    falcon_theme() +
    ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 45, hjust = 1),
                   panel.grid = ggplot2::element_blank())
}


#' Survival by age acceleration
#'
#' Kaplan-Meier curves for the fastest-ageing tail against the slowest, with the
#' log-rank p-value in the subtitle.
#'
#' @section Why the tails and not a median split:
#' The middle of the acceleration distribution is where a clock discriminates
#' least, so pooling it into two halves dilutes whatever signal the tails carry.
#' The published convention is the extreme deciles.
#'
#' @param x A `falcon_result`.
#' @param clock Which clock's acceleration to stratify on.
#' @param time_col,event_col Columns of the sample annotation holding follow-up
#'   time and the event indicator (1 = event, 0 = censored).
#' @param age_col Column holding chronological age.
#' @param quantile Size of each tail; 0.1 gives top and bottom 10%.
#' @return A ggplot.
#' @examples
#' \dontrun{
#' plot_kaplan_meier(res, "horvath2013", time_col = "time", event_col = "status")
#' }
#' @export
plot_kaplan_meier <- function(x, clock, time_col, event_col,
                              age_col = "age", quantile = 0.1) {
  need_ggplot()
  # The log-rank statistic comes from the Python core, not from a second
  # implementation here: one numerical core is the whole premise, and a
  # p-value that differed between the languages would be the worst possible
  # thing to disagree about. The figure the core draws is discarded -- only its
  # summary table is used -- and the curve is redrawn in ggplot2 so it carries
  # the same theme as every other R figure.
  out <- py_do(fa()$plot$kaplan_meier(x$py, clock, time_col = time_col,
                                      event_col = event_col, age_col = age_col,
                                      quantile = quantile))
  tbl <- as_df(out[[2]])

  # Rebuild the step curves from the same strata the core used.
  aa <- acceleration(x, age_col = age_col, clocks = clock)[[clock]]
  obs <- obs(x)
  d <- data.frame(aa = aa,
                  time = suppressWarnings(as.numeric(obs[[time_col]])),
                  event = suppressWarnings(as.numeric(obs[[event_col]])))
  d <- d[stats::complete.cases(d), , drop = FALSE]
  cuts <- stats::quantile(d$aa, c(quantile, 1 - quantile), names = FALSE)

  steps <- do.call(rbind, lapply(
    list(list(sub = d[d$aa <= cuts[1], ], grp = sprintf("slowest %d%%", round(quantile * 100))),
         list(sub = d[d$aa >= cuts[2], ], grp = sprintf("fastest %d%%", round(quantile * 100)))),
    function(g) {
      km <- .km(g$sub$time, g$sub$event)
      data.frame(time = km$time, surv = km$surv, group = g$grp)
    }))

  ggplot2::ggplot(steps, ggplot2::aes(x = .data$time, y = .data$surv,
                                      colour = .data$group)) +
    ggplot2::geom_step(linewidth = .thm("line_width")) +
    ggplot2::ylim(0, 1) +
    ggplot2::scale_colour_manual(
      values = stats::setNames(c(.sem("control"), .sem("case")),
                               unique(steps$group))) +
    .labs(.txt("kaplan_meier", clock = clock, n = nrow(d),
               events = sum(d$event), p = signif(tbl$logrank_p[1], 3))) +
    falcon_theme()
}

# Product-limit estimator. Kept here rather than pulling in `survival`, which is
# a heavier dependency than twenty lines of arithmetic deserves.
.km <- function(time, event) {
  o <- order(time)
  t <- time[o]; e <- as.logical(event[o])
  times <- 0; surv <- 1; s <- 1; n <- length(t)
  for (ti in unique(t)) {
    at <- t == ti
    d <- sum(e[at])
    if (d > 0) {
      s <- s * (1 - d / n)
      times <- c(times, ti); surv <- c(surv, s)
    }
    n <- n - sum(at)
  }
  list(time = times, surv = surv)
}


#' Volcano plot of association results
#'
#' Effect size against evidence, for the table [associate()] returns.
#'
#' @section The threshold that is drawn:
#' The dashed line is the Benjamini-Hochberg cut at `fdr`, taken from the `q`
#' column rather than recomputed. Drawing a raw p-value cut instead is the
#' common error: across many tests the two differ by orders of magnitude, and
#' the raw one calls noise significant.
#'
#' @param assoc Output of [associate()].
#' @param effect,p Column names for the effect size and p-value.
#' @param fdr False-discovery rate for the significance threshold.
#' @param label_top How many of the strongest hits to label.
#' @return A ggplot.
#' @examples
#' \dontrun{
#' plot_volcano(associate(res, "mortality"))
#' }
#' @export
plot_volcano <- function(assoc, effect = "beta", p = "p", fdr = 0.05,
                         label_top = 10) {
  need_ggplot()
  d <- as.data.frame(assoc)
  if (!all(c(effect, p) %in% names(d))) {
    stop(sprintf("volcano: no '%s' column; associate() returns %s",
                 setdiff(c(effect, p), names(d))[1],
                 paste(names(d), collapse = ", ")), call. = FALSE)
  }
  d$clock <- rownames(d)
  d <- d[is.finite(d[[effect]]) & is.finite(d[[p]]), , drop = FALSE]
  d$y <- -log10(pmax(d[[p]], .Machine$double.xmin))
  d$hit <- if ("q" %in% names(d)) d$q <= fdr else d[[p]] <= fdr
  ycut <- if (any(d$hit)) min(d$y[d$hit]) else NA_real_

  g <- ggplot2::ggplot(d, ggplot2::aes(x = .data[[effect]], y = .data$y,
                                       colour = .data$hit)) +
    .refline("v", 0) +
    ggplot2::geom_point(size = .thm("point_size")) +
    ggplot2::scale_colour_manual(values = c("FALSE" = .sem("neutral"),
                                            "TRUE" = .sem("case")),
                                 guide = "none")
  if (is.finite(ycut)) g <- g + .refline("h", ycut)

  top <- utils::head(d[order(-d$y), , drop = FALSE], label_top)
  g +
    ggplot2::geom_text(data = top,
                       ggplot2::aes(label = .data$clock),
                       hjust = -0.1, vjust = -0.4, show.legend = FALSE,
                       size = .thm("caption_size") / 3) +
    .labs(.txt("volcano", n = nrow(d), hits = sum(d$hit), fdr = fdr)) +
    falcon_theme()
}
