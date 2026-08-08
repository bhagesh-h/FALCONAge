# =============================================================================
# The pooled atlas, drawn in R
# =============================================================================
#
# The Python side composes six matplotlib axes on a shared GridSpec; here the
# same six panels are ggplots joined with patchwork. They are not the same
# drawing code -- they cannot be -- but they are the same six panels, in the
# same order, from the same numbers, with title, subtitle and caption read from
# the same colorscheme.yaml entry. That is the level at which the two languages
# are meant to agree for figures; only the clock SCORES are bit-identical.
# =============================================================================

#' Clock atlas: every algorithm across every pooled study
#'
#' The one figure that answers "of these twenty-odd algorithms, which are
#' measuring aging at all". Rows are clocks on a shared vertical axis; panels
#' are, left to right: type badge, MedAE, signed MedE, per-study detection dots,
#' AA2/AA1 counts, and mean coverage.
#'
#' @section How to read it:
#' Panel D, down the page. A row of hollow circles on zero is an algorithm that
#' detected nothing in any cohort; filled dots pushed right are the ones that
#' did, and several in a row means the effect held across studies rather than in
#' one. Panels B and C are diagnostics for reading D, not scores -- a clock that
#' merely returned chronological age would be perfect in B, empty in D, and
#' useless for every purpose anyone scores a clock for. Panel F separates the
#' two ways of finding nothing: a clock that saw its features and found no
#' difference, from one that never had the features to look with.
#'
#' @section Optional by design:
#' It needs several studies to say anything, so it is not part of the default
#' figure set and errors below `min_datasets`.
#'
#' @param x A combined `falcon_result`, normally from [combine()].
#' @param bench The matching output of [run_benchmark()].
#' @param dataset_col Column naming the study.
#' @param min_datasets Refuse below this many studies. Two is the floor at which
#'   "consistent across cohorts" means anything.
#' @param coverage_floor Drawn on panel F; use the value the run was scored with.
#' @param max_clocks Keep the highest-scoring this many, so a full catalogue
#'   still fits a page.
#'
#' @return A patchwork of ggplots when \pkg{patchwork} is installed, otherwise a
#'   named list of the six panels, so the figure remains usable without it.
#' @examples
#' \dontrun{
#' res <- combine(list(r1, r2, r3))
#' b   <- run_benchmark(res, dataset_col = "dataset")
#' plot_clock_atlas(res, b)
#' }
#' @export
plot_clock_atlas <- function(x, bench, dataset_col = "dataset", min_datasets = 2L,
                             coverage_floor = 0.8, max_clocks = 40L) {
  need_ggplot()
  o <- obs(x)
  if (!dataset_col %in% names(o)) {
    stop("the atlas pools studies and needs a '", dataset_col, "' column naming ",
         "them. combine() adds one.", call. = FALSE)
  }
  datasets <- sort(unique(as.character(o[[dataset_col]])))
  if (length(datasets) < min_datasets) {
    stop("clock_atlas: ", length(datasets), " study(ies) pooled, need at least ",
         min_datasets, ". This figure compares clocks ACROSS cohorts; with one ",
         "cohort it is the per-study panels drawn twice.", call. = FALSE)
  }

  per <- bench$per_dataset
  s <- bench$summary
  if (!nrow(per)) {
    stop("clock_atlas: the benchmark produced no comparisons", call. = FALSE)
  }

  scored <- colnames(as.data.frame(x))
  s <- s[rownames(s) %in% scored, , drop = FALSE]
  if (!nrow(s)) {
    stop("clock_atlas: no scored clock appears in the benchmark", call. = FALSE)
  }

  # Ascending, so the best clock ends up at the top once the discrete y scale
  # runs bottom to top. Accuracy breaks ties.
  ord <- rownames(s)[order(s$total, s$AA2, -s$MedAE)]
  if (length(ord) > max_clocks) ord <- utils::tail(ord, max_clocks)

  covdf <- coverage(x)
  covmean <- vapply(ord, function(cl) {
    v <- covdf$coverage[covdf$clock == cl]
    if (length(v)) mean(v, na.rm = TRUE) else NA_real_
  }, numeric(1))

  scales <- vapply(ord, function(cl) clock_info_quiet(cl)$scale_type, character(1))

  d <- data.frame(
    clock = factor(ord, levels = ord),
    scale_type = unname(scales),
    MedAE = s[ord, "MedAE"], MedE = s[ord, "MedE"],
    AA2 = s[ord, "AA2"], AA1 = s[ord, "AA1"],
    coverage = as.numeric(covmean),
    stringsAsFactors = FALSE)

  per <- per[per$clock %in% ord, , drop = FALSE]
  per$clock <- factor(per$clock, levels = ord)
  per$significant <- as.character(per$significant)
  ccol <- .group_cols(sort(unique(as.character(per$condition))))

  ttl <- .txt("clock_atlas", n_clocks = length(ord), n_datasets = length(datasets),
              n_samples = nrow(as.data.frame(x)),
              n_significant = sum(bench$per_dataset$significant),
              n_comparisons = nrow(bench$per_dataset))

  # Only the leftmost panel keeps its clock labels; the other five share the
  # axis, so repeating the names would be noise five times over.
  bare_y <- ggplot2::theme(axis.text.y = ggplot2::element_blank(),
                           axis.title.y = ggplot2::element_blank())
  panel_title <- function(txt) {
    ggplot2::theme(plot.title = ggplot2::element_text(
      size = .thm("base_size") - 0.5, colour = "#333333", hjust = 0))
  }

  p_type <- ggplot2::ggplot(d, ggplot2::aes(x = 1, y = .data$clock,
                                            fill = .data$scale_type)) +
    ggplot2::geom_tile(width = 0.9, height = 0.7) +
    ggplot2::scale_fill_manual(values = .group_cols(sort(unique(d$scale_type)))) +
    ggplot2::labs(title = "A  type", x = NULL, y = NULL) +
    falcon_theme() + panel_title() +
    ggplot2::theme(axis.text.x = ggplot2::element_blank(),
                   panel.grid = ggplot2::element_blank(),
                   legend.position = "bottom",
                   legend.text = ggplot2::element_text(size = .thm("caption_size") - 1))

  p_medae <- ggplot2::ggplot(d, ggplot2::aes(x = .data$MedAE, y = .data$clock)) +
    ggplot2::geom_col(fill = .sem("neutral"), alpha = 0.55, width = 0.62) +
    ggplot2::labs(title = "B  MedAE (years)", x = NULL, y = NULL) +
    falcon_theme() + panel_title() + bare_y

  p_mede <- ggplot2::ggplot(d, ggplot2::aes(x = .data$MedE, y = .data$clock,
                                            fill = .data$MedE > 0)) +
    ggplot2::geom_col(alpha = 0.78, width = 0.62) +
    .refline("v", 0) +
    ggplot2::scale_fill_manual(values = c("FALSE" = .sem("decelerated"),
                                          "TRUE" = .sem("accelerated")),
                               guide = "none") +
    ggplot2::labs(title = "C  MedE, signed", x = NULL, y = NULL) +
    falcon_theme() + panel_title() + bare_y

  p_detect <- ggplot2::ggplot(per, ggplot2::aes(x = .data$delta, y = .data$clock,
                                                colour = .data$condition,
                                                shape = .data$significant)) +
    .refline("v", 0) +
    ggplot2::geom_point(position = ggplot2::position_dodge(width = 0.6),
                        size = 2.2, stroke = 0.9) +
    # Filled for significant, hollow for not. The most important distinction in
    # the figure, and one of the few that survives greyscale printing.
    ggplot2::scale_shape_manual(values = c("FALSE" = 1, "TRUE" = 19), guide = "none") +
    ggplot2::scale_colour_manual(values = ccol) +
    ggplot2::labs(
      title = "D  case - control acceleration, one dot per study (filled = q < 0.05)",
      x = "difference in median acceleration (years)", y = NULL) +
    falcon_theme() + panel_title() + bare_y +
    ggplot2::theme(legend.position = "bottom",
                   legend.text = ggplot2::element_text(size = .thm("caption_size") - 1))

  bars <- rbind(
    data.frame(clock = d$clock, test = "AA2", n = d$AA2),
    data.frame(clock = d$clock, test = "AA1", n = d$AA1))
  p_bench <- ggplot2::ggplot(bars, ggplot2::aes(x = .data$n, y = .data$clock,
                                                fill = .data$test)) +
    ggplot2::geom_col(width = 0.62) +
    ggplot2::scale_fill_manual(values = falcon_palette()[1:2]) +
    ggplot2::labs(title = "E  datasets hit", x = NULL, y = NULL) +
    falcon_theme() + panel_title() + bare_y +
    ggplot2::theme(legend.position = "bottom",
                   legend.text = ggplot2::element_text(size = .thm("caption_size") - 1))

  d$band <- cut(d$coverage, c(-Inf, coverage_floor, 0.95, Inf),
                labels = c("below floor", "marginal", "usable"))
  p_cov <- ggplot2::ggplot(d, ggplot2::aes(x = .data$coverage, y = .data$clock,
                                           fill = .data$band)) +
    ggplot2::geom_col(width = 0.62, alpha = 0.85) +
    .refline("v", coverage_floor) +
    ggplot2::scale_fill_manual(values = c("below floor" = .sem("fail"),
                                          "marginal" = .sem("warn"),
                                          "usable" = .sem("pass")), guide = "none") +
    ggplot2::xlim(0, 1) +
    ggplot2::labs(title = "F  coverage", x = NULL, y = NULL) +
    falcon_theme() + panel_title() + bare_y

  panels <- list(type = p_type, medae = p_medae, mede = p_mede,
                 detect = p_detect, bench = p_bench, coverage = p_cov)

  if (!requireNamespace("patchwork", quietly = TRUE)) {
    message("install.packages('patchwork') to get the six panels composed into ",
            "one figure; returning them as a list.")
    return(panels)
  }

  # Panel D is widest because it carries the answer.
  patchwork::wrap_plots(panels, nrow = 1,
                        widths = c(0.5, 1, 1, 3.25, 0.92, 0.92)) +
    patchwork::plot_annotation(
      title = ttl$title, subtitle = ttl$subtitle, caption = ttl$caption,
      theme = ggplot2::theme(
        plot.title = ggplot2::element_text(size = .thm("title_size") + 2, hjust = 0),
        plot.subtitle = ggplot2::element_text(size = .thm("subtitle_size"),
                                              colour = "#555555", hjust = 0),
        plot.caption = ggplot2::element_text(size = .thm("caption_size"),
                                             colour = "#666666", hjust = 0)))
}
