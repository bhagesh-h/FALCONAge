# Reference output for cAge from the authors' own predictor.
#
# Fetches cage_predictor.R and its data files from elenabernabeu/cage_bage at
# a pinned commit into a temporary directory (the repository has no licence,
# so none of it is kept here), and runs the script on synthetic betas: the
# FALCONAge adult-blood reference for cAge's CpGs, moved along the sign of
# each CpG's weight by a different amount per sample so that the predictions
# span both the linear model and the log-age model, plus noise. Every CpG is
# supplied, so the authors' training means are never used. The one change to
# their script is dropping library(tidyverse), which it loads and does not use.
#   docker run --rm -v "$PWD":/w -w /w --entrypoint Rscript <project image> \
#     python/tests/data/cage_reference.R
sha <- "301f414da3c2777998440063e44d8580677c5d1b"
base <- sprintf("https://raw.githubusercontent.com/elenabernabeu/cage_bage/%s/cage_predictor/", sha)
dir <- tempfile(); dir.create(file.path(dir, "data"), recursive = TRUE)
for (f in c("cage_predictor.R", "data/elnet_coefficients_linear.tsv", "data/elnet_coefficients_log.tsv",
            "data/cpg_meanbeta_gs20k.tsv", "data/intercept_linear.txt", "data/intercept_log.txt"))
  download.file(paste0(base, f), file.path(dir, f), quiet = TRUE)

terms <- read.csv("python/src/falconage/registry/data/coefficients/cage.csv")
terms <- terms[terms$feature_id != "(Intercept)", ]
cpg <- unique(sub("_2$", "", terms$feature_id))
w <- setNames(rep(0, length(cpg)), cpg)
lin <- terms[!grepl("_2$", terms$feature_id) & !is.na(terms$coefficient), ]
w[lin$feature_id] <- lin$coefficient
ref <- read.csv(gzfile("python/src/falconage/registry/data/references/blood_adult.csv.gz"))
m <- setNames(rep(0.5, length(cpg)), cpg)
hit <- intersect(cpg, ref$feature_id)
m[hit] <- ref$value[match(hit, ref$feature_id)]
set.seed(20260926)
shift <- c(-0.20, -0.10, -0.06, -0.035, -0.02, 0, 0.01, 0.02)
betas <- sapply(shift, function(s) pmin(pmax(m + s * sign(w) + rnorm(length(m), 0, 0.01), 0.001), 0.999))
dimnames(betas) <- list(cpg, sprintf("S%d", seq_along(shift)))
write.csv(data.frame(cpg = cpg, apply(betas, 2, function(v) sprintf("%.6f", v))),
          gzfile("python/tests/data/cage_betas.csv.gz"), row.names = FALSE, quote = FALSE)

src <- readLines(file.path(dir, "cage_predictor.R"))
src <- src[src != "library(tidyverse)"]
src <- sub('^methylationTable <- ""', sprintf('methylationTable <- "%s"', file.path(dir, "betas.csv")), src)
src <- sub('^methylationTable_format <- ""', 'methylationTable_format <- "csv"', src)
betas6 <- read.csv(gzfile("python/tests/data/cage_betas.csv.gz"), row.names = 1)
write.csv(betas6, file.path(dir, "betas.csv"))
old <- setwd(dir); eval(parse(text = src)); setwd(old)
out <- read.delim(file.path(dir, "cage_predictions.tsv"))
out <- out[match(colnames(betas6), out$Sample), ]
write.csv(data.frame(sample = out$Sample, cage = sprintf("%.12g", out$Predicted_Age)),
          "python/tests/data/cage_reference.csv", row.names = FALSE, quote = FALSE)
print(out)
