# Reference output for falconage.preprocess.horvath, from Horvath's own code.
#
# Runs BMIQcalibration from Horvath 2013 Genome Biology Additional file 24
# (fetched from the publisher and checked by SHA-256, then sourced unmodified)
# against goldstandard2 (Additional file 22, via FALCONAge's shipped copy) on
# three synthetic samples: the gold standard pushed through a different
# logit-scale compression and shift for each, plus noise, so that each has to
# be pulled back. Also records set.seed(1); sample(...) as the code draws it.
# Needs RPMM and network access:
#   docker run --rm -v "$PWD":/w -w /w --entrypoint Rscript <project image with
#     install.packages("RPMM")> python/tests/data/horvath_norm_reference.R
library(RPMM)
url <- paste0("https://static-content.springer.com/esm/art%3A10.1186%2Fgb-2013-14-10-r115/",
              "MediaObjects/13059_2013_3156_MOESM24_ESM.txt")
src <- tempfile(fileext = ".R")
download.file(url, src, quiet = TRUE, headers = c("User-Agent" = "Mozilla/5.0"))
sha <- system2("sha256sum", src, stdout = TRUE)
stopifnot(startsWith(sha, "1f51e692e3663672fde5fd11de20962c440385458b88c3ea5a10b424f632b5b6"))
printFlush <- function(...) invisible(NULL)     # a WGCNA helper the file calls
source(src)

gs <- read.csv(gzfile("python/src/falconage/registry/data/references/horvath2013_goldstandard2.csv.gz"))
gold <- gs$value
set.seed(20260926)
shape <- data.frame(slope = c(0.80, 0.90, 1.10), shift = c(0.30, -0.20, 0.10), sd = c(0.15, 0.25, 0.20))
datM <- t(sapply(seq_len(nrow(shape)), function(i) {
  z <- qlogis(pmin(pmax(gold, 1e-4), 1 - 1e-4))
  round(plogis(shape$shift[i] + shape$slope[i] * z + rnorm(length(z), 0, shape$sd[i])), 6)
}))
colnames(datM) <- gs$feature_id
rownames(datM) <- sprintf("S%d", seq_len(nrow(datM)))
write.csv(datM, gzfile("python/tests/data/horvath_norm_input.csv.gz"), quote = FALSE)

set.seed(1)
idx <- sample(1:length(gold), min(c(20000, length(gold))), replace = FALSE)
writeLines(as.character(idx), gzfile("python/tests/data/horvath_norm_sample.txt.gz"))

datIn <- as.matrix(read.csv(gzfile("python/tests/data/horvath_norm_input.csv.gz"), row.names = 1))
norm <- BMIQcalibration(datM = datIn, goldstandard.beta = gold, plots = FALSE)
write.csv(apply(norm, 2, function(v) sprintf("%.12g", v)), gzfile("python/tests/data/horvath_norm_output.csv.gz"),
          quote = FALSE)
cat(sprintf("RPMM %s\n", packageVersion("RPMM")))
