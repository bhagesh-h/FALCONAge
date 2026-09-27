# Reference output for falconage.models.pace, from the authors' own code.
#
# PACEProjector.R is fetched from danbelsky/DunedinPACE at 4b56998 when this
# runs and is not stored here; the model it scores is synthetic, with the
# layout of mPACE_Models (400 background probes, 30 model probes), so nothing
# of the licensed model is either. preprocessCore (LGPL) must be installed:
#   docker run --rm -v "$PWD":/w -w /w --entrypoint Rscript <project image with
#     BiocManager::install("preprocessCore")> python/tests/data/pace_reference.R
set.seed(20260926)
src <- "https://raw.githubusercontent.com/danbelsky/DunedinPACE/4b569983543e51d1022aecec9a25e694bb3a336a/R/PACEProjector.R"
tmp <- tempfile(fileext = ".R"); download.file(src, tmp, quiet = TRUE)
source(tmp)

gold <- sprintf("cg%08d", 1:400)
probes <- gold[seq(5, 400, by = 13)][1:30]
gm <- setNames(round(runif(400, 0.02, 0.98), 4), gold)
mPACE_Models <- list(
  model_names = "DunedinPACE",
  gold_standard_probes = list(DunedinPACE = gold),
  gold_standard_means = list(DunedinPACE = gm),
  model_probes = list(DunedinPACE = probes),
  model_weights = list(DunedinPACE = setNames(rnorm(30, 0, 0.1), probes)),
  model_intercept = list(DunedinPACE = -1.9),
  model_means = list(DunedinPACE = setNames(round(runif(30, 0.1, 0.9), 4), probes)))
save(mPACE_Models, file = "python/tests/data/pace_model.rda")

# 10 samples; values on a 0.01 grid so ties are common.
b <- matrix(round(pmin(pmax(outer(gm, rep(1, 10)) + rnorm(4000, 0, 0.05), 0.001), 0.999), 2),
            400, 10, dimnames = list(gold, sprintf("S%02d", 1:10)))
b <- b[-c(2, 40, 41, 150, 399, 57, 70), ]           # absent: background and model probes
b[c("cg00000011", "cg00000200"), "S03"] <- NA          # partly missing: cohort mean
b[c("cg00000031", "cg00000044"), c("S01", "S04", "S06")] <- NA   # 30% missing
b[sample(nrow(b), 90), "S09"] <- NA                    # S09 misses 22%: dropped
write.csv(b, "python/tests/data/pace_betas.csv")
out <- PACEProjector(b)$DunedinPACE
write.csv(data.frame(sample = names(out), score = sprintf("%.15g", out)),
          "python/tests/data/pace_reference.csv", row.names = FALSE, quote = FALSE)

# normalize.quantiles.use.target on its own: both branches.
m <- matrix(round(runif(300), 1), 50, 6)               # heavy ties
m[c(3, 17), 6] <- NA                                    # the interpolation branch
tg <- runif(50)
t64 <- runif(64)                                        # a target of another length
r1 <- preprocessCore::normalize.quantiles.use.target(m, tg)
r2 <- preprocessCore::normalize.quantiles.use.target(m[, 1:5], t64)
write.csv(m, "python/tests/data/qnorm_input.csv", row.names = FALSE)
write.csv(data.frame(t50 = sprintf("%.17g", tg)), "python/tests/data/qnorm_target50.csv", row.names = FALSE, quote = FALSE)
write.csv(data.frame(t64 = sprintf("%.17g", t64)), "python/tests/data/qnorm_target64.csv", row.names = FALSE, quote = FALSE)
write.csv(format(r1, digits = 17), "python/tests/data/qnorm_out50.csv", row.names = FALSE, quote = FALSE)
write.csv(format(r2, digits = 17), "python/tests/data/qnorm_out64.csv", row.names = FALSE, quote = FALSE)
