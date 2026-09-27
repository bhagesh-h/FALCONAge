# Reference output for five clocks traced in item 17, from the authors' own code.
#
# Input, for a clock whose CpGs sorted are c_1 < ... < c_n, samples s = 0, 1, 2:
#   beta[i, s] = 0.05 + 0.9 * (((i + 1) * 37 + (s + 1) * 101) %% 997) / 996,  i = 0..n-1
# test_catalogue_clocks.py rebuilds the same matrix.
#
#   bohlin        predictGA() from JonBohlin/predictGA @ 242801d (GPL >= 2), its
#                 default se = TRUE (lambda.1se); gestational age in days
#   stoch, stocz, stocp
#                 RunStochClocks() and glmStocALL.Rd, Tong et al. 2024 Supplementary
#                 Software (GPL-3); library(EpiDISH) is not loaded, as it is used only
#                 when a reference matrix is passed and none is
#   intrinclock   predict(model, x) and returnAge() as in the authors' demo, on their
#                 model object (Zenodo 10.5281/zenodo.10426597), here from the identical
#                 copy methylCIPHER @ bfe5d02 (BSD-3) ships
# Needs glmnet and network access:
#   docker run --rm -v "$PWD":/w -w /w --entrypoint Rscript <project image with
#     install.packages("glmnet")> python/tests/data/catalogue_clocks_reference.R
suppressMessages(library(glmnet))
fmt <- function(x) formatC(x, digits = 17, format = "g")
get <- function(url, dest) download.file(url, dest, mode = "wb", quiet = TRUE,
                                         headers = c("User-Agent" = "falconage-tests"))
synth <- function(cpgs) {
  cpgs <- sort(cpgs); n <- length(cpgs)
  m <- sapply(0:2, function(s) 0.05 + 0.9 * ((((0:(n - 1)) + 1) * 37 + (s + 1) * 101) %% 997) / 996)
  dimnames(m) <- list(cpgs, c("S0", "S1", "S2")); m
}
tmp <- tempfile(); dir.create(tmp)
out <- list()

gh <- "https://raw.githubusercontent.com/JonBohlin/predictGA/242801dc3ef1cc49570f6602565d0f62db344f10/R/"
for (f in c("sysdata.rda", "extractSites.R", "predictGA.R")) get(paste0(gh, f), file.path(tmp, f))
load(file.path(tmp, "sysdata.rda")); source(file.path(tmp, "extractSites.R")); source(file.path(tmp, "predictGA.R"))
ga <- predictGA(t(synth(extractSites(type = "se"))))
out$bohlin <- data.frame(clock = "bohlin", sample = rownames(ga), value = fmt(ga$GA), unit = "days")

zip <- file.path(tmp, "stoc.zip")
get(paste0("https://static-content.springer.com/esm/art%3A10.1038%2Fs43587-024-00600-8/",
           "MediaObjects/43587_2024_600_MOESM3_ESM.zip"), zip)
unzip(zip, exdir = file.path(tmp, "stoc"))
d <- dirname(list.files(file.path(tmp, "stoc"), "RunStochClocks.R", recursive = TRUE, full.names = TRUE)[1])
src <- readLines(file.path(d, "RunStochClocks.R"))
src <- src[!grepl("^library\\(EpiDISH\\)", src)]
old <- setwd(d); eval(parse(text = src))
e <- new.env(); load("glmStocALL.Rd", envir = e); L <- e$glmStocALL.lo
for (k in names(L)) {
  x <- synth(rownames(coef(L[[k]]))[-1])
  r <- RunStochClocks(x)$mage[[match(k, names(L))]]
  out[[k]] <- data.frame(clock = paste0("stoc", tolower(k)), sample = colnames(x), value = fmt(r), unit = "years")
}
setwd(old)

get(paste0("https://raw.githubusercontent.com/HigginsChenLab/methylCIPHER/",
           "bfe5d02817e7dd5e923e8bf96528d5bfd52e1636/data/IntrinClock_data.rda"), file.path(tmp, "ic.rda"))
e <- new.env(); load(file.path(tmp, "ic.rda"), envir = e); m <- e$IntrinClock_data$model
returnAge <- function(ages) { for (a in seq_along(ages)) { if (ages[a] >= 0) ages[a] <- 21 * ages[a] + 20 else ages[a] <- 21 * exp(ages[a]) - 1 }; ages }
cf <- as.matrix(coef(m)); used <- rownames(cf)[cf[, 1] != 0][-1]
x <- synth(used)
full <- matrix(0.5, nrow = 3, ncol = nrow(cf) - 1, dimnames = list(colnames(x), rownames(cf)[-1]))
full[, used] <- t(x)          # the zero-weight CpGs take 0.5 and do not enter
out$intrinclock <- data.frame(clock = "intrinclock", sample = colnames(x),
                              value = fmt(returnAge(as.numeric(predict(m, full)))), unit = "years")

res <- do.call(rbind, out)
res$value <- trimws(res$value)
write.csv(res, "python/tests/data/catalogue_clocks_reference.csv", row.names = FALSE, quote = FALSE)
print(res, row.names = FALSE)
