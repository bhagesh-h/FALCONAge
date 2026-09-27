# Reference output for four bundled linear clocks, from the authors' own code.
#
#   leecontrol, leerobust, leerefinedrobust: planet::predictAge (Bioconductor
#     planet, maintained by the Robinson lab, whose Lee et al. 2019 paper
#     defines the three placental clocks), types CPC, RPC and RRPC.
#   hrsinchphenoage: calcHRSInChPhenoAge from methylCIPHER, the first author's
#     laboratory package, at commit bfe5d02 (fetched below).
#
# The input is synthetic: every CpG the four clocks use, 6 samples of betas
# drawn uniformly on (0.05, 0.95). Needs planet and network access:
#   docker run --rm -v "$PWD":/w -w /w --entrypoint Rscript <project image with
#     BiocManager::install("planet")> python/tests/data/authors_linear_reference.R
suppressMessages(library(planet))
sha <- "bfe5d02817e7dd5e923e8bf96528d5bfd52e1636"
raw <- function(p) sprintf("https://raw.githubusercontent.com/HigginsChenLab/methylCIPHER/%s/%s", sha, p)
HRSInCHPhenoAge_CpGs <- read.csv(raw("data-raw/HRSInChPhenoAge_CpG.csv"))[, c("CpG", "Weight")]
source(raw("R/calcHRSInChPhenoAge.R"))

data(ageCpGs, package = "planet")
cpgs <- sort(unique(c(setdiff(ageCpGs$CpGs, "(Intercept)"), HRSInCHPhenoAge_CpGs$CpG)))
set.seed(20260926)
betas <- matrix(runif(length(cpgs) * 6, 0.05, 0.95), nrow = length(cpgs),
                dimnames = list(cpgs, sprintf("S%02d", 1:6)))
out <- data.frame(
  sample = colnames(betas),
  leecontrol = predictAge(betas, type = "CPC"),
  leerobust = predictAge(betas, type = "RPC"),
  leerefinedrobust = predictAge(betas, type = "RRPC"),
  hrsinchphenoage = calcHRSInChPhenoAge(t(betas))
)
write.csv(data.frame(cpg = cpgs, apply(betas, 2, function(v) sprintf("%.17g", v))),
          gzfile("python/tests/data/authors_linear_betas.csv.gz"), row.names = FALSE, quote = FALSE)
write.csv(data.frame(sample = out$sample, lapply(out[-1], function(v) sprintf("%.17g", v))),
          "python/tests/data/authors_linear_reference.csv", row.names = FALSE, quote = FALSE)
writeLines(sprintf("planet %s", packageVersion("planet")))
