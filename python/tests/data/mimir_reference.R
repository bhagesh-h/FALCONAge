# Reference output for falconage.models.metabolomics, from MiMIR's own code.
#
# MiMIR (DanieleBizzarri/MiMIR at 1746f2d, GPL-3) is fetched when this runs:
# its R functions, its parameters, and its synthetic metabolic dataset, the
# first 200 samples of which are the input (they carry zeros and missing
# values, so QC removes some). Run in the project image:
#   docker run --rm -v "$PWD":/w -w /w --entrypoint Rscript \
#     bhagesh/falconage:1.0.0-cpu python/tests/data/mimir_reference.R
base <- "https://raw.githubusercontent.com/DanieleBizzarri/MiMIR/1746f2d98d6d9c3c5bfe35dd42e63b99314cf909/"
get <- function(p) { f <- tempfile(); download.file(paste0(base, p), f, quiet = TRUE, mode = "wb"); f }
source(get("R/utils.R")); source(get("R/predictors_functions.R"))
for (r in c("PARAM_metaboAge", "mort_betas", "synthetic_metabolic_dataset")) load(get(paste0("data/", r, ".rda")))

need <- union(PARAM_metaboAge$MET, mort_betas$Abbreviation)
x <- as.matrix(synthetic_metabolic_dataset[1:200, need])
write.csv(x, "python/tests/data/mimir_input.csv")

qc <- QCprep(x, PARAM_metaboAge, quiet = TRUE)
age <- apply.fit(qc, FIT = PARAM_metaboAge$FIT_COEF)
write.csv(data.frame(sample = rownames(age), metaboage = sprintf("%.15g", age$MetaboAge)),
          "python/tests/data/mimir_metaboage.csv", row.names = FALSE, quote = FALSE)
mort <- comp.mort_score(as.data.frame(x), betas = mort_betas, quiet = TRUE)
write.csv(data.frame(sample = rownames(mort), metabohealth = sprintf("%.15g", mort$mortScore)),
          "python/tests/data/mimir_metabohealth.csv", row.names = FALSE, quote = FALSE)
cat(nrow(x), "in,", nrow(age), "past QC\n")
