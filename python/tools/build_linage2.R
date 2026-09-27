# Run the LinAge2 authors' linAge2.R (Fong et al. 2025, npj Aging, Supplementary
# code, CC BY 4.0) to the end of its model fitting, and write what it fitted.
#
# Called by python/tools/build_linage2.py with the unpacked archive and an
# output directory. The script is run as published except: it stops before
# userDataOut() (which reformats the result and then prompts for SEQ numbers),
# and its prompt "cotinine values (C) or smoking status (S)" is answered "C",
# since the example user data carry cotinine in ng/mL.
args <- commandArgs(TRUE)
options(la2 = list(src = args[1], out = args[2], fixtures = args[3], wd = getwd()))
lines <- readLines(file.path(args[1], "linAge2.R"))
stop_at <- grep("outMat <- userDataOut()", lines, fixed = TRUE)[1]
lines <- lines[seq_len(stop_at - 1)]
lines <- sub('digiCotFlag <- readline("Have you entered cotinine values (C) or smoking status (S) ? > ")',
             'digiCotFlag <- "C"', lines, fixed = TRUE)
stopifnot(!any(grepl("^[^#]*readline\\(", lines)))
setwd(args[1])
pdf(NULL)
eval(parse(text = lines))
la2 <- getOption("la2")
setwd(la2$wd)

feats <- colnames(dataMat)[-1]
stopifnot(identical(colnames(dataMat), colnames(dataMat_user)))
lam <- unlist(boxCox_lam[1, feats])
skip <- c("fs1Score", "fs2Score", "fs3Score", "LBXCOT", "LBDBANO")
sel <- qDataMat[, "yearsNHANES"] == "9900" & qDataMat[, "RIDAGEYR"] <= 50
male <- qDataMat[sel, "RIAGENDR"] == 1
ref <- dataMat_trans[sel, feats]
norm <- lapply(feats, function(f) {
  if (f %in% skip) return(list(normalise = FALSE))
  list(normalise = TRUE, median_male = median(ref[male, f]), mad_male = mad(ref[male, f]),
       median_female = median(ref[!male, f]), mad_female = mad(ref[!male, f]))
})
names(norm) <- feats
cox <- function(full, null, V) {
  terms <- names(full$coefficients)
  pcs <- terms[terms != "chronAge"]
  idx <- as.integer(sub("PC", "", pcs))
  stopifnot(nrow(V) == length(feats))
  list(terms = terms, coefficients = unname(full$coefficients), means = unname(full$means[terms]),
       ties = full$method, null_coefficient = unname(null$coefficients[1]),
       null_mean = unname(null$means[1]),
       loadings = setNames(lapply(idx, function(k) unname(V[, k])), pcs))
}
params <- list(
  source = "Fong et al. 2025, npj Aging, Supplementary code linAge2.R (CC BY 4.0)",
  features = feats,
  lambda = lapply(feats, function(f) if (is.na(lam[[f]])) NULL else unname(lam[[f]])),
  normalisation = norm,
  z_max = zScoreMax,
  age_unit = "months",
  male = cox(coxModelM, nullModelM, vMatDat99_M),
  female = cox(coxModelF, nullModelF, vMatDat99_F)
)
names(params$lambda) <- feats
writeLines(jsonlite::toJSON(params, auto_unbox = TRUE, digits = NA, null = "null", pretty = TRUE),
           file.path(la2$out, "linage2.json"))

# Fixtures: the example user data and sanity samples as the script read them
# (cotinine still in ng/mL), and the script's LinAge2 for each, in years.
inp <- rbind(read.csv(file.path(la2$src, "userData.csv")), read.csv(file.path(la2$src, "userData_sanity.csv")))
write.csv(inp, file.path(la2$fixtures, "linage2_input.csv"), row.names = FALSE)
write.csv(data.frame(SEQN = qDataMat_user[, "SEQN"], linage2 = sprintf("%.12g", bioAge_user / 12)),
          file.path(la2$fixtures, "linage2_reference.csv"), row.names = FALSE, quote = FALSE)
# The 2001-2002 wave the authors test on, for a wider check outside the suite.
write.csv(data.frame(SEQN = demoTest[, "SEQN"], linage2 = sprintf("%.12g", bioAge / 12)),
          file.path(la2$out, "linage2_nhanes0102.csv"), row.names = FALSE, quote = FALSE)
