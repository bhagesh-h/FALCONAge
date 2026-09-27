# Reference output for falconage.analysis.mixed, from nlme::lme (REML).
#
# 30 people, three visits, about 10% of visits missing, sex constant within a
# person. Two models: y ~ visit (every fixed effect varies within a person)
# and y ~ visit + sex (sex is constant within a person, so it takes the outer
# containment df). Run in the project image:
#   docker run --rm -v "$PWD":/w -w /w --entrypoint Rscript \
#     bhagesh/falconage:1.0.0-cpu python/tests/data/lmm_nlme.R
library(nlme)
set.seed(20260926)
G <- 30
subject <- rep(sprintf("P%02d", seq_len(G)), each = 3)
visit <- rep(c("v1", "v2", "v3"), times = G)
sex <- rep(ifelse(seq_len(G) %% 2 == 0, "F", "M"), each = 3)
b <- rep(rnorm(G, 0, 5), each = 3)
y <- 50 + b + c(v1 = 0, v2 = 1.0, v3 = 0.5)[visit] + ifelse(sex == "F", 2, 0) + rnorm(3 * G, 0, 2)
d <- data.frame(subject, visit, sex, y = round(y, 6))
d <- d[-sample(nrow(d), 9), ]
write.csv(d, "python/tests/data/lmm_nlme_data.csv", row.names = FALSE, quote = FALSE)

out <- NULL
for (f in c("y ~ visit", "y ~ visit + sex")) {
  m <- lme(as.formula(f), random = ~ 1 | subject, data = d, method = "REML")
  tt <- summary(m)$tTable
  a <- anova(m)
  out <- rbind(out, data.frame(
    model = f, term = c(rownames(tt), "sigma", "tau", "logLik", "F_visit", "p_visit", "df_visit"),
    estimate = c(tt[, "Value"], m$sigma, as.numeric(VarCorr(m)[1, "StdDev"]), as.numeric(logLik(m)),
                 a["visit", "F-value"], a["visit", "p-value"], a["visit", "denDF"]),
    se = c(tt[, "Std.Error"], rep(NA, 6)),
    df = c(tt[, "DF"], rep(NA, 6)),
    p = c(tt[, "p-value"], rep(NA, 6))))
}
write.csv(out, "python/tests/data/lmm_nlme_reference.csv", row.names = FALSE, quote = TRUE)
cat(format(packageVersion("nlme")), "\n")
