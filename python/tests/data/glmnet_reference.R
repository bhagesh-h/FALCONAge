# Reference output for falconage.analysis.elasticnet, from glmnet itself.
#
# A synthetic problem (80 samples, 30 features, 8 of them informative), a tight
# convergence threshold so the comparison measures the algorithm and not the
# stopping rule, and cv.glmnet with fixed fold assignments. Needs glmnet:
#   docker run --rm -v "$PWD":/w -w /w --entrypoint Rscript <project image with
#     install.packages("glmnet")> python/tests/data/glmnet_reference.R
library(glmnet)
# Global, so cv.glmnet's fold fits get it too; its control argument does not
# reach them.
glmnet.control(thresh = 1e-14, maxit = 1e7)
set.seed(20260926)
n <- 80; p <- 30
X <- matrix(rnorm(n * p), n, p, dimnames = list(NULL, sprintf("f%02d", 1:p)))
X[, 5] <- X[, 5] * 4 + 2                       # a feature on another scale
y <- 40 + X[, 1:8] %*% c(3, -2, 1.5, 1, -1, 0.5, 0.5, -0.5) + rnorm(n, 0, 2)
foldid <- rep(1:5, length.out = n)[sample(n)]
write.csv(data.frame(X, y = y, foldid = foldid), "python/tests/data/glmnet_input.csv", row.names = FALSE)

fit <- glmnet(X, y, alpha = 0.5)
B <- as.matrix(coef(fit))
write.csv(data.frame(lambda = sprintf("%.17g", fit$lambda), t(apply(B, 2, function(v) sprintf("%.17g", v)))),
          "python/tests/data/glmnet_path.csv", row.names = FALSE, quote = FALSE)
# Cross-validation from glmnet's own fold fits, aggregated by cv.glmnet's rules
# (cvcompute): the fold MSEs weighted by fold size, their standard error with
# nfolds - 1, lambda.min the largest lambda at the minimum, lambda.1se the
# largest within one SE of it. cv.glmnet's own fold fits in 5.1 differ from
# direct glmnet() calls on the same rows by about 1e-4 in MSE; its choices are
# recorded beside these, not compared.
lam <- fit$lambda
mse <- t(sapply(1:5, function(k) {
  tr <- foldid != k
  f <- glmnet(X[tr, ], y[tr], alpha = 0.5, lambda = lam)
  colMeans((y[!tr] - predict(f, X[!tr, ], s = lam))^2)
}))
w <- as.numeric(table(foldid))
cvm <- colSums(w * mse) / sum(w)
cvsd <- sqrt(colSums(w * sweep(mse, 2, cvm)^2) / sum(w) / (5 - 1))
lmin <- max(lam[cvm <= min(cvm)])
l1se <- max(lam[cvm <= (cvm + cvsd)[lam == lmin]])
write.csv(data.frame(lambda = sprintf("%.17g", lam), cvm = sprintf("%.17g", cvm),
                     cvsd = sprintf("%.17g", cvsd)),
          "python/tests/data/glmnet_cv.csv", row.names = FALSE, quote = FALSE)
cv <- cv.glmnet(X, y, alpha = 0.5, foldid = foldid)
writeLines(c(sprintf("lambda_min,%.17g", lmin), sprintf("lambda_1se,%.17g", l1se),
             sprintf("cvglmnet_lambda_min,%.17g", cv$lambda.min),
             sprintf("cvglmnet_lambda_1se,%.17g", cv$lambda.1se),
             sprintf("glmnet,%s", packageVersion("glmnet"))), "python/tests/data/glmnet_choice.csv")
