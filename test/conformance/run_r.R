# Stage 2a: methylclock, dnaMethyAge and methylCIPHER on the matrix prepare.py wrote.
# Writes out/r_packages.csv (package, clock, name, sample, value) and out/r_errors.csv.
# Each package is called as its documentation shows, with imputation and
# normalisation off: the input is complete, and normalisation is compared elsewhere.
suppressMessages({library(yaml)})
here <- "test/conformance"
cfg <- yaml::read_yaml(file.path(here, "pairs.yaml"))$clocks
betas <- as.matrix(read.csv(file.path(here, "out", "betas.csv"), row.names = 1, check.names = FALSE))
pheno <- read.csv(file.path(here, "out", "pheno.csv"))
rows <- list(); errs <- list()
add <- function(pkg, cid, name, v) {
  if (length(v) != ncol(betas)) stop(sprintf("%d values for %d samples", length(v), ncol(betas)))
  rows[[length(rows) + 1]] <<- data.frame(package = pkg, clock = cid, name = name,
                                          sample = colnames(betas), value = as.numeric(v))
}
err <- function(pkg, cid, name, e) {
  errs[[length(errs) + 1]] <<- data.frame(package = pkg, clock = cid, name = name,
                                          error = substr(conditionMessage(e), 1, 300))
}
want <- function(pkg) Filter(function(x) !is.null(x[[pkg]]), cfg)

# methylclock: one call gives every clock it has, as columns
mc <- want("methylclock")
if (length(mc) && requireNamespace("methylclock", quietly = TRUE)) {
  df <- data.frame(ProbeID = rownames(betas), betas, check.names = FALSE)
  out <- tryCatch(suppressMessages(suppressWarnings(
    methylclock::DNAmAge(df, cell.count = FALSE, normalize = FALSE, fastImp = FALSE))), error = function(e) e)
  for (cid in names(mc)) {
    nm <- mc[[cid]]$methylclock
    if (inherits(out, "error")) { err("methylclock", cid, nm, out); next }
    if (!nm %in% colnames(out)) { err("methylclock", cid, nm, simpleError("no such column")); next }
    add("methylclock", cid, nm, out[[nm]][match(colnames(betas), out$id)])
  }
}

# dnaMethyAge: one clock per call
dm <- want("dnaMethyAge")
if (length(dm) && requireNamespace("dnaMethyAge", quietly = TRUE)) {
  # methyAge() loads its coefficients with data(), which finds them only when
  # the package is attached.
  suppressMessages(library(dnaMethyAge))
  args <- names(formals(dnaMethyAge::methyAge))
  for (cid in names(dm)) {
    nm <- dm[[cid]]$dnaMethyAge
    r <- tryCatch({
      a <- list(betas = betas, clock = nm)
      if ("do_plot" %in% args) a$do_plot <- FALSE
      if ("inputation" %in% args) a$inputation <- FALSE
      # simple_mode skips Horvath's normalisation for the Horvath clocks (compared
      # separately); for epiTOC2 it selects a simplified estimate, so it stays off there.
      if ("simple_mode" %in% args) a$simple_mode <- nm %in% c("HorvathS2013", "HorvathS2018")
      o <- suppressMessages(suppressWarnings(do.call(dnaMethyAge::methyAge, a)))
      o$mAge[match(colnames(betas), o$Sample)]
    }, error = function(e) e)
    if (inherits(r, "error")) err("dnaMethyAge", cid, nm, r) else add("dnaMethyAge", cid, nm, r)
  }
}

# methylCIPHER: samples in rows
cp <- want("methylCIPHER")
if (length(cp) && requireNamespace("methylCIPHER", quietly = TRUE)) {
  suppressMessages(library(methylCIPHER))
  D <- t(betas)
  for (cid in names(cp)) {
    nm <- cp[[cid]]$methylCIPHER
    r <- tryCatch({
      f <- get(nm, envir = asNamespace("methylCIPHER"))
      a <- list(DNAm = D)
      fa <- names(formals(f))
      if ("imputation" %in% fa) a$imputation <- FALSE
      o <- suppressMessages(suppressWarnings(do.call(f, a)))
      if (is.data.frame(o)) o <- o[[ncol(o)]]
      as.numeric(o)
    }, error = function(e) e)
    if (inherits(r, "error")) err("methylCIPHER", cid, nm, r) else add("methylCIPHER", cid, nm, r)
  }
}

write.csv(do.call(rbind, rows), file.path(here, "out", "r_packages.csv"), row.names = FALSE)
e <- if (length(errs)) do.call(rbind, errs) else data.frame(package = character(), clock = character(), name = character(), error = character())
write.csv(e, file.path(here, "out", "r_errors.csv"), row.names = FALSE)
v <- function(p) if (requireNamespace(p, quietly = TRUE)) as.character(packageVersion(p)) else "absent"
cat(sprintf("methylclock %s, dnaMethyAge %s, methylCIPHER %s: %d scores, %d errors\n",
            v("methylclock"), v("dnaMethyAge"), v("methylCIPHER"),
            sum(vapply(rows, nrow, 1L)), length(errs)))
