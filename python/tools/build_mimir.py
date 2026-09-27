#!/usr/bin/env python3
"""Extract the MetaboAge and MetaboHealth parameters from MiMIR.

MiMIR (Bizzarri et al. 2022, Bioinformatics 38:3847; DanieleBizzarri/MiMIR,
GPL-3) is the reference implementation both scores are usually computed with,
and it ships their parameters:

    PARAM_metaboAge   MetaboAge (van den Akker et al. 2020, Circ Genom Precis Med
                      13:541): the 56 measures, their BBMRI-NL means and SDs
                      on the raw and the log scale, and the 57 coefficients.
    mort_betas        MetaboHealth (Deelen et al. 2019, Nat Commun 10:3346): the
                      14 measures and their weights.
    metabo_names_translator
                      Nightingale export names and their BBMRI-NL equivalents.

Written as CSVs under ``registry/data/metabolomics/``; the values are printed
with 17 significant digits, so nothing is rounded on the way.

Usage
-----
    python python/tools/build_mimir.py            # needs Rscript
    python python/tools/build_mimir.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import build_references as br  # noqa: E402

OUT = br.ROOT / "python" / "src" / "falconage" / "registry" / "data" / "metabolomics"
RAW = ("https://raw.githubusercontent.com/DanieleBizzarri/MiMIR/"
       "1746f2d98d6d9c3c5bfe35dd42e63b99314cf909/data/")

br.SOURCES.update({
    "mimir_metaboage": {"url": RAW + "PARAM_metaboAge.rda",
                        "sha256": "00287569aa85f2f07692993075d08024d7123bcfefe0dc3a50a59c99dd7080f4"},
    "mimir_mort": {"url": RAW + "mort_betas.rda",
                   "sha256": "a2c6c29300b33e33184fe40b1256881b95a9aca417b99476b94da747365cfdb7"},
    "mimir_names": {"url": RAW + "metabo_names_translator.rda",
                    "sha256": "302b480d44f50afd03e0e7e78f47392faf3d895141916a61fba7f07b209b8a3c"},
})

_R = r"""
f <- function(x) sprintf("%%.17g", x)
e <- new.env(); load("%(age)s", envir = e); p <- e$PARAM_metaboAge
fit <- p$FIT_COEF
stopifnot(names(fit)[1] == "(Intercept)", identical(names(fit)[-1], p$MET))
write.csv(data.frame(feature_id = c("(Intercept)", p$MET), coefficient = f(fit),
                     mean = c(NA, f(p$MEAN[p$MET])), sd = c(NA, f(p$SD[p$MET])),
                     log_mean = c(NA, f(p$logMEAN[p$MET])), log_sd = c(NA, f(p$logSD[p$MET]))),
          "%(out)s/metaboage.csv", row.names = FALSE, quote = FALSE, na = "")
e <- new.env(); load("%(mort)s", envir = e); b <- e$mort_betas
write.csv(data.frame(feature_id = b$Abbreviation, coefficient = f(b$Beta_value)),
          "%(out)s/metabohealth.csv", row.names = FALSE, quote = FALSE)
e <- new.env(); load("%(names)s", envir = e); t <- e$metabo_names_translator
alt <- vapply(t$alternative_names, function(a) paste(unlist(a), collapse = "|"), "")
write.csv(data.frame(bbmri = t$BBMRI_names, alternatives = gsub(", ", "|", alt)),
          "%(out)s/nmr_names.csv", row.names = FALSE, quote = TRUE)
"""


def build(cache: Path, out: Path) -> None:
    code = _R % {"age": br.fetch("mimir_metaboage", cache).as_posix(),
                 "mort": br.fetch("mimir_mort", cache).as_posix(),
                 "names": br.fetch("mimir_names", cache).as_posix(),
                 "out": out.as_posix()}
    subprocess.run(["Rscript", "-e", code], check=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--cache", default=None)
    args = ap.parse_args(argv)
    cache = Path(args.cache) if args.cache else Path.home() / ".cache" / "falconage-references"
    if args.check:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            build(cache, Path(tmp))
            stale = [n for n in ("metaboage.csv", "metabohealth.csv", "nmr_names.csv")
                     if not (OUT / n).exists()
                     or (OUT / n).read_bytes() != (Path(tmp) / n).read_bytes()]
        if stale:
            print(f"stale: {', '.join(stale)}; run python/tools/build_mimir.py")
            return 1
        print("3 MiMIR files are current")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    build(cache, OUT)
    for n in ("metaboage.csv", "metabohealth.csv", "nmr_names.csv"):
        print(f"  wrote {n}: sha256 {hashlib.sha256((OUT / n).read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
