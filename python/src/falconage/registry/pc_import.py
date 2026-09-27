"""Register the PC clocks from the authors' own data file.

WHAT THE FILE IS. Higgins-Chen et al. 2022 (Nature Aging 2:644,
doi:10.1038/s43587-022-00248-2) rebuilt six clocks on principal components of
78,464 CpGs. Their scoring code (MorganLevineLab/PC-Clocks,
``run_calcPCClocks.R``) reads everything from one file, ``CalcAllPCClocks.RData``,
which the repository points to on Yale Box. Neither carries a licence, so the
file is not redistributed here; this reads the copy a user downloads.

WHAT THE AUTHORS' CODE DOES, AND WHAT THIS REPRODUCES. A CpG the data lacks
entirely is filled from ``imputeMissingCpGs``; any other missing value takes
its CpG's mean in the data; then each clock is
``(x - center) %*% rotation %*% model + intercept``, passed through
``anti.trafo`` for the two Horvath clocks. That is linear in ``x``, so it
collapses without approximation to one weight per CpG,
``w = rotation %*% model``, and a constant ``intercept - center . w``. The
collapsed weights go in as the clock's coefficients, the constant as their
``(Intercept)`` row, and ``imputeMissingCpGs`` as its reference values, which
FALCONAge's alignment uses exactly as the authors' code does: for absent CpGs
only, with each present CpG's own mean for its odd missing value.

``anti.trafo`` is saved inside the file rather than in the scripts, so it is
read from there and checked against Horvath's transform with adult age 20,
which the registry declares for these two clocks. A file whose transform is
anything else is refused.

PCGrimAge is not imported. It is eight sub-models on the components plus sex
and age, combined by a ninth, and FALCONAge has no composite model class yet.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from ..core.errors import RegistryError

#: Object in CalcAllPCClocks.RData -> registry id.
CLOCKS = {
    "CalcPCHorvath1": "pchorvath2013",
    "CalcPCHorvath2": "pcskinandblood",
    "CalcPCHannum": "pchannum",
    "CalcPCPhenoAge": "pcphenoage",
    "CalcPCDNAmTL": "pcdnamtl",
}

#: Horvath's anti.trafo (2013, Additional file 20), whitespace and braces
#: removed, since R deparses a one-line function without them.
HORVATH_ANTI_TRAFO = ("function(x,adult.age=20)ifelse(x<0,(1+adult.age)*exp(x)-1,"
                      "(1+adult.age)*x+adult.age)")

_R = r"""
e <- new.env(); load("%(path)s", envir = e)
need <- c(%(objects)s, "CpGs", "imputeMissingCpGs", "anti.trafo")
gone <- setdiff(need, ls(e))
if (length(gone)) stop("not in the file: ", paste(gone, collapse = ", "))
out <- "%(out)s"
cat(deparse(e$anti.trafo), sep = "\n", file = file.path(out, "anti_trafo.txt"))
cpgs <- as.character(e$CpGs)
ids <- c(%(pairs)s)
for (k in names(ids)) {
  o <- e[[k]]
  w <- as.vector(as.matrix(o$rotation) %%*%% as.numeric(o$model))
  if (length(w) != length(cpgs) || length(o$center) != length(cpgs))
    stop(k, ": rotation or center does not match the ", length(cpgs), " CpGs")
  b <- as.numeric(o$intercept) - sum(as.numeric(o$center) * w)
  write.csv(data.frame(feature_id = c("(Intercept)", cpgs),
                       coefficient = sprintf("%%.17g", c(b, w))),
            file.path(out, paste0(ids[[k]], ".csv")), row.names = FALSE, quote = FALSE)
}
ref <- e$imputeMissingCpGs[cpgs]
ok <- !is.na(ref)
write.csv(data.frame(feature_id = cpgs[ok], value = sprintf("%%.17g", ref[ok])),
          file.path(out, "reference.csv"), row.names = FALSE, quote = FALSE)
"""


def import_pc_clocks(registry, path, out_dir=None) -> dict[str, str]:
    """Read ``CalcAllPCClocks.RData`` and register the five single-model PC clocks.

    Returns clock id -> SHA-256 of the coefficient file written, which the run
    manifest records as user-supplied. Needs ``Rscript`` (the FALCONAge image
    has it) because the file is R's own format.
    """
    src = Path(path).expanduser()
    if not src.exists():
        raise RegistryError(f"no such file: {src}")
    if shutil.which("Rscript") is None:
        raise RegistryError(
            "reading CalcAllPCClocks.RData needs R (Rscript on the PATH); the "
            "FALCONAge Docker image carries it")
    out = Path(out_dir).expanduser() if out_dir else src.parent / "falconage_pc_clocks"
    out.mkdir(parents=True, exist_ok=True)

    code = _R % {
        "path": src.as_posix(), "out": out.as_posix(),
        "objects": ", ".join(f'"{k}"' for k in CLOCKS),
        "pairs": ", ".join(f'"{k}" = "{v}"' for k, v in CLOCKS.items()),
    }
    run = subprocess.run(["Rscript", "-e", code], capture_output=True, text=True)
    if run.returncode != 0:
        raise RegistryError(f"{src.name}: {run.stderr.strip().splitlines()[-1]}")

    trafo = re.sub(r"[\s{}]+", "", (out / "anti_trafo.txt").read_text())
    if trafo != HORVATH_ANTI_TRAFO:
        raise RegistryError(
            f"{src.name}: anti.trafo is not Horvath's transform with adult age 20, "
            f"which the registry applies to pchorvath2013 and pcskinandblood:\n  {trafo}")

    registered = {}
    for cid in CLOCKS.values():
        registered[cid] = registry.register_local_weights(cid, out / f"{cid}.csv")
        registry.register_local_reference(cid, out / "reference.csv")
    return registered
