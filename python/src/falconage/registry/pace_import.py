"""Register DunedinPACE from a user's copy of the authors' R package.

The weights, the 20,000-probe background and its reference means live in the
package's ``mPACE_Models`` object (danbelsky/DunedinPACE). The package is GPL-3
but its README restricts the algorithm to research use and names an exclusive
commercial licensee, so FALCONAge ships none of it. This reads it from wherever
the user has it:

* an installed package directory (``.../library/DunedinPACE``), where R keeps
  internal data in ``R/sysdata.rdb`` and reads it with ``lazyLoad``;
* a source checkout, which has ``R/sysdata.rda``;
* that ``.rda`` file itself.

Scoring then follows ``PACEProjector`` exactly: :mod:`falconage.models.pace`.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pandas as pd

from ..core.errors import RegistryError

_R = r"""
p <- "%(path)s"; e <- new.env()
if (dir.exists(p)) {
  if (file.exists(file.path(p, "R", "sysdata.rdb"))) {
    lazyLoad(file.path(p, "R", "sysdata"), envir = e)
  } else if (file.exists(file.path(p, "R", "sysdata.rda"))) {
    load(file.path(p, "R", "sysdata.rda"), envir = e)
  } else stop("no R/sysdata.rdb or R/sysdata.rda under ", p)
} else load(p, envir = e)
if (!exists("mPACE_Models", envir = e)) stop("no mPACE_Models in ", p)
m <- get("mPACE_Models", envir = e)
k <- "%(model)s"
if (!(k %%in%% m$model_names)) stop("no model ", k, "; the file has ", paste(m$model_names, collapse = ", "))
out <- "%(out)s"
w <- m$model_weights[[k]]; probes <- m$model_probes[[k]]
if (!identical(names(w), probes)) w <- w[probes]
write.csv(data.frame(feature_id = c("(Intercept)", probes),
                     coefficient = sprintf("%%.17g", c(m$model_intercept[[k]], w))),
          file.path(out, "weights.csv"), row.names = FALSE, quote = FALSE)
g <- m$gold_standard_means[[k]][m$gold_standard_probes[[k]]]
write.csv(data.frame(feature_id = names(g), value = sprintf("%%.17g", g)),
          file.path(out, "gold_standard.csv"), row.names = FALSE, quote = FALSE)
mm <- m$model_means[[k]]
write.csv(data.frame(feature_id = names(mm), value = sprintf("%%.17g", mm)),
          file.path(out, "model_means.csv"), row.names = FALSE, quote = FALSE)
"""


def import_dunedinpace(registry, path, out_dir=None, *, clock_id: str = "dunedinpace",
                       model: str = "DunedinPACE") -> str:
    """Read ``mPACE_Models`` and register the clock. Returns the SHA-256 of the
    weight file, which the run manifest records as user-supplied."""
    src = Path(path).expanduser()
    if not src.exists():
        raise RegistryError(f"no such file or directory: {src}")
    if shutil.which("Rscript") is None:
        raise RegistryError("reading the DunedinPACE package needs R (Rscript on the "
                            "PATH); the FALCONAge Docker image carries it")
    # Not beside the package: an installed R library is often read-only.
    out = (Path(out_dir).expanduser() if out_dir
           else Path.home() / ".cache" / "falconage" / "dunedinpace")
    out.mkdir(parents=True, exist_ok=True)
    code = _R % {"path": src.as_posix(), "out": out.as_posix(), "model": model}
    run = subprocess.run(["Rscript", "-e", code], capture_output=True, text=True)
    if run.returncode != 0:
        raise RegistryError(f"{src.name}: {run.stderr.strip().splitlines()[-1]}")

    gold = pd.read_csv(out / "gold_standard.csv", index_col=0)["value"].astype(float)
    means = pd.read_csv(out / "model_means.csv", index_col=0)["value"].astype(float)
    digest = registry.register_local_weights(clock_id, out / "weights.csv")
    feats, _ = registry.coefficients(clock_id)
    if not set(feats) <= set(gold.index):
        raise RegistryError(f"{src.name}: some model probes are not in the background set")
    registry._local_extra[clock_id] = {"gold_means": gold, "model_means": means}
    return digest
