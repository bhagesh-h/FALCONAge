"""FALCONAge -- multiomic biological age and aging clock scoring.

One numerical core, two languages. The R package wraps this one through
reticulate, so an R result and a Python result are the same bits rather than
two implementations that agree to six decimals.

Quick start
-----------
>>> import falconage as fa
>>> data = fa.read("betas.csv")
>>> res  = fa.score(data, clocks="compatible")
>>> res.summary()

What ships
----------
161 catalogued clocks. 22 of them carry coefficients inside the wheel and run
offline; 28 are scaffolds whose coefficients are research-use-only and are not
ours to distribute (:mod:`falconage.registry` explains each one and names an
open alternative); the rest are catalogued with metadata and await a traced
extractor. ``fa.registry.load().filter(availability="A")`` is the list that
works today.
"""

from __future__ import annotations

from . import analysis, core, download as _download_mod, io, models, plot, preprocess, registry, score as _score_mod
from ._version import REGISTRY_VERSION, __version__
from .analysis import acceleration, agreement, associate, cox_hazard, icc, run_benchmark
from .core import (
    FalconConfig,
    FalconData,
    FalconError,
    RunManifest,
    WeightsUnavailableError,
    configure,
    describe,
)
from .download import download
from .io import read, read_betas, read_clinical, read_rrbs_dir, read_series_matrix, write_results
from .preprocess import prepare, prepare_clinical, qc
from .score import FalconResult, combine, score

__all__ = [
    "FalconConfig", "FalconData", "FalconError", "FalconResult", "REGISTRY_VERSION",
    "RunManifest", "WeightsUnavailableError", "__version__", "acceleration",
    "agreement", "analysis", "associate", "combine", "configure", "core", "cox_hazard",
    "describe", "download", "icc", "io", "models", "plot", "prepare",
    "prepare_clinical", "preprocess", "qc", "read", "read_betas", "read_clinical",
    "read_rrbs_dir", "read_series_matrix", "registry", "run_benchmark", "score",
    "write_results",
]


def config() -> dict:
    """What this installation resolved to: versions, devices, registry size.

    Backs ``falconage config`` and ``FALCONAge::falconage_config()``; the same
    dict crosses the reticulate bridge, so both languages report identically.
    """
    reg = registry.load()
    tiers = {t: len(reg.filter(availability=t)) for t in ("A", "B", "C")}
    return {
        "falconage": __version__,
        "registry_version": reg.version,
        "registry_path": str(reg.path),
        "n_clocks": len(reg),
        "clocks_by_tier": tiers,
        "runnable_offline": tiers["A"],
        **describe(),
    }
