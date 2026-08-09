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

# `download` and `score` name both a module and a function. The verb wins at
# the top level -- fa.score(data) is the API -- so the modules are bound to
# private aliases, which also keeps fa.download.cache_info reachable.
from . import analysis, core, io, models, plot, preprocess, registry, report, uncertainty
from . import download as _download_mod  # noqa: F401  (module, not the verb)
from . import score as _score_mod  # noqa: F401  (module, not the verb)
from ._version import REGISTRY_VERSION, __version__
from .analysis import (acceleration, agreement, associate, cell_composition, consensus,
                       cox_hazard, detectable_effect, icc, power, run_benchmark)
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
from .io import (list_computage_bench, read, read_bedmethyl, read_bedmethyl_dir,
                 read_betas, read_clinical, read_computage_bench, read_panel,
                 read_rrbs_dir, read_series_matrix, write_results)
from .preprocess import (apply_batch_reference, fit_batch_reference, idat_to_betas,
                         prepare, prepare_clinical, probe_loss, qc, read_idat_dir)
from .score import FalconResult, combine, score
from .uncertainty import (conformal_interval, icc_from_replicates, interval,
                          technical_se)

__all__ = [
    "FalconConfig", "FalconData", "FalconError", "FalconResult", "REGISTRY_VERSION",
    "RunManifest", "WeightsUnavailableError", "__version__", "acceleration",
    "apply_batch_reference", "cell_composition", "fit_batch_reference",
    "idat_to_betas", "read_idat_dir",
    "agreement", "analysis", "associate", "combine", "configure",
    "conformal_interval", "consensus",
    "core", "cox_hazard", "detectable_effect",
    "describe", "download", "icc", "icc_from_replicates", "interval", "io",
    "list_computage_bench", "read_computage_bench",
    "models", "plot", "power", "prepare",
    "prepare_clinical", "preprocess", "probe_loss", "qc", "read", "read_bedmethyl",
    "read_bedmethyl_dir", "read_betas",
    "report",
    "read_clinical", "read_panel",
    "read_rrbs_dir", "read_series_matrix", "registry", "run_benchmark", "score",
    "technical_se", "uncertainty",
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
