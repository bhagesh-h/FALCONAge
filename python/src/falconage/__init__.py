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
175 catalogued clocks. 46 are ``bundled``: their coefficients live inside the
wheel and they run offline. 40 are ``licensed`` scaffolds whose coefficients are
research-use-only and are not ours to distribute (:mod:`falconage.registry`
explains each one and names an open alternative). The remaining 89 are
``untraced`` -- catalogued with real metadata, awaiting a traced extractor for
the numbers. ``fa.registry.load().filter(availability="bundled")`` is the list
that works today.

Beyond the clocks
-----------------
Three modules read an aging methylome without predicting an age, because a clock
is not the only way to and on the evidence may not be the most informative one:

:mod:`falconage.disorder`
    Entropy, drift and the noise barometer. Tong et al. (Nature Aging 2024)
    showed most of a clock's accuracy against chronological age is reproducible
    by a purely stochastic model; these are the readouts of that stochastic part
    directly.
:mod:`falconage.immune`
    Repertoire structure -- the covariate blood clocks do not carry -- and a
    simulator for what clone structure alone does to a clock score.
:mod:`falconage.registry.coefficient_mass`
    Where a clock's weight sits, against any annotation you supply.
"""

from __future__ import annotations

# `download` and `score` name both a module and a function. The verb wins at
# the top level -- fa.score(data) is the API -- so the modules are bound to
# private aliases, which also keeps fa.download.cache_info reachable.
from . import (analysis, core, disorder, immune, io, models, plot, preprocess,
               registry, report, uncertainty)
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
from .download import download, download_intervention, how_to_get, interventions
from .io import (list_computage_bench, read, read_bedmethyl, read_bedmethyl_dir,
                 read_betas, read_clinical, read_computage_bench, read_panel,
                 read_rrbs_dir, read_series_matrix, write_results)
from .preprocess import (apply_batch_reference, fit_batch_reference, idat_to_betas,
                         prepare, prepare_clinical, probe_loss, qc, read_idat_dir)
from .disorder import drift, entropy, noise_barometer, variable_sites
from .immune import repertoire_diversity, simulate_clonality
from .models.single_cell import mosaic
from .registry import coefficient_mass
from .score import FalconResult, combine, score
from .uncertainty import (conformal_interval, icc_from_replicates, interval,
                          technical_se, variance_components)

__all__ = [
    "FalconConfig", "FalconData", "FalconError", "FalconResult", "REGISTRY_VERSION",
    "RunManifest", "WeightsUnavailableError", "__version__", "acceleration",
    "apply_batch_reference", "cell_composition", "fit_batch_reference",
    "idat_to_betas", "read_idat_dir",
    "agreement", "analysis", "associate", "combine", "configure",
    "conformal_interval", "consensus",
    "coefficient_mass", "core", "cox_hazard", "detectable_effect",
    "describe", "disorder", "download", "download_intervention", "drift",
    "entropy", "how_to_get", "icc",
    "icc_from_replicates", "immune", "interval", "interventions", "io",
    "list_computage_bench", "mosaic", "noise_barometer", "read_computage_bench",
    "repertoire_diversity", "simulate_clonality", "variable_sites",
    "variance_components",
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
    tiers = {t: len(reg.filter(availability=t))
             for t in ("bundled", "untraced", "licensed")}
    return {
        "falconage": __version__,
        "registry_version": reg.version,
        "registry_path": str(reg.path),
        "n_clocks": len(reg),
        "clocks_by_availability": tiers,
        "runnable_offline": tiers["bundled"],
        **describe(),
    }
