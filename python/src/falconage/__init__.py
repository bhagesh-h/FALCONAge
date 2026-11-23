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
68 human clocks: 62 DNA methylation, four clinical chemistry and two NMR
metabolomics. Every one is ``bundled``: its coefficients live inside the wheel
and it scores offline. Clocks of your own join the same registry through
:func:`falconage.add_clock`.

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
                       cox_hazard, detectable_effect, icc, leave_one_marker_out, power,
                       run_benchmark)
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
from .io import (list_computage_bench, read, read_betas, read_clinical,
                 read_computage_bench, read_nightingale, read_panel,
                 read_series_matrix, write_results)
from .preprocess import prepare, prepare_clinical, probe_loss, qc
from .disorder import drift, entropy, noise_barometer, variable_sites
from .immune import repertoire_diversity, simulate_clonality
from .models.single_cell import mosaic
from .analysis.elasticnet import fit_clock
from .analysis.learners import compare_learners
from .analysis.stacking import stack_clocks
from .registry import add_clock
from .demo import demo_data
from .models.clinical import blood_count_ratios
from .registry import coefficient_mass
from .score import FalconResult, combine, score
from .uncertainty import (conformal_interval, icc_from_replicates, interval,
                          technical_se, variance_components)

__all__ = [
    "FalconConfig", "FalconData", "FalconError", "FalconResult", "REGISTRY_VERSION",
    "RunManifest", "WeightsUnavailableError", "__version__", "acceleration",
    "add_clock", "cell_composition", "compare_learners", "demo_data",
    "agreement", "analysis", "associate", "combine", "configure",
    "conformal_interval", "consensus",
    "coefficient_mass", "core", "cox_hazard", "detectable_effect",
    "describe", "disorder", "download", "download_intervention", "drift",
    "entropy", "how_to_get", "icc",
    "icc_from_replicates", "immune", "interval", "interventions", "io",
    "blood_count_ratios", "fit_clock", "leave_one_marker_out",
    "list_computage_bench", "mosaic", "noise_barometer", "read_computage_bench",
    "repertoire_diversity", "simulate_clonality", "variable_sites",
    "variance_components",
    "models", "plot", "power", "prepare",
    "prepare_clinical", "preprocess", "probe_loss", "qc", "read", "read_betas",
    "report",
    "read_clinical", "read_nightingale", "read_panel",
    "read_series_matrix", "registry", "run_benchmark", "score",
    "stack_clocks",
    "technical_se", "uncertainty",
    "write_results",
]


def config() -> dict:
    """What this installation resolved to: versions, registry size.

    Backs ``falconage config`` and ``FALCONAge::falconage_config()``; the same
    dict crosses the reticulate bridge, so both languages report identically.
    """
    reg = registry.load()
    shipped = [c for c in reg if c.origin == "registry"]
    tiers = {t: sum(c.availability == t for c in shipped)
             for t in ("bundled", "untraced", "licensed")}
    return {
        "falconage": __version__,
        "registry_version": reg.version,
        "registry_path": str(reg.path),
        "n_clocks": len(shipped),
        "user_clocks": sorted(c.id for c in reg if c.origin == "user"),
        "clocks_by_availability": tiers,
        "runnable_offline": tiers["bundled"],
        **describe(),
    }
