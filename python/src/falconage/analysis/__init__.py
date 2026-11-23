"""Downstream statistics: acceleration, association, reliability, benchmarking.

The rule that runs through all of it: a clock's ``scale_type`` decides which
operations are defined. Age acceleration is a residual against chronological
age; it means something for a clock that outputs years, nothing for one that
outputs a log-hazard, and something actively misleading for a pace of aging,
which is already a rate. :class:`~falconage.core.errors.IllegalOperationError`
is raised rather than computed, because the alternative is a number that looks
like every other number in the table.
"""

from __future__ import annotations

from ._common import (  # noqa: F401
    _check_legal,
    _bh,
)
from .acceleration import (  # noqa: F401
    cell_composition,
    _regress_out,
    acceleration,
    _residual,
    _level,
    _strata,
    _reference_acceleration,
)
from .association import (  # noqa: F401
    associate,
    cox_hazard,
    _cox_breslow,
    _cox_newton,
    FULL_PANEL,
    _marker_frame,
    _numeric_covariates,
    leave_one_marker_out,
)
from .reliability import (  # noqa: F401
    icc,
    pool_icc,
    agreement,
)
from .benchmark import (  # noqa: F401
    BenchmarkResult,
    run_benchmark,
    _rank_effects,
)
from .design import (  # noqa: F401
    PowerResult,
    power,
    detectable_effect,
    _OUTCOME_TRAINED,
    ConsensusReport,
    _paired_test,
    _mixed_test,
    consensus,
)
from .learners import ComparisonResult, compare_learners  # noqa: F401
from .stacking import StackFit, stack_clocks  # noqa: F401

__all__ = [
    "BenchmarkResult",
    "ComparisonResult",
    "StackFit",
    "compare_learners",
    "stack_clocks",
    "ConsensusReport",
    "PowerResult",
    "acceleration",
    "agreement",
    "associate",
    "consensus",
    "cox_hazard",
    "detectable_effect",
    "icc",
    "leave_one_marker_out",
    "power",
    "run_benchmark",
]
