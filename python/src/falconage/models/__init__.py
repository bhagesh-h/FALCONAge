"""Model classes and the shared operation catalogue.

Every model's ``predict`` is handed the run's
:class:`~falconage.core.backend.ComputeSpec`, which fixes the precision.
"""

from . import aggregation, clinical, division, neural, ops, pc, single_cell
from .aggregation import AggregationClock, is_aggregation, parse_statistic
from .clinical import ClinicalClock, HDReference, KDMReference, fit_hd, fit_kdm
from .division import DivisionClock, is_division_model, read_division_parameters
from .linear import Alignment, LinearClock, ScaffoldClock, align, build
from .neural import NeuralClock, NeuralWeights, read_neural_weights
from .pc import PCLinearClock, PCRotation, read_rotation
from .single_cell import ScAgeReference, fit_scage_reference, scage



__all__ = [
    "AggregationClock", "Alignment", "ClinicalClock", "DivisionClock",
    "HDReference",
    "KDMReference", "LinearClock", "NeuralClock", "NeuralWeights",
    "PCLinearClock", "PCRotation", "ScAgeReference", "ScaffoldClock",
    "aggregation", "align", "division",
    "build", "clinical",
    "fit_hd", "fit_kdm", "fit_scage_reference", "is_aggregation", "neural",
    "is_division_model", "ops", "parse_statistic", "pc",
    "read_division_parameters",
    "read_neural_weights", "read_rotation", "scage", "single_cell",
]
