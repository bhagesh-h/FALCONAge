"""Model classes and the shared operation catalogue.

DEVICE CONTRACT. Every model's ``predict`` is handed the run's
:class:`~falconage.core.backend.DeviceSpec`. A class that cannot use it says so
with ``CPU_ONLY = True`` rather than accepting the argument and ignoring it,
because the caller records what ran where and an unread argument makes that
record a fiction. :func:`effective_spec` is how the caller asks.
"""

from . import aggregation, clinical, division, neural, ops, pc, single_cell
from .aggregation import AggregationClock, is_aggregation, parse_statistic
from .clinical import ClinicalClock, HDReference, KDMReference, fit_hd, fit_kdm
from .division import DivisionClock, is_division_model, read_division_parameters
from .linear import Alignment, LinearClock, ScaffoldClock, align, build
from .neural import NeuralClock, NeuralWeights, read_neural_weights
from .pc import PCLinearClock, PCRotation, read_rotation
from .single_cell import ScAgeReference, fit_scage_reference, scage


def effective_spec(model, spec):
    """The device and precision ``model`` will actually compute in.

    Almost always ``spec`` unchanged. The exception is a model class that
    declares ``CPU_ONLY``: it is handed the run's spec for interface uniformity
    and computes in numpy regardless, so the honest answer is ``spec.as_cpu()``.

    Two clocks in one run can therefore differ, and that is not a bug to
    smooth over. Requesting ``device="cuda"`` on a mixed set puts the linear
    clocks on the card and leaves PhenoAge in numpy, and the manifest has to be
    able to say so per clock.
    """
    return spec.as_cpu() if getattr(model, "CPU_ONLY", False) else spec


__all__ = [
    "AggregationClock", "Alignment", "ClinicalClock", "DivisionClock",
    "HDReference",
    "KDMReference", "LinearClock", "NeuralClock", "NeuralWeights",
    "PCLinearClock", "PCRotation", "ScAgeReference", "ScaffoldClock",
    "aggregation", "align", "division",
    "build", "clinical", "effective_spec",
    "fit_hd", "fit_kdm", "fit_scage_reference", "is_aggregation", "neural",
    "is_division_model", "ops", "parse_statistic", "pc",
    "read_division_parameters",
    "read_neural_weights", "read_rotation", "scage", "single_cell",
]
