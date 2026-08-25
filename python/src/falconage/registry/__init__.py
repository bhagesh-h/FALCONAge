"""Clock catalogue, availability tiers and coefficient resolution."""

from .annotate import coefficient_mass
from .registry import (LEGAL_OPS, Clock, ClockRegistry, CoefficientSource, evidence,
                       load)

__all__ = ["LEGAL_OPS", "Clock", "ClockRegistry", "CoefficientSource",
           "coefficient_mass", "evidence", "load"]


def register_local_weights(clock_id: str, path, sha256: str | None = None) -> str:
    """Supply a coefficient file for a clock FALCONAge does not distribute.

    Convenience wrapper over ``load().register_local_weights`` so the call reads
    the same in both languages:

    >>> import falconage as fa
    >>> fa.registry.register_local_weights("grimage2", "~/licensed/grimage2.csv")
    """
    return load().register_local_weights(clock_id, path, sha256)
