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


def add_clock(path, *, replace: bool = False) -> str:
    """Add a clock from a YAML entry and its coefficient file, for this session.

    Convenience wrapper over ``load().add_clock``; see that method for the
    entry format. ``falconage run --clocks-dir`` does the same for a folder.
    """
    return load().add_clock(path, replace=replace)
