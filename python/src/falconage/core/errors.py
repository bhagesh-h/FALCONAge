"""The exception hierarchy.

Every failure FALCONAge raises deliberately descends from :class:`FalconError`,
so a caller can wrap a whole pipeline in one ``except`` and still let genuine
programming errors (``KeyError``, ``AttributeError``) escape and be fixed.

The messages carry the remedy, not just the diagnosis. A user who hits
``WeightsUnavailableError`` should leave the traceback knowing which open clock
to use instead; a user who hits ``UnitsNotDeclaredError`` should leave it
knowing which two strings are accepted.
"""

from __future__ import annotations


class FalconError(Exception):
    """Base for every error FALCONAge raises on purpose."""


# ---------------------------------------------------------------------------
# input and units
# ---------------------------------------------------------------------------
class DataError(FalconError):
    """The input data cannot be used as given."""


class UnitsNotDeclaredError(DataError):
    """Units were required and not supplied.

    Raised rather than guessed. Albumin at 4.2 is g/dL and albumin at 42 is g/L,
    and PhenoAge fitted on one returns nonsense for the other -- but both values
    are inside the plausible range for *some* unit, so no heuristic can separate
    them. The published clinical clocks disagree about units between papers, and
    silently picking one is the field's most common wrong answer.
    """


class UnitConversionError(DataError):
    """A conversion between two declared units is not defined."""


class PlatformError(DataError):
    """The methylation platform could not be determined, or is not supported."""


class FeatureCoverageError(DataError):
    """A clock has too few of its features present to return a number."""


# ---------------------------------------------------------------------------
# registry and weights
# ---------------------------------------------------------------------------
class RegistryError(FalconError):
    """The clock registry is malformed, or does not contain what was asked for."""


class ClockNotFoundError(RegistryError):
    """No clock with that identifier."""

    def __init__(self, clock_id: str, suggestions: list[str] | None = None) -> None:
        msg = f"no clock with id {clock_id!r}"
        if suggestions:
            msg += "\n\n  did you mean: " + ", ".join(suggestions)
        msg += "\n\n  falconage.registry.load().list()  lists every id"
        super().__init__(msg)
        self.clock_id = clock_id


class WeightsUnavailableError(RegistryError):
    """A clock's coefficients are not present and cannot be fetched.

    The three tiers behave differently here, and the message says which one
    applies:

    * tier A never raises -- the coefficients ship inside the wheel;
    * tier B raises only when the fetch fails, and names the URL;
    * tier C always raises until the user registers a file, and names both
      where to obtain one and which open clocks answer the same question.
    """

    def __init__(self, clock_id: str, message: str) -> None:
        super().__init__(message)
        self.clock_id = clock_id


# ---------------------------------------------------------------------------
# compute
# ---------------------------------------------------------------------------
class DeviceError(FalconError):
    """The requested compute device is unavailable or unusable."""


class ScoringError(FalconError):
    """A clock could not be evaluated."""


class AnalysisError(FalconError):
    """A downstream statistic cannot be computed from what was given."""


class IllegalOperationError(AnalysisError):
    """An operation the clock's scale does not support.

    Age acceleration is a residual against chronological age. That is defined
    for a clock whose output is an age in years, undefined for one whose output
    is a mortality log-hazard, and actively misleading for a pace of aging,
    which is already a rate. The registry records each clock's ``scale_type``
    and this is what enforces it.
    """


# ---------------------------------------------------------------------------
# network
# ---------------------------------------------------------------------------
class DownloadError(FalconError):
    """A remote resource could not be retrieved."""


class ChecksumMismatchError(DownloadError):
    """A downloaded file does not match its published digest."""
