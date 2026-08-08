"""Logging and verbosity.

Three levels rather than the usual five, matching the R side's
``options(falconage.verbose = "none" | "inform" | "debug")`` so that a script
translated between the languages behaves the same way.

Warnings raised during a run are also *collected*, not only printed. A run that
imputed 40% of a clock's features and said so in a line that scrolled past is
indistinguishable, three months later, from one that did not -- so every warning
also lands in the run manifest next to the number it affected.
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field

_LEVELS = {"none": logging.ERROR, "inform": logging.INFO, "debug": logging.DEBUG}

_logger = logging.getLogger("falconage")


@dataclass
class WarningCollector:
    """Accumulates warnings for the run manifest as well as the console."""

    records: list[dict[str, str]] = field(default_factory=list)

    def warn(self, message: str, *, clock: str | None = None,
             category: str = "general") -> None:
        self.records.append({"category": category, "clock": clock or "", "message": message})
        _logger.warning(message if clock is None else f"[{clock}] {message}")

    def extend(self, other: WarningCollector) -> None:
        self.records.extend(other.records)

    def __len__(self) -> int:
        return len(self.records)


def configure(verbose: str | None = None) -> None:
    """Set the console verbosity. ``None`` reads ``FALCONAGE_VERBOSE``."""
    level = (verbose or os.environ.get("FALCONAGE_VERBOSE") or "inform").lower()
    if level not in _LEVELS:
        raise ValueError(f"verbose must be one of {sorted(_LEVELS)}, got {level!r}")
    _logger.setLevel(_LEVELS[level])
    if not _logger.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("%(levelname)-7s %(message)s"))
        _logger.addHandler(h)
    _logger.propagate = False


def get_logger(name: str | None = None) -> logging.Logger:
    configure() if not _logger.handlers else None
    return _logger if name is None else _logger.getChild(name)


@contextmanager
def quiet():
    """Silence FALCONAge for the duration of the block."""
    old = _logger.level
    _logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        _logger.setLevel(old)
