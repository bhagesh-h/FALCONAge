"""Array backend: numpy on the CPU, in a declared floating-point precision.

Every model's forward pass is written against the array-module handle this
returns, so precision is chosen in one place and recorded in the manifest.

WHY FP64 IS THE DEFAULT. Two of the published preprocessing chains are
ill-conditioned enough to move a reported age in single precision -- the PC
rotations most of all, where a 78,464-dimensional centred projection
accumulates error across the sum. The registry flags those clocks
``requires_fp64`` and this module refuses to downgrade them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .errors import ConfigError

DType = Literal["float64", "float32"]


@dataclass(frozen=True)
class ComputeSpec:
    """The precision one clock is computed in, plus the array module that serves it."""

    dtype: str = "float64"

    #: Kept as attributes so a manifest written by this version reads the same
    #: way as one written by any other: everything runs in numpy on the CPU.
    device: str = "cpu"
    backend: str = "numpy"

    def __str__(self) -> str:  # pragma: no cover - display only
        return self.dtype

    def xp(self):
        """The array module to compute with."""
        return np

    def asarray(self, a: Any) -> np.ndarray:
        """A numpy array at this precision."""
        return np.asarray(a, dtype=np.float64 if self.dtype == "float64" else np.float32)

    def tonumpy(self, a: Any) -> np.ndarray:
        return np.asarray(a)


def resolve(dtype: DType | None = None, *, requires_fp64: bool = False) -> ComputeSpec:
    """Pick the precision, and say no clearly when the request is impossible.

    Parameters
    ----------
    dtype
        ``None`` means float64, or ``FALCONAGE_DTYPE`` when it is set.
    requires_fp64
        Set from the clock's registry entry. Overrides a float32 request, with a
        warning, because the alternative is returning a number that is wrong in
        the second decimal and saying nothing.
    """
    want = dtype or os.environ.get("FALCONAGE_DTYPE") or "float64"
    if want not in ("float64", "float32"):
        raise ConfigError(f"unknown dtype {want!r}; expected float64 or float32")
    if requires_fp64 and want != "float64":
        import warnings

        warnings.warn(
            "this clock is flagged requires_fp64 in the registry; the float32 "
            "request was overridden. Its preprocessing is ill-conditioned enough "
            "that single precision moves the reported value.",
            stacklevel=2,
        )
        want = "float64"
    return ComputeSpec(dtype=want)


def describe() -> dict[str, Any]:
    """What this machine provides. Backs ``falconage config``."""
    return {"numpy": np.__version__, "cpu_count": os.cpu_count()}
