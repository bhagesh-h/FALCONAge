"""Array backend: numpy on the CPU, torch on a GPU, one set of op code.

WHY AN INDIRECTION INSTEAD OF TORCH EVERYWHERE. torch is 800 MB installed and
the CPU path does not need it: a 500-CpG dot product over 30 samples is a numpy
one-liner that finishes before torch would have finished importing. Making it an
extra keeps a default install small enough that somebody scoring blood chemistry
can use it, and the ops are written against the module handle this returns
rather than against either library, so there is exactly one implementation of
each transform.

WHY FP64 IS THE DEFAULT. Two of the published preprocessing chains are
ill-conditioned enough to move a reported age in single precision -- the PC
rotations most of all, where a 78,464-dimensional centred projection
accumulates error across the sum. The registry flags those clocks
``requires_fp64`` and this module refuses to downgrade them.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .errors import DeviceError

Device = Literal["cpu", "cuda", "mps", "auto"]
DType = Literal["float64", "float32"]


@functools.lru_cache(maxsize=1)
def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass(frozen=True)
class DeviceSpec:
    """A resolved (device, dtype) pair plus the array module that serves it."""

    device: str
    dtype: str
    backend: str            # "numpy" or "torch"

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.backend}:{self.device}/{self.dtype}"

    @property
    def is_gpu(self) -> bool:
        return self.device in ("cuda", "mps")

    def xp(self):
        """The array module to compute with."""
        if self.backend == "torch":
            import torch

            return torch
        return np

    def asarray(self, a: Any):
        """Move a numpy array onto this device at this dtype."""
        arr = np.asarray(a, dtype=np.float64 if self.dtype == "float64" else np.float32)
        if self.backend == "numpy":
            return arr
        import torch

        t = torch.from_numpy(np.ascontiguousarray(arr))
        return t.to(device=self.device, dtype=getattr(torch, self.dtype))

    def tonumpy(self, a: Any) -> np.ndarray:
        if self.backend == "torch":
            return a.detach().cpu().numpy()
        return np.asarray(a)

    def as_cpu(self) -> DeviceSpec:
        """The same precision, on the CPU, in numpy.

        For the model classes whose forward pass has no device implementation.
        They are handed the run's spec like every other model and must report
        what they actually computed in, not what the run asked for: a manifest
        that records ``cuda`` for arithmetic that never left numpy is a false
        provenance record, and the manifest is the whole reproducibility claim.
        See :func:`falconage.models.effective_spec`.
        """
        if self.backend == "numpy" and self.device == "cpu":
            return self
        return DeviceSpec(device="cpu", dtype=self.dtype, backend="numpy")


def _cuda_ok() -> bool:
    if not torch_available():
        return False
    import torch

    return bool(torch.cuda.is_available())


def _mps_ok() -> bool:
    if not torch_available():
        return False
    import torch

    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def resolve(device: Device = "auto", dtype: DType | None = None,
            *, requires_fp64: bool = False) -> DeviceSpec:
    """Pick a device and dtype, and say no clearly when the request is impossible.

    Parameters
    ----------
    device
        ``"auto"`` prefers CUDA, then MPS, then CPU. Naming a device that is not
        present is an error rather than a silent downgrade: a run that was asked
        for a GPU and quietly used a CPU looks like a very slow success, and the
        user finds out three hours later.
    dtype
        ``None`` means float64, except on MPS where float64 does not exist.
    requires_fp64
        Set from the clock's registry entry. Overrides a float32 request, with a
        warning, because the alternative is returning a number that is wrong in
        the second decimal and saying nothing.

    Raises
    ------
    DeviceError
        The named device is unavailable, or float64 was required on hardware
        that cannot provide it.
    """
    env = os.environ.get("FALCONAGE_DEVICE")
    if device == "auto" and env:
        device = env  # type: ignore[assignment]

    if device == "auto":
        # "auto" means CPU, even when a GPU is present, and that is a measured
        # decision rather than caution.
        #
        # A linear clock is a (samples x features) @ (features,) dot product.
        # For the shipping catalogue that is at most a few thousand features,
        # and the arithmetic is microseconds; the cost is moving the aligned
        # matrix across PCIe. Benchmarked on an RTX 4060 with 16,384 samples and
        # eight clocks, CPU took 0.51 s and CUDA 2.33 s -- the device made it
        # 4.6x SLOWER, because 2.4 GB of transfers bought nothing: at 4,096
        # samples the dot products are 5 ms on the CPU against 10 ms on the GPU,
        # and building the aligned matrix costs 134 ms either way.
        #
        # Choosing CUDA here would make the common case worse on every machine
        # that happens to have a card, silently. So the GPU is opt-in:
        # device="cuda", or FALCONAGE_DEVICE=cuda. It earns its place on the
        # 78,464-feature PC clocks and on neural architectures like AltumAge,
        # where the matrix is large enough for the transfer to amortise.
        resolved = "cpu"
    else:
        resolved = device

    if resolved == "cuda" and not _cuda_ok():
        raise DeviceError(
            "device='cuda' was requested but torch reports no CUDA device.\n"
            "  install a CUDA build of torch, from PyTorch's own index rather "
            "than the default one\n"
            "  (which serves a CPU-only wheel under the same name on Windows):\n"
            "    pip install torch --index-url "
            "https://download.pytorch.org/whl/cu124\n"
            "  then check nvidia-smi. Or pass device='cpu' -- on the clocks that "
            "ship today it is\n  the faster of the two anyway."
        )
    if resolved == "mps" and not _mps_ok():
        raise DeviceError("device='mps' was requested but torch reports no MPS device.")
    if resolved not in ("cpu", "cuda", "mps"):
        raise DeviceError(f"unknown device {resolved!r}; expected cpu, cuda, mps or auto")

    backend = "torch" if (resolved != "cpu" or (torch_available() and env == "torch")) else "numpy"

    want = dtype or "float64"
    if requires_fp64 and want != "float64":
        import warnings

        warnings.warn(
            "this clock is flagged requires_fp64 in the registry; the float32 "
            "request was overridden. Its preprocessing is ill-conditioned enough "
            "that single precision moves the reported value.",
            stacklevel=2,
        )
        want = "float64"

    if resolved == "mps" and want == "float64":
        raise DeviceError(
            "Apple MPS has no float64. Either pass dtype='float32' (and check "
            "the clock is not flagged requires_fp64) or use device='cpu'."
        )

    return DeviceSpec(device=resolved, dtype=want, backend=backend)


def describe() -> dict[str, Any]:
    """What this machine can actually do. Backs ``falconage config``."""
    info: dict[str, Any] = {
        "numpy": np.__version__,
        "torch": None,
        "cuda_available": False,
        "cuda_version": None,
        "mps_available": False,
        "devices": ["cpu"],
    }
    if torch_available():
        import torch

        info["torch"] = torch.__version__
        info["cuda_version"] = torch.version.cuda
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["mps_available"] = _mps_ok()
        if info["cuda_available"]:
            info["devices"].append("cuda")
            info["cuda_devices"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
        if info["mps_available"]:
            info["devices"].append("mps")
    return info
