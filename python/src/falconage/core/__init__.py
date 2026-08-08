"""Device resolution, the data container, units, config, manifest and errors."""

from .backend import DeviceSpec, describe, resolve, torch_available
from .config import FalconConfig
from .container import FalconData
from .errors import (
    AnalysisError,
    ChecksumMismatchError,
    ClockNotFoundError,
    DataError,
    DeviceError,
    DownloadError,
    FalconError,
    FeatureCoverageError,
    IllegalOperationError,
    PlatformError,
    RegistryError,
    ScoringError,
    UnitConversionError,
    UnitsNotDeclaredError,
    WeightsUnavailableError,
)
from .logging import WarningCollector, configure, get_logger, quiet
from .manifest import RunManifest, file_sha256

__all__ = [
    "AnalysisError", "ChecksumMismatchError", "ClockNotFoundError", "DataError",
    "DeviceError", "DeviceSpec", "DownloadError", "FalconConfig", "FalconData",
    "FalconError", "FeatureCoverageError", "IllegalOperationError", "PlatformError",
    "RegistryError", "RunManifest", "ScoringError", "UnitConversionError",
    "UnitsNotDeclaredError", "WarningCollector", "WeightsUnavailableError",
    "configure", "describe", "file_sha256", "get_logger", "quiet", "resolve",
    "torch_available",
]
