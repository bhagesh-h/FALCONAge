"""The run manifest: what produced a number, recorded next to the number.

Two runs that report the same score either used the same coefficients, the same
imputation and the same precision, or the manifest says they did not. That is
the whole claim, and every field here exists to support it. Nothing is derived
at read time -- a manifest is a statement about a run that already happened.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .._version import REGISTRY_VERSION, __version__


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class RunManifest:
    """Provenance for one scoring run."""

    falconage_version: str = __version__
    registry_version: str = REGISTRY_VERSION
    started_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_utc: str | None = None

    device: str = "cpu"
    dtype: str = "float64"
    backend: str = "numpy"

    python: str = field(default_factory=lambda: sys.version.split()[0])
    platform: str = field(default_factory=platform.platform)
    #: Set by the R bridge so an R-originated run is distinguishable from a
    #: Python one even though the arithmetic is identical.
    caller: str = "python"

    config: dict[str, Any] = field(default_factory=dict)
    inputs: list[dict[str, str]] = field(default_factory=list)
    #: clock_id -> {sha256, source, n_features, tier}. The sha256 is of the
    #: coefficient file actually used, so a licensed tier C copy is visibly
    #: different from a redistributed one.
    weights: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: clock_id -> coverage, imputed count, and any per-clock note
    coverage: dict[str, dict[str, Any]] = field(default_factory=dict)
    warnings: list[dict[str, str]] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)

    def add_input(self, path: str | Path, role: str = "data") -> None:
        p = Path(path)
        self.inputs.append({
            "role": role,
            "path": str(p),
            "sha256": file_sha256(p) if p.is_file() else "",
            "bytes": str(p.stat().st_size) if p.is_file() else "",
        })

    def finish(self) -> RunManifest:
        self.finished_utc = datetime.now(timezone.utc).isoformat()
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
                     encoding="utf-8", newline="\n")
        return p

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (f"RunManifest(falconage {self.falconage_version}, registry "
                f"{self.registry_version}, {self.backend}:{self.device}/{self.dtype}, "
                f"{len(self.weights)} clock(s), {len(self.warnings)} warning(s))")
