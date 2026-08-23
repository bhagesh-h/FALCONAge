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

    #: What the arithmetic actually ran on, across every clock in the run, or
    #: ``"mixed"`` when they differ. Two things make them differ legitimately:
    #: a clock flagged ``requires_fp64`` keeps double precision through a
    #: float32 request, and a model class declaring ``CPU_ONLY`` computes in
    #: numpy whatever device was asked for. Read ``compute`` for the breakdown.
    #:
    #: Before this release these three were overwritten once per clock inside the
    #: scoring loop, so they reported the *last* clock scored and described a
    #: run that had not happened: ``device="cuda"`` for PhenoAge, which has no
    #: device path at all. A provenance field that is wrong is worse than one
    #: that is missing, because it is quoted.
    device: str = "cpu"
    dtype: str = "float64"
    backend: str = "numpy"
    #: What the ``device=`` argument resolved to before any clock narrowed it.
    #: The ceiling, not the fact.
    device_requested: str = "cpu"

    python: str = field(default_factory=lambda: sys.version.split()[0])
    platform: str = field(default_factory=platform.platform)
    #: Set by the R bridge so an R-originated run is distinguishable from a
    #: Python one even though the arithmetic is identical.
    caller: str = "python"

    config: dict[str, Any] = field(default_factory=dict)
    inputs: list[dict[str, str]] = field(default_factory=list)
    #: clock_id -> {sha256, source, n_features, tier}. The sha256 is of the
    #: coefficient file actually used, so a user-supplied licensed copy is visibly
    #: different from a redistributed one.
    weights: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: clock_id -> {device, dtype, backend} actually used for that clock.
    #: Populated only for clocks that produced a score.
    compute: dict[str, dict[str, str]] = field(default_factory=dict)
    #: clock_id -> coverage, imputed count, and any per-clock note
    coverage: dict[str, dict[str, Any]] = field(default_factory=dict)
    warnings: list[dict[str, str]] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    #: What happened to the specimens before the assay: which of the recognised
    #: pre-analytical fields were supplied, which were not, and which crossed a
    #: published threshold. See :mod:`falconage.core.preanalytical`. Recorded
    #: even when empty -- a manifest that says "storage time unknown" is more
    #: use than one that says nothing at all.
    preanalytical: dict[str, Any] = field(default_factory=dict)

    def add_input(self, path: str | Path, role: str = "data") -> None:
        p = Path(path)
        self.inputs.append({
            "role": role,
            "path": str(p),
            "sha256": file_sha256(p) if p.is_file() else "",
            "bytes": str(p.stat().st_size) if p.is_file() else "",
        })

    def record_compute(self, clock_id: str, spec: Any) -> None:
        """Note what one clock actually computed in, and refresh the summary.

        Called per clock by :func:`falconage.score.score` with the spec the
        model reported through :func:`falconage.models.effective_spec`, which
        is not always the spec the run requested.
        """
        self.compute[clock_id] = {"device": spec.device, "dtype": spec.dtype,
                                  "backend": spec.backend}
        self.refresh_compute_summary()

    def refresh_compute_summary(self) -> None:
        """Re-derive the three scalar fields from ``compute``.

        Public because :func:`falconage.score.combine` fills ``compute`` from
        several manifests at once rather than one clock at a time.
        """
        if not self.compute:
            return
        for f in ("device", "dtype", "backend"):
            seen = {c[f] for c in self.compute.values()}
            setattr(self, f, seen.pop() if len(seen) == 1 else "mixed")

    def compute_summary(self) -> str:
        """One line naming every device and precision the run used.

        ``"torch:cuda/float64"`` when the run was uniform, and
        ``"torch:cuda/float64 (20 clocks), numpy:cpu/float64 (3)"`` when it was
        not. The scalar fields alone cannot say the second thing, and printing
        ``"mixed:mixed/mixed"`` in a report would be worse than saying nothing.
        """
        if not self.compute:
            return f"{self.backend}:{self.device}/{self.dtype}"
        counts: dict[str, int] = {}
        for c in self.compute.values():
            counts[f"{c['backend']}:{c['device']}/{c['dtype']}"] = (
                counts.get(f"{c['backend']}:{c['device']}/{c['dtype']}", 0) + 1)
        if len(counts) == 1:
            return next(iter(counts))
        ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return ", ".join(f"{k} ({n} clock{'s' if n != 1 else ''})"
                         for k, n in ordered)

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
                f"{self.registry_version}, {self.compute_summary()}, "
                f"{len(self.weights)} clock(s), {len(self.warnings)} warning(s))")
