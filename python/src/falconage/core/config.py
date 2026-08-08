"""Resolved run configuration.

Precedence, highest first: explicit argument, environment variable, user config
file, package default. The resolved object is written into every run manifest,
so a result carries the settings that produced it rather than the settings that
happen to be current when somebody reads it back.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .._version import REGISTRY_VERSION


def default_cache_dir() -> Path:
    env = os.environ.get("FALCONAGE_CACHE")
    if env:
        return Path(env)
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "falconage"
    return Path.home() / ".cache" / "falconage"


def default_config_path() -> Path:
    env = os.environ.get("FALCONAGE_CONFIG")
    if env:
        return Path(env)
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "falconage" / "config.yaml"


@dataclass
class FalconConfig:
    device: str = "auto"
    dtype: str = "float64"
    verbose: str = "inform"
    #: Pinning this is what makes a result reproducible across package
    #: upgrades: the code may change, the coefficients may not.
    registry_version: str = REGISTRY_VERSION
    cache_dir: Path = field(default_factory=default_cache_dir)
    #: How to fill a clock feature the data does not carry. "reference" uses
    #: the value the clock's own authors published; "none" refuses and raises.
    #: Never zero -- zero is a real, extreme beta value, not a missing one.
    imputation: str = "reference"
    #: Below this fraction of a clock's features present, refuse rather than
    #: return a number computed mostly from imputed values.
    min_coverage: float = 0.8
    offline: bool = False

    @classmethod
    def load(cls, path: Path | None = None, **overrides: Any) -> FalconConfig:
        cfg = cls()
        p = path or default_config_path()
        if p.exists():
            doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            for k, v in doc.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, Path(v) if k == "cache_dir" else v)

        for env_key, attr, cast in (
            ("FALCONAGE_DEVICE", "device", str),
            ("FALCONAGE_DTYPE", "dtype", str),
            ("FALCONAGE_VERBOSE", "verbose", str),
            ("FALCONAGE_REGISTRY_VERSION", "registry_version", str),
            ("FALCONAGE_OFFLINE", "offline", lambda s: s.lower() in ("1", "true", "yes")),
            ("FALCONAGE_MIN_COVERAGE", "min_coverage", float),
        ):
            if env_key in os.environ:
                setattr(cfg, attr, cast(os.environ[env_key]))

        for k, v in overrides.items():
            if v is not None and hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cache_dir"] = str(self.cache_dir)
        return d
