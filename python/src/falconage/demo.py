"""A small synthetic dataset, so the whole pipeline can be seen before real data.

The values are drawn, not measured. Each probe's beta drifts linearly with a
known "age", which is enough for every step of a run to produce output (scores,
uncertainty, acceleration, figures, the report) and not enough for any number
in it to mean anything biologically. Every file it writes says so.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .core.container import FalconData

__all__ = ["demo_data", "write_demo"]

NOTE = ("SYNTHETIC DEMO DATA. Beta values were generated at random with a linear "
        "age trend; no number computed from them describes a real person.")


def demo_data(n: int = 40, seed: int = 20261122) -> FalconData:
    """A synthetic methylation dataset covering every bundled array clock.

    ``n`` samples aged 20 to 80, half labelled ``control`` and half ``case``,
    all whole blood, on the 450K platform.
    """
    from . import registry as _registry

    reg = _registry.load()
    feats: set[str] = set()
    for c in reg.filter(availability="bundled"):
        if c.formula or c.data_type != "dna_methylation":
            continue
        try:
            feats.update(reg.feature_ids(c.id))
        except Exception:      # an entry whose features need a supplied file
            continue
    features = sorted(f for f in feats if not f.lower().startswith("(intercept"))

    rng = np.random.default_rng(seed)
    age = np.round(np.linspace(20, 80, n) + rng.normal(0, 1.5, n), 1)
    base = rng.uniform(0.15, 0.85, size=len(features))
    drift = rng.normal(0, 0.0012, size=len(features))
    X = np.clip(base[None, :] + drift[None, :] * (age[:, None] - 50.0)
                + rng.normal(0, 0.01, size=(n, len(features))), 0.001, 0.999)

    ids = [f"DEMO{i + 1:03d}" for i in range(n)]
    obs = pd.DataFrame({
        "age": age,
        "sex": ["F" if i % 2 else "M" for i in range(n)],
        "tissue": "whole blood",
        "group": ["control" if i % 2 else "case" for i in range(n)],
    }, index=ids)
    return FalconData(X=pd.DataFrame(X, index=ids, columns=features), obs=obs,
                      modality="dna_methylation", platform="450K")


def write_demo(outdir: str | Path, n: int = 40) -> tuple[Path, Path]:
    """Write ``demo_betas.csv`` and ``demo_samples.csv`` into ``outdir``.

    The two files have exactly the layout ``falconage run`` expects from real
    data: samples as rows and probes as columns, and a sample sheet keyed by
    the same sample ids.
    """
    d = Path(outdir)
    d.mkdir(parents=True, exist_ok=True)
    data = demo_data(n)
    betas = d / "demo_betas.csv"
    samples = d / "demo_samples.csv"
    data.X.round(5).to_csv(betas, index_label="sample_id")
    data.obs.to_csv(samples, index_label="sample_id")
    (d / "README_DEMO.txt").write_text(NOTE + "\n", encoding="utf-8")
    return betas, samples
