"""Shared fixtures.

Two kinds of data here. **Synthetic** fixtures are built from a fixed seed and
are what the arithmetic is asserted against -- they let a test say "this exact
number" rather than "something plausible". **Corpus** fixtures are the real
public data in ``test/data``; they are skipped when it is absent, because a test
suite that cannot run without a 586 MB download is a test suite nobody runs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import falconage as fa

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "test" / "data"


def pytest_collection_modifyitems(config, items):
    have = (CORPUS / "checksums.sha256").exists()
    skip = pytest.mark.skip(reason="test corpus absent; see test/data/README.md")
    for item in items:
        if "corpus" in item.keywords and not have:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def corpus() -> Path:
    if not (CORPUS / "checksums.sha256").exists():
        pytest.skip("test corpus absent")
    return CORPUS


@pytest.fixture(scope="session")
def registry():
    return fa.registry.load()


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(20260807)


@pytest.fixture(scope="session")
def synthetic_betas(registry, rng):
    """A methylation dataset carrying every feature the tier A clocks need.

    Built so coverage is exactly 1.0 for the bundled clocks: coverage failures
    have their own tests, and a fixture that half-covers everything would make
    every other test a coverage test by accident.

    Betas are drawn per feature from a Beta distribution whose mean drifts with
    a latent "age" so that the clocks see a real age signal rather than noise --
    enough for monotonicity properties, not enough to pretend the numbers mean
    anything biologically.
    """
    feats: set[str] = set()
    for c in registry.filter(availability="A"):
        if c.formula:
            continue
        feats.update(registry.feature_ids(c.id))
    features = sorted(feats)

    n = 24
    age = np.linspace(20, 80, n)
    base = rng.uniform(0.15, 0.85, size=len(features))
    drift = rng.normal(0, 0.0012, size=len(features))
    X = np.clip(base[None, :] + drift[None, :] * (age[:, None] - 50.0)
                + rng.normal(0, 0.01, size=(n, len(features))), 0.001, 0.999)

    ids = [f"S{i:03d}" for i in range(n)]
    obs = pd.DataFrame({
        "age": age,
        "sex": ["M" if i % 2 else "F" for i in range(n)],
        "condition": ["HC"] * 14 + ["CASE"] * 10,
        "dataset": ["SYN1"] * n,
    }, index=ids)
    return fa.FalconData(X=pd.DataFrame(X, index=ids, columns=features),
                         obs=obs, modality="dna_methylation", platform="450K")


@pytest.fixture(scope="session")
def synthetic_clinical(rng):
    """A clinical cohort in the units PhenoAge's coefficients expect."""
    n = 200
    age = rng.uniform(25, 85, n)
    ids = [f"C{i:03d}" for i in range(n)]
    df = pd.DataFrame({
        "albumin": rng.normal(43 - 0.03 * age, 2.5),
        "creatinine": rng.normal(70 + 0.25 * age, 12),
        "glucose": rng.normal(5.0 + 0.012 * age, 0.7),
        "crp": np.exp(rng.normal(-1.0 + 0.012 * age, 0.6)),
        "lymphocyte_percent": rng.normal(32 - 0.06 * age, 5),
        "mean_cell_volume": rng.normal(89 + 0.03 * age, 4),
        "red_cell_distribution_width": rng.normal(12.8 + 0.012 * age, 0.7),
        "alkaline_phosphatase": rng.normal(70 + 0.20 * age, 15),
        "white_blood_cell_count": rng.normal(6.5, 1.4),
        "age": age,
    }, index=ids)
    obs = pd.DataFrame({"age": age, "sex": ["F" if i % 2 else "M" for i in range(n)]},
                       index=ids)
    return fa.FalconData(X=df, obs=obs, modality="clinical_chemistry",
                         units={"albumin": "g/L", "creatinine": "umol/L",
                                "glucose": "mmol/L", "crp": "mg/dL",
                                "lymphocyte_percent": "%", "mean_cell_volume": "fL",
                                "red_cell_distribution_width": "%",
                                "alkaline_phosphatase": "U/L",
                                "white_blood_cell_count": "10^3/uL", "age": "years"})


@pytest.fixture(autouse=True)
def _reset_local_weights():
    """Undo any user-registered weights between tests.

    ``registry.load()`` is cached for the life of the process -- deliberately,
    because re-parsing 161 entries per call would be absurd -- and
    ``register_local_weights`` mutates it. Without this, a test that supplies a
    synthetic GrimAge2 file turns GrimAge2 into a working clock for every test
    that runs after it, and the scaffold assertions silently stop asserting.
    """
    yield
    fa.registry.load()._local.clear()


@pytest.fixture
def fresh_registry():
    """An unshared registry, for tests that register local weights."""
    return fa.registry.ClockRegistry.from_yaml()
