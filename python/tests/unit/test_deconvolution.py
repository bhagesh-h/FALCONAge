"""Six-cell blood deconvolution against the IDOL library (Salas et al. 2018).

The projection is FlowSorted.Blood.EPIC's projectCellType_CP with its
documented arguments for a beta matrix (nonnegative = TRUE, lessThanOne =
FALSE): non-negative least squares per sample over the CpGs it observes,
rounded to four decimals. Its optimality is checked here through the KKT
conditions, which hold for the solution and nothing else whatever solver found
it.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

import falconage as fa
from falconage.core.errors import RegistryError
from falconage.models.deconvolution import project
from falconage.registry.registry import DATA_DIR, ClockRegistry

SIX = {"deconvolutebloodepiccd8tcell": "CD8T", "deconvolutebloodepiccd4tcell": "CD4T",
       "deconvolutebloodepicnkcell": "NK", "deconvolutebloodepicbcell": "Bcell",
       "deconvolutebloodepicmonocyte": "Mono", "deconvolutebloodepicneutrophil": "Neu"}


def test_the_idol_tables_are_the_authors(registry):
    epic = registry.deconvolution_table("deconvolutebloodepicbcell", "EPICv1")
    legacy = registry.deconvolution_table("deconvolutebloodepicbcell", "450K")
    assert epic.shape == (450, 6) and legacy.shape == (350, 6)
    assert list(epic.columns) == ["CD8T", "CD4T", "NK", "Bcell", "Mono", "Neu"]
    # IDOLOptimizedCpGs.compTable, first row, as R prints it: 0.1970004.
    assert epic.loc["cg08769189", "CD8T"] == 0.197000439067314
    for t in registry.get("deconvolutebloodepicbcell").deconvolution.tables:
        assert hashlib.sha256((DATA_DIR / t.file).read_bytes()).hexdigest() == t.sha256


def test_a_deconvolution_entry_has_no_coefficient_vector(registry):
    cid = "deconvolutebloodepicneutrophil"
    assert not registry.has_coefficient_vector(cid)
    assert len(registry.feature_ids(cid)) == 450
    with pytest.raises(RegistryError, match="reference table"):
        registry.coefficients(cid)


def test_the_projection_is_the_nonnegative_least_squares_optimum(registry, rng):
    X = registry.deconvolution_table("deconvolutebloodepicbcell", "EPICv1").to_numpy()
    w_true = rng.dirichlet(np.ones(6), size=20)
    w_true[:5, 2] = 0.0                     # some cell types truly absent
    Y = w_true @ X.T + rng.normal(0, 0.03, size=(20, X.shape[0]))
    W = project(Y, X, decimals=12)
    assert (W >= 0).all()
    for w, y in zip(W, Y):
        g = X.T @ (X @ w - y)               # gradient of 1/2 ||Xw - y||^2
        on = w > 1e-9
        assert np.abs(g[on]).max(initial=0.0) < 1e-6
        assert (g[~on] > -1e-6).all()


def test_a_known_mixture_is_recovered_to_the_rounding(registry, rng):
    X = registry.deconvolution_table("deconvolutebloodepicbcell", "EPICv1").to_numpy()
    w_true = rng.dirichlet(np.ones(6), size=10)
    W = project(w_true @ X.T, X)
    assert np.abs(W - w_true).max() <= 5e-5 + 1e-12


def test_a_sample_uses_only_the_cpgs_it_observes(registry, rng):
    X = registry.deconvolution_table("deconvolutebloodepicbcell", "EPICv1").to_numpy()
    y = rng.dirichlet(np.ones(6)) @ X.T + rng.normal(0, 0.02, X.shape[0])
    gone = rng.choice(X.shape[0], 60, replace=False)
    y_na = y.copy()
    y_na[gone] = np.nan
    keep = np.setdiff1d(np.arange(X.shape[0]), gone)
    assert np.array_equal(project(y_na[None, :], X), project(y[None, keep], X[keep]))


def _mixture(registry, w: np.ndarray, platform: str):
    table = registry.deconvolution_table("deconvolutebloodepicbcell", platform)
    ids = [f"M{i}" for i in range(len(w))]
    X = pd.DataFrame(w @ table.to_numpy().T, index=ids, columns=table.index)
    obs = pd.DataFrame({"tissue": "whole blood"}, index=ids)
    return fa.FalconData(X=X, obs=obs, modality="dna_methylation", platform=platform)


@pytest.mark.parametrize("platform", ["EPICv1", "450K"])
def test_scoring_returns_each_cell_types_share(registry, rng, platform):
    w = rng.dirichlet(np.ones(6), size=8)
    res = fa.score(_mixture(registry, w, platform), clocks=list(SIX))
    cols = list(registry.deconvolution_table("deconvolutebloodepicbcell", platform).columns)
    for cid, cell in SIX.items():
        assert np.allclose(res.scores[cid], w[:, cols.index(cell)], atol=5e-5)
    comp = fa.cell_composition(res)
    assert list(comp.columns) == list(res.scores.columns)


def test_the_table_follows_the_array(registry):
    cid = "deconvolutebloodepicmonocyte"
    assert len(registry.deconvolution_table(cid, "450K")) == 350
    assert len(registry.deconvolution_table(cid, "EPICv2")) == 450
    legacy = registry.deconvolution_table(cid, "450K").index
    assert len(registry.deconvolution_table(cid, None, present=legacy)) == 350


def test_the_twelve_cell_library_is_licensed_and_takes_a_local_table(tmp_path, rng):
    reg = ClockRegistry.from_yaml()           # a private copy; registration mutates it
    cid = "twelvecelldeconvolutebloodepictreg"
    c = reg.get(cid)
    assert c.availability == "licensed" and c.deconvolution.cell_type == "Treg"
    assert "Dartmouth" in reg.unavailable_message(cid)

    cells = ["Bas", "Bmem", "Bnv", "CD4mem", "CD4nv", "CD8mem", "CD8nv", "Eos",
             "Mono", "Neu", "NK", "Treg"]
    cpgs = [f"cg{i:08d}" for i in range(300)]
    table = pd.DataFrame(rng.uniform(0.05, 0.95, size=(300, 12)), index=cpgs, columns=cells)
    table.index.name = "feature_id"
    p = tmp_path / "extended.csv"
    table.to_csv(p)
    reg.register_local_weights(cid, p)

    w = rng.dirichlet(np.ones(12), size=5)
    ids = [f"E{i}" for i in range(5)]
    data = fa.FalconData(X=pd.DataFrame(w @ table.to_numpy().T, index=ids, columns=cpgs),
                         obs=pd.DataFrame({"tissue": "whole blood"}, index=ids),
                         modality="dna_methylation", platform="EPICv1")
    res = fa.score(data, clocks=[cid], registry=reg)
    assert np.allclose(res.scores[cid], w[:, cells.index("Treg")], atol=5e-5)
    assert res.manifest.weights[cid]["source"] == "user_supplied"


def test_a_local_table_without_the_entrys_column_is_refused(tmp_path, rng):
    reg = ClockRegistry.from_yaml()
    p = tmp_path / "wrong.csv"
    pd.DataFrame(rng.uniform(size=(10, 2)), columns=["A", "B"],
                 index=pd.Index([f"cg{i}" for i in range(10)], name="feature_id")).to_csv(p)
    with pytest.raises(RegistryError, match="Treg"):
        reg.register_local_weights("twelvecelldeconvolutebloodepictreg", p)
