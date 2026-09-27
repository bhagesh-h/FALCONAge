"""The PC clock importer, against the authors' own scoring code.

The authors' data file cannot ship, so this builds a small one with the same
objects (``CalcPCHorvath1`` ... ``CpGs``, ``imputeMissingCpGs``,
``anti.trafo``), scores a matrix with absent and partly missing CpGs using the
lines of ``run_calcPCClocks.R`` (MorganLevineLab/PC-Clocks at 5e65bce) that do
the imputation and the five single-model clocks, and requires FALCONAge to
return the same numbers after ``import_pc_clocks``.
"""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

import falconage as fa
from falconage.core.errors import RegistryError
from falconage.registry.registry import ClockRegistry

pytestmark = pytest.mark.skipif(shutil.which("Rscript") is None, reason="needs Rscript")

AUTHORS = r"""
set.seed(7)
CpGs <- sprintf("cg%%08d", 1:300)
mk <- function(k, b) list(center = runif(300, 0.2, 0.8),
                          rotation = matrix(rnorm(300 * k, 0, 0.2), 300, k),
                          model = rnorm(k), intercept = b)
CalcPCHorvath1 <- mk(6, 0.4); CalcPCHorvath2 <- mk(5, -0.3)
CalcPCHannum <- mk(7, 50); CalcPCPhenoAge <- mk(4, 45); CalcPCDNAmTL <- mk(3, 7)
imputeMissingCpGs <- setNames(runif(300, 0.1, 0.9), CpGs)
anti.trafo <- function(x, adult.age = 20) {
  ifelse(x < 0, (1 + adult.age) * exp(x) - 1, (1 + adult.age) * x + adult.age)
}
save(CalcPCHorvath1, CalcPCHorvath2, CalcPCHannum, CalcPCPhenoAge, CalcPCDNAmTL,
     CpGs, imputeMissingCpGs, anti.trafo, file = "%(dir)s/CalcAllPCClocks.RData")

# The input: 12 samples, 25 CpGs absent, 30 values missing among the rest.
X <- matrix(runif(12 * 300, 0.05, 0.95), 12, 300, dimnames = list(sprintf("S%%02d", 1:12), CpGs))
X <- X[, -(1:25)]
X[cbind(sample(12, 30, TRUE), sample(ncol(X), 30, TRUE))] <- NA
write.csv(X, "%(dir)s/betas.csv")

# run_calcPCClocks.R, lines 47-66 and 81-85, unchanged but for the load.
datMeth <- as.data.frame(X)
missingCpGs <- c(CpGs[!(CpGs %%in%% colnames(datMeth))])
datMeth[, missingCpGs] <- NA
datMeth = datMeth[, CpGs]
missingCpGs <- CpGs[apply(datMeth[, CpGs], 2, function(x) all(is.na(x)))]
for (i in 1:length(missingCpGs)) {
  datMeth[, missingCpGs[i]] <- imputeMissingCpGs[missingCpGs[i]]
}
datMeth <- datMeth[, CpGs]
meanimpute <- function(x) ifelse(is.na(x), mean(x, na.rm = T), x)
datMeth <- apply(datMeth, 2, meanimpute)
f <- function(o) sweep(as.matrix(datMeth), 2, o$center) %%*%% o$rotation %%*%% o$model + o$intercept
out <- data.frame(pchorvath2013 = as.numeric(anti.trafo(f(CalcPCHorvath1))),
                  pcskinandblood = as.numeric(anti.trafo(f(CalcPCHorvath2))),
                  pchannum = as.numeric(f(CalcPCHannum)),
                  pcphenoage = as.numeric(f(CalcPCPhenoAge)),
                  pcdnamtl = as.numeric(f(CalcPCDNAmTL)), row.names = rownames(X))
write.csv(out, "%(dir)s/authors.csv")
"""

IDS = ["pchorvath2013", "pcskinandblood", "pchannum", "pcphenoage", "pcdnamtl"]


@pytest.fixture(scope="module")
def pcdir(tmp_path_factory):
    d = tmp_path_factory.mktemp("pcclocks")
    subprocess.run(["Rscript", "-e", AUTHORS % {"dir": d.as_posix()}], check=True,
                   capture_output=True)
    return d


def test_the_import_reproduces_the_authors_code(pcdir):
    reg = ClockRegistry.from_yaml()           # registration mutates the registry
    digests = reg.import_pc_clocks(pcdir / "CalcAllPCClocks.RData", out_dir=pcdir / "out")
    assert sorted(digests) == sorted(IDS)
    X = pd.read_csv(pcdir / "betas.csv", index_col=0)
    obs = pd.DataFrame({"tissue": "whole blood"}, index=X.index)
    data = fa.FalconData(X=X, obs=obs, modality="dna_methylation", platform="450K")
    res = fa.score(data, clocks=IDS, registry=reg, min_coverage=0.0)
    want = pd.read_csv(pcdir / "authors.csv", index_col=0)
    for cid in IDS:
        assert np.allclose(res.scores[cid], want[cid], rtol=1e-10, atol=1e-10), cid
        assert res.manifest.coverage[cid]["n_from_reference"] == 25
        assert res.manifest.weights[cid]["source"] == "user_supplied"


def test_a_different_transform_is_refused(pcdir, tmp_path):
    bad = tmp_path / "CalcAllPCClocks.RData"
    code = (f'load("{(pcdir / "CalcAllPCClocks.RData").as_posix()}"); '
            'anti.trafo <- function(x, adult.age = 18) x; '
            f'save(list = ls(), file = "{bad.as_posix()}")')
    subprocess.run(["Rscript", "-e", code], check=True, capture_output=True)
    with pytest.raises(RegistryError, match="not Horvath's transform"):
        ClockRegistry.from_yaml().import_pc_clocks(bad, out_dir=tmp_path / "out")


def test_an_intercept_row_is_a_constant_not_a_cpg(tmp_path):
    """A user's file with an (Intercept) row: the row is added, never aligned
    as a CpG the data lacks and filled."""
    reg = ClockRegistry.from_yaml()
    p = tmp_path / "w.csv"
    pd.DataFrame({"feature_id": ["(Intercept)", "cg1", "cg2"],
                  "coefficient": [10.0, 2.0, -1.0]}).to_csv(p, index=False)
    reg.register_local_weights("pchannum", p)
    feats, w = reg.coefficients("pchannum")
    assert feats == ["cg1", "cg2"] and reg.intercept("pchannum") == 10.0
    X = pd.DataFrame({"cg1": [0.5, 0.2], "cg2": [0.1, 0.4]}, index=["a", "b"])
    data = fa.FalconData(X=X, obs=pd.DataFrame({"tissue": "whole blood"}, index=X.index),
                         modality="dna_methylation")
    got = fa.score(data, clocks=["pchannum"], registry=reg, min_coverage=0.0).scores["pchannum"]
    assert np.allclose(got, [10 + 1.0 - 0.1, 10 + 0.4 - 0.4])
