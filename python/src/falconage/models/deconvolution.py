"""Reference-based cell-type deconvolution: Houseman's constrained projection.

WHAT A SAMPLE'S PROPORTIONS ARE. Bulk blood methylation at a CpG is, to first
order, the mixture of each cell type's methylation there, weighted by how much
of the sample that cell type is. Given a reference table of cell-type means at
CpGs chosen to separate them (``X``, CpGs x cell types), the proportions ``w``
of one sample ``y`` are the least-squares fit ``min ||X w - y||^2`` with every
``w`` non-negative (Houseman et al. 2012, BMC Bioinformatics 13:86).

WHICH IMPLEMENTATION, EXACTLY. ``projectCellType_CP`` in FlowSorted.Blood.EPIC
(GPL-3), with the arguments its documentation gives for a beta matrix:
``nonnegative = TRUE, lessThanOne = FALSE``, the same defaults as
``estimateCellCounts2`` and minfi. Per sample it keeps the CpGs that sample
observes, solves the quadratic programme with ``quadprog::solve.QP`` and rounds
to four decimals. A quadratic programme whose only constraints are ``w >= 0``
is non-negative least squares, and with the reference columns linearly
independent its solution is unique, so ``scipy.optimize.nnls`` returns the same
point. The proportions are not forced to sum to one, as in the authors'
default; the sum is how far the reference explains the sample.

THE REFERENCE. For the six-cell entries, the IDOL libraries of Salas et al.
2018 (Genome Biology 19:64): 450 CpGs on EPIC and a 350-CpG legacy set for the
450K array, from ``build_idol.py``. The entries share the table, and each
returns its own column.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..core.backend import DeviceSpec
from ..core.errors import FeatureCoverageError
from ..registry.registry import Clock
from .linear import Alignment, align_present


def project(Y: np.ndarray, X: np.ndarray, *, decimals: int = 4) -> np.ndarray:
    """``projectCellType_CP(Y, X, nonnegative = TRUE, lessThanOne = FALSE)``.

    ``Y`` is samples x CpGs with NaN where a sample has no value; ``X`` is
    CpGs x cell types. Returns samples x cell types, rounded as the reference
    implementation rounds. A sample with fewer observed CpGs than cell types
    has no unique answer and gets NaN.
    """
    from scipy.optimize import nnls

    n, k = Y.shape[0], X.shape[1]
    out = np.full((n, k), np.nan)
    for i in range(n):
        obs = ~np.isnan(Y[i])
        if obs.sum() < k:
            continue
        w, _ = nnls(X[obs], Y[i, obs])
        out[i] = w
    return np.round(out, decimals)


@dataclass
class DeconvolutionClock:
    """One cell type's proportion from a shared reference table."""

    clock: Clock
    registry: object

    def predict(self, data, spec: DeviceSpec, *, imputation: str = "reference",
                min_coverage: float = 0.8) -> tuple[pd.Series, Alignment]:
        table = self.registry.deconvolution_table(
            self.clock.id, data.platform, present=data.X.columns)
        feats = list(table.index)
        # Absent CpGs are left out rather than filled, as the reference
        # implementation does per sample.
        al = align_present(data, feats)
        if al.coverage < min_coverage:
            raise FeatureCoverageError(
                f"{self.clock.id}: {al.coverage:.1%} of the {len(feats)} reference "
                f"CpGs are present, below the {min_coverage:.0%} floor.\n"
                f"  The dataset is {data.platform or 'an unknown platform'}; the "
                "reference tables are for EPIC and 450K.")
        props = project(al.matrix, table.to_numpy())
        j = list(table.columns).index(self.clock.deconvolution.cell_type)
        return pd.Series(props[:, j], index=data.sample_ids, name=self.clock.id), al

    @classmethod
    def from_registry(cls, registry, clock_id: str) -> DeconvolutionClock:
        return cls(clock=registry.get(clock_id), registry=registry)


def is_deconvolution(clock: Clock) -> bool:
    return clock.deconvolution is not None
