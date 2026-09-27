"""cAge: an elastic net on CpGs and their squares, with a log-age model for the young.

THE MODEL (Bernabeu et al. 2023, Genome Medicine 15:12). Two elastic nets
fitted on 18,413 Generation Scotland blood samples, both on beta values and
on the squares of some of them:

.. math::

    \\hat a = b_0 + \\sum_j w_j x_j + \\sum_j v_j x_j^2

one trained on age (Additional file 4, Table S8) and one on log(age) (Table
S9). A sample the first puts at 20 years or younger is re-scored with the
second, and ``exp()`` of that is returned (the paper's Methods): a linear
model on age is least accurate where age changes fastest. A term named ``cg..._2`` in the tables is the square of that CpG's
beta.

The coefficient file carries both tables, one column each, a term absent from
a model having weight 0 there. A missing value of a CpG the data carries takes
that CpG's cohort mean, as the authors' code does; a CpG the data lacks takes
its reference value (the authors use their training means, which are in their
repository but not in the paper and carry no licence, so this uses the
registry's reference). The authors' ``cage_predictor.R`` keeps ``> 20`` and
``< 20`` and so drops a sample predicted at exactly 20; the paper's "20 years
or younger" is followed here.

Because a squared term has no single weight per CpG, the registry reports no
coefficient vector for this clock. The coefficient-mass coverage below weighs
each CpG by the sum of its linear and squared weights in the linear model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..core.errors import FeatureCoverageError
from ..registry.registry import Clock
from .linear import Alignment, align

#: At or below this linear prediction the log-age model replaces it.
SWITCH_AGE = 20.0


@dataclass
class QuadraticClock:
    clock: Clock
    features: list[str]              # base CpGs, in the file's first-seen order
    terms: pd.DataFrame              # rows: features; columns lin, lin_sq, log, log_sq
    intercept: float
    intercept_log: float
    reference: dict[str, float] | None = None

    def predict(self, data, spec, *, imputation: str = "reference",
                min_coverage: float = 0.8) -> tuple[pd.Series, Alignment]:
        mass = (self.terms["lin"].abs() + self.terms["lin_sq"].abs()).to_numpy()
        al = align(data, self.features, imputation=imputation, reference=self.reference,
                   coefficients=mass)
        if al.coverage < min_coverage:
            raise FeatureCoverageError(
                f"{self.clock.id}: {al.coverage:.1%} of its {len(self.features)} CpGs are "
                f"present, below the {min_coverage:.0%} floor.")
        if al.mass_coverage is not None and al.mass_coverage < min_coverage:
            raise FeatureCoverageError(
                f"{self.clock.id}: the CpGs present carry {al.mass_coverage:.1%} of the "
                f"linear model's weight, below the {min_coverage:.0%} floor.")
        X = np.asarray(al.matrix, dtype=np.float64)
        t = self.terms
        linear = self.intercept + X @ t["lin"].to_numpy() + (X ** 2) @ t["lin_sq"].to_numpy()
        logage = self.intercept_log + X @ t["log"].to_numpy() + (X ** 2) @ t["log_sq"].to_numpy()
        out = np.where(linear <= SWITCH_AGE, np.exp(logage), linear)
        return pd.Series(out, index=data.sample_ids, name=self.clock.id), al

    @classmethod
    def from_registry(cls, registry, clock_id: str) -> QuadraticClock:
        feats, terms, b0, b0_log = registry.quadratic_model(clock_id)
        return cls(clock=registry.get(clock_id), features=feats, terms=terms,
                   intercept=b0, intercept_log=b0_log,
                   reference=registry.reference_values(clock_id))


def is_quadratic(clock: Clock) -> bool:
    """An entry whose model has squared-CpG terms (cAge)."""
    return "quadratic terms" in (clock.model_type or "").lower()


def read_terms(path) -> tuple[list[str], pd.DataFrame, float, float]:
    """Base CpGs, their four weights, and the two intercepts, from a file with
    columns ``feature_id, coefficient, log_coefficient``."""
    t = pd.read_csv(path, dtype={"feature_id": str})
    t["feature_id"] = t["feature_id"].str.strip()
    icpt = t["feature_id"].str.lower().isin(["(intercept)", "intercept"])
    b0 = float(t.loc[icpt, "coefficient"].iloc[0])
    b0_log = float(t.loc[icpt, "log_coefficient"].iloc[0])
    t = t[~icpt]
    sq = t["feature_id"].str.endswith("_2")
    base = t["feature_id"].str.replace(r"_2$", "", regex=True)
    feats = list(dict.fromkeys(base))
    lin = pd.DataFrame(0.0, index=feats, columns=["lin", "lin_sq", "log", "log_sq"])
    for is_sq, col_lin, col_log in ((False, "lin", "log"), (True, "lin_sq", "log_sq")):
        part = t[sq == is_sq]
        idx = base[sq == is_sq]
        lin.loc[idx.to_numpy(), col_lin] = part["coefficient"].fillna(0.0).to_numpy()
        lin.loc[idx.to_numpy(), col_log] = part["log_coefficient"].fillna(0.0).to_numpy()
    return feats, lin, b0, b0_log
