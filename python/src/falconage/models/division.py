"""Clocks that count stem-cell divisions from a methylation transmission model.

epiTOC2 and epiTOC3 are the two entries in the catalogue that are neither a
weighted sum nor a summary statistic, and the difference is not a detail of
implementation. Every other clock here answers "what does this methylome look
like"; these answer "how many times has this tissue divided", in divisions per
stem cell, on an absolute scale with a fetal zero.

THE MODEL. A CpG in this set is unmethylated in the fetal stage and picks up
methylation with each division at its own rate. Teschendorff's transmission
model gives each site a de-novo methylation probability per division, delta_i,
and a ground-state methylation beta0_i, and inverts the relation site by site::

    TNSC = 2 * mean_i[ (beta_i - beta0_i) / (delta_i * (1 - beta0_i)) ]

The factor of two is in the published model: it converts the per-allele
estimate to divisions per stem cell.

WHY IT IS NOT A LINEAR CLOCK, which is the question the shape invites. Three
reasons, and any one of them is enough:

* The divisor is the number of CpGs *present in this dataset*, not a constant.
  Drop a probe and every remaining term is reweighted, which no fixed
  coefficient vector can express.
* Each site carries two parameters rather than one weight, and the second one
  is subtracted from the data before scaling rather than added to the result.
* The reference implementation reports a second estimate that assumes every
  ground state is zero, for the case where measured betas fall below it. That
  is a modelling decision about the data in hand, not a postprocess.

WHY THE FILE HAS THREE COLUMNS. ``feature_id, coefficient, ground_state``. The
coefficient is the per-site weight ``1 / (delta_i * (1 - beta0_i))`` -- already
folded, because delta and the ground state never appear apart in the forward
pass and storing the quotient keeps the arithmetic in one place. The registry's
own reader takes the first two columns and ignores the rest, so the shipped
file passes the same digest and schema checks as every other coefficient file
while carrying the extra column this class needs.

BELOW THE GROUND STATE. A measured beta under the fitted fetal value makes that
site's term negative, which the model does not admit: it says the tissue has
divided a negative number of times. The author's script warns and offers the
simplified estimate; this class does the same thing in the other direction,
reporting how many sites fell below and leaving the estimate alone, because
silently clipping a negative term would turn a data problem into a plausible
number.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.backend import DeviceSpec
from ..core.errors import FeatureCoverageError, RegistryError
from ..core.logging import get_logger
from ..registry.registry import Clock
from . import ops
from .linear import Alignment, align

__all__ = ["DivisionClock", "is_division_model", "read_division_parameters"]


def is_division_model(clock: Clock) -> bool:
    """True for the entries whose forward pass is the transmission model."""
    return "transmission model" in (clock.model_type or "").lower()


def read_division_parameters(path: str | Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """``feature_id, coefficient, ground_state`` from one of the shipped files.

    Returns the ids, the per-site weights and the per-site ground states.
    """
    feats: list[str] = []
    weights: list[float] = []
    ground: list[float] = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rdr = csv.reader(fh)
        header = next(rdr, None) or []
        if [h.strip().lower() for h in header[:3]] != [
                "feature_id", "coefficient", "ground_state"]:
            raise RegistryError(
                f"{path}: header is {header[:3]}, expected "
                "['feature_id', 'coefficient', 'ground_state']. A division "
                "model needs both parameters per site; two columns is a "
                "linear clock's file.")
        for row in rdr:
            if len(row) < 3 or not row[0].strip():
                continue
            feats.append(row[0].strip())
            weights.append(float(row[1]))
            ground.append(float(row[2]))
    return (feats, np.asarray(weights, dtype=np.float64),
            np.asarray(ground, dtype=np.float64))


@dataclass
class DivisionClock:
    """``2 * mean_i[(x_i - ground_i) * weight_i]`` over the sites present."""

    clock: Clock
    features: list[str]
    coefficients: np.ndarray
    ground_state: np.ndarray

    def predict(self, data, spec: DeviceSpec, *, imputation: str = "reference",
                min_coverage: float = 0.8) -> tuple[pd.Series, Alignment]:
        al = align(data, self.features, imputation=imputation,
                   coefficients=self.coefficients)
        if al.coverage < min_coverage:
            raise FeatureCoverageError(
                f"{self.clock.id}: {al.coverage:.1%} of its {len(self.features)} "
                f"sites are present, below the {min_coverage:.0%} floor.\n"
                "  Each site contributes its own division estimate and the "
                "result is their mean, so a fraction of the set is an estimate "
                "from a different set rather than a noisier one.")

        xp = spec.xp()
        x = spec.asarray(al.matrix)                       # samples x sites
        w = spec.asarray(self.coefficients)
        g = spec.asarray(self.ground_state)
        terms = (x - g[None, :]) * w[None, :]
        raw = 2.0 * xp.nanmean(terms, axis=1)

        # Counted, not corrected. A site under its fitted fetal methylation
        # contributes a negative division count, which is a statement about the
        # data rather than about the tissue.
        below = int(np.asarray(spec.tonumpy(x < g[None, :])).sum())
        if below:
            total = al.matrix.size
            get_logger(__name__).warning(
                 f"[{self.clock.id}] {below} of {total} site-by-sample values "
                 f"({below / total:.1%}) are below the fitted fetal ground "
                 "state, so those sites contribute a negative division count. "
                 "Usually a normalisation difference rather than a biological "
                 "one; the estimate is reported unchanged.")

        out = ops.apply_chain(raw, self.clock.postprocess, ops.POSTPROCESS, xp=xp)
        values = np.asarray(spec.tonumpy(out), dtype=np.float64).ravel()
        return pd.Series(values, index=data.sample_ids, name=self.clock.id), al

    @classmethod
    def from_registry(cls, registry, clock_id: str) -> DivisionClock:
        from ..registry.registry import DATA_DIR

        c = registry.get(clock_id)
        if clock_id in getattr(registry, "_local", {}):
            path = registry._local[clock_id][0]
        else:
            path = DATA_DIR / c.coefficient_source.file
        feats, weights, ground = read_division_parameters(path)
        return cls(clock=c, features=list(feats), coefficients=weights,
                   ground_state=ground)
