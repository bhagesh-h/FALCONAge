"""Principal-component clocks: rotate first, then take the linear combination.

A PC clock is not a variant of a linear clock with more features. The published
form is

    score = ((x - centre) @ rotation) @ pc_coefficients   then the usual chain

where ``rotation`` is a CpG-by-component matrix fitted on a large reference
panel. The clock's weights live in *component* space, and no weight is
attributable to any single probe.

WHY THE FIELD BOTHERS. Higgins-Chen et al. (Nature Aging 2022;2:644-661) rebuilt
six clocks this way to attack technical noise. The original clocks put large
weights on individual CpGs, and an array measures an individual CpG with enough
error to move a prediction by up to nine years between replicates of the same
sample. Averaging tens of thousands of probes into components dilutes that: the
same replicate pairs then agree to within about a year. It is also why PC clocks
largely shrug off EPICv2 probe loss while the originals do not -- losing a probe
perturbs a component slightly instead of removing a term outright.

WHAT THIS COSTS, AND WHY IT MATTERS HERE. The rotation is the model. PCHorvath1
carries 78,464 CpGs against Horvath2013's 353, and the matrices run from 78 MB
to about 1.2 GB. That is two things at once: the first architecture in this
package where a GPU should actually earn its transfer cost (see docs/gpu.md,
where the shipping clocks are *slower* on CUDA because a 2,340-feature matmul is
a rounding error next to the alignment that feeds it), and the reason a
coefficient CSV cannot express one.

STATUS. The architecture is implemented and tested against synthetic rotations.
No PC clock ships coefficients: the published rotations are research-use-only
and are not ours to redistribute. ``register_local_weights_npz`` takes a file
you have obtained yourself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..core.backend import DeviceSpec
from ..core.errors import FeatureCoverageError, RegistryError
from ..registry.registry import Clock
from . import ops
from .linear import Alignment, align


@dataclass
class PCRotation:
    """The fitted reference basis: which probes, centred where, rotated how."""

    features: list[str]
    centre: np.ndarray        # (n_features,)
    rotation: np.ndarray      # (n_features, n_components)
    coefficients: np.ndarray  # (n_components,)

    def __post_init__(self) -> None:
        n, k = len(self.features), self.rotation.shape[1]
        if self.rotation.shape[0] != n:
            raise RegistryError(
                f"rotation has {self.rotation.shape[0]} rows for "
                f"{n} features")
        if self.centre.shape != (n,):
            raise RegistryError(
                f"centre has shape {self.centre.shape}, expected ({n},)")
        if self.coefficients.shape != (k,):
            raise RegistryError(
                f"{k} components in the rotation but "
                f"{self.coefficients.shape[0]} coefficients")

    @property
    def n_components(self) -> int:
        return self.rotation.shape[1]

    @property
    def nbytes(self) -> int:
        return int(self.rotation.nbytes + self.centre.nbytes
                   + self.coefficients.nbytes)


def read_rotation(path) -> PCRotation:
    """Load a rotation from ``.npz``.

    Four arrays: ``features`` (string), ``centre``, ``rotation``,
    ``coefficients``. A CSV of feature/coefficient pairs -- the format every
    other clock here uses -- cannot express a 78,464 x 121 matrix in any form
    worth parsing, so this is a deliberately different door rather than an
    overloaded one.
    """
    with np.load(path, allow_pickle=False) as z:
        missing = {"features", "centre", "rotation", "coefficients"} - set(z.files)
        if missing:
            raise RegistryError(
                f"{path}: missing {', '.join(sorted(missing))}. A PC rotation "
                "needs features, centre, rotation and coefficients.")
        return PCRotation(
            features=[str(f) for f in z["features"]],
            centre=np.asarray(z["centre"], dtype=np.float64),
            rotation=np.asarray(z["rotation"], dtype=np.float64),
            coefficients=np.asarray(z["coefficients"], dtype=np.float64),
        )


@dataclass
class PCLinearClock:
    """``postprocess(((x - centre) @ rotation) @ pc_coefficients)``."""

    clock: Clock
    rotation: PCRotation

    def predict(self, data, spec: DeviceSpec, *, imputation: str = "reference",
                min_coverage: float = 0.8) -> tuple[pd.Series, Alignment]:
        feats = self.rotation.features
        al = align(data, feats, imputation=imputation)

        # No mass floor here, and its absence is the point. Coefficient mass is
        # defined per feature, and a PC clock has no per-feature coefficient --
        # the weights are in component space. Reporting one would mean
        # attributing a component's weight back to its probes, which is exactly
        # the attribution PCA destroys. Feature coverage still applies.
        if al.coverage < min_coverage:
            raise FeatureCoverageError(
                f"{self.clock.id}: {al.coverage:.1%} of its {len(feats)} "
                f"features are present, below the {min_coverage:.0%} floor.\n"
                f"  The dataset is {data.platform or 'an unknown platform'} and "
                f"this clock was trained on "
                f"{', '.join(self.clock.platform) or 'unknown'}.\n"
                "  PC clocks tolerate probe loss better than their originals, "
                "because a missing probe perturbs a component rather than "
                "removing a term -- but they are not immune, and the rotation "
                "was fitted with every probe present."
            )

        xp = spec.xp()
        x = spec.asarray(al.matrix)
        centre = spec.asarray(self.rotation.centre)
        rot = spec.asarray(self.rotation.rotation)
        w = spec.asarray(self.rotation.coefficients)

        x = ops.apply_chain(x, self.clock.preprocess, ops.PREPROCESS, xp=xp)
        raw = ((x - centre) @ rot) @ w
        out = ops.apply_chain(raw, self.clock.postprocess, ops.POSTPROCESS, xp=xp)

        values = np.asarray(spec.tonumpy(out), dtype=np.float64).ravel()
        return pd.Series(values, index=data.sample_ids, name=self.clock.id), al

    @classmethod
    def from_rotation(cls, clock: Clock, path) -> PCLinearClock:
        return cls(clock=clock, rotation=read_rotation(path))
