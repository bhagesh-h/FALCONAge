"""Neural clocks, and the honest shape of a foundation-model backend.

TWO DIFFERENT THINGS ARE OFTEN CALLED THE SAME NAME.

**A neural clock** is a small feed-forward network over a fixed probe set --
AltumAge is the canonical one, a handful of dense layers over 20,318 CpGs. It is
a clock: features in, an age out, and everything the registry says about scale
and legal operations applies unchanged. :class:`NeuralClock` runs one.

**A methylation foundation model** -- CpGPT, MethylGPT -- is not a clock. Its
value here is not another age prediction: it is zero-shot *imputation* of probes
an array does not carry, and conversion between array versions. That attacks
probe loss and cross-platform harmonisation, which are two of this package's
open problems, and it is a preprocessing step rather than a model.

Conflating them is easy and expensive. A foundation model asked for an age gives
one; that number is not comparable with a registry clock's, has no published
scale, and no coefficient file to checksum.

WHAT SHIPS HERE. The architecture, loaded from ``safetensors`` only. No weights
for any neural clock are redistributable, so this is a scaffold in exactly the
sense :class:`PCLinearClock` is: implemented, tested on synthetic weights, and
usable the day a user supplies a file.

WHY SAFETENSORS AND NOT A PICKLE. ``torch.load`` on an untrusted file executes
arbitrary code during unpickling. A clock's weights arrive by download from a
third party, which is precisely the threat model, and at least one widely used
aging-clock package accepts pickles today. ``safetensors`` is a flat tensor
container with no code path, and this module will not read anything else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.backend import DeviceSpec
from ..core.errors import FeatureCoverageError, ScoringError, WeightsUnavailableError
from ..registry.registry import Clock
from . import ops
from .linear import Alignment, align

__all__ = ["NeuralClock", "NeuralWeights", "is_neural", "read_neural_weights"]


@dataclass
class NeuralWeights:
    """A loaded feed-forward network: ordered (weight, bias) pairs."""

    layers: list[tuple[np.ndarray, np.ndarray]]
    features: list[str]
    activation: str = "relu"
    source: str = ""
    sha256: str = ""

    @property
    def n_parameters(self) -> int:
        return int(sum(w.size + b.size for w, b in self.layers))

    def __repr__(self) -> str:  # pragma: no cover - display only
        shapes = " -> ".join(str(w.shape[0]) for w, _ in self.layers)
        return (f"NeuralWeights({len(self.features)} features, {shapes} -> "
                f"{self.layers[-1][1].size}, {self.n_parameters:,} parameters)")


def is_neural(clock) -> bool:
    """True for the entries whose forward pass is a feed-forward network."""
    return "neural network" in (clock.model_type or "").lower()


def read_neural_weights(path: str | Path, *, features: list[str] | None = None,
                        activation: str = "relu") -> NeuralWeights:
    """Load layer weights from a ``.safetensors`` file.

    Tensors are paired by name: ``layer0.weight`` with ``layer0.bias`` and so
    on, ordered by the numeric part. A feature list may live in the file's
    metadata under ``features`` as newline-separated ids, or be supplied.

    Refuses anything that is not safetensors, by extension and by content. The
    alternative -- accepting a ``.pt`` because it happens to be there -- is the
    arbitrary-code path this module exists to close.
    """
    import hashlib
    import re

    p = Path(path)
    if p.suffix.lower() != ".safetensors":
        raise ScoringError(
            f"{p.name}: neural weights must be .safetensors.\n"
            "  torch.load executes arbitrary code while unpickling, and a clock's "
            "weights arrive by download from a third party. That is the threat "
            "model, not a hypothetical one.")
    try:
        from safetensors.numpy import load_file
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ScoringError(
            "reading neural weights needs safetensors: pip install safetensors"
        ) from exc

    tensors = load_file(str(p))
    # The metadata block, which this function has always documented and never
    # read: a file written with its own feature order was still refused for
    # want of one. safetensors keeps metadata out of load_file, so it takes a
    # second open.
    meta: dict[str, str] = {}
    try:
        from safetensors import safe_open

        with safe_open(str(p), framework="numpy") as fh:
            meta = fh.metadata() or {}
    except Exception:                      # pragma: no cover - metadata is optional
        meta = {}
    idx: dict[int, dict[str, np.ndarray]] = {}
    for name, arr in tensors.items():
        m = re.search(r"(\d+)", name)
        if m is None:
            continue
        slot = idx.setdefault(int(m.group(1)), {})
        slot["weight" if "weight" in name else "bias"] = np.asarray(arr, dtype=np.float64)

    layers = []
    for k in sorted(idx):
        part = idx[k]
        if "weight" not in part or "bias" not in part:
            raise ScoringError(f"{p.name}: layer {k} has "
                               f"{sorted(part)}; both weight and bias are needed")
        layers.append((part["weight"], part["bias"]))
    if not layers:
        raise ScoringError(
            f"{p.name}: no numbered weight/bias pairs found. Expected names like "
            "layer0.weight and layer0.bias.")

    feats = list(features) if features else [
        f for f in (meta.get("features", "").split(chr(10))) if f]
    activation = meta.get("activation", activation)
    if not feats:
        raise ScoringError(
            f"{p.name}: no feature list supplied.\n"
            "  A network without its probe order is not a clock -- the columns "
            "have to be in the order it was trained on, and nothing in the "
            "tensor file records that.")

    return NeuralWeights(layers=layers, features=feats, activation=activation,
                         source=str(p),
                         sha256=hashlib.sha256(p.read_bytes()).hexdigest())


@dataclass
class NeuralClock:
    """A feed-forward network over a fixed probe set.

    Forward pass only. Nothing here trains, and the architecture is deliberately
    the plainest one that fits published aging networks: dense layers, one
    activation, no dropout at inference, no batch norm state to get wrong.

    THE ONE ARCHITECTURE HERE WHERE A GPU EARNS ITS TRANSFER. A linear clock is
    a single dot product over a few thousand features, and ``docs/gpu.md``
    measures the card losing to the CPU on those by up to 4.6x because moving
    the matrix costs more than multiplying it. A network is a chain of dense
    layers with real depth: AltumAge is 20,318 inputs, so the arithmetic per
    byte transferred is an order of magnitude higher and the transfer amortises.
    The whole pass therefore runs through the ``xp`` handle, activations
    included, and honours ``device=`` like the linear path does.
    """

    clock: Clock
    weights: NeuralWeights
    scale: tuple[np.ndarray, np.ndarray] | None = None   # per-feature (mean, sd)
    notes: list[str] = field(default_factory=list)

    def _activate(self, x, xp):
        """Apply the activation through the backend handle.

        Written against the op catalogue rather than against ``np`` directly.
        ``relu`` as ``clip(low=0)`` reuses the one place that already knows
        numpy spells it ``clip`` and torch spells it ``clamp``, so this stays a
        single implementation instead of a second branch to keep in step.
        """
        a = self.weights.activation
        if a == "relu":
            return ops.clip(x, low=0.0, xp=xp)
        if a == "tanh":
            return xp.tanh(x)
        if a in ("sigmoid", "logistic"):
            return ops.expit(x, xp=xp)
        if a == "selu":
            # The two constants are not tuning: they are the fixed point that
            # makes the activation self-normalising, and they are printed in
            # Klambauer et al. 2017. A rounded copy would drift the output of
            # a five-layer network by more than the third decimal.
            alpha = 1.6732632423543772848170429916717
            scale = 1.0507009873554804934193349852946
            neg = alpha * (ops.exp_op(ops.clip(x, high=0.0, xp=xp), xp=xp) - 1.0)
            return scale * (ops.clip(x, low=0.0, xp=xp) + neg)
        if a in ("identity", "linear"):
            return x
        raise ScoringError(f"unknown activation {a!r}")

    def predict(self, data, spec: DeviceSpec, *, imputation: str = "reference",
                min_coverage: float = 0.8) -> tuple[pd.Series, Alignment]:
        al = align(data, self.weights.features, imputation=imputation)
        if al.coverage < min_coverage:
            raise FeatureCoverageError(
                f"{self.clock.id}: {al.coverage:.1%} of its "
                f"{len(self.weights.features)} features are present, below the "
                f"{min_coverage:.0%} floor.\n"
                "  A network has no per-feature weight to inspect, so there is "
                "no coefficient-mass check to fall back on here: the count is "
                "all there is.")

        xp = spec.xp()
        x = spec.asarray(al.matrix)
        if self.scale is not None:
            mu, sd = self.scale
            # The zero-variance guard is computed in numpy, on the host, before
            # anything moves: a constant feature has no scale to divide by and
            # substituting 1.0 leaves it centred, which is what the training
            # code did. Doing it on the device would be one kernel for a
            # decision that is already made.
            safe_sd = np.where(np.asarray(sd) == 0, 1.0, np.asarray(sd))
            x = (x - spec.asarray(mu)[None, :]) / spec.asarray(safe_sd)[None, :]

        last = len(self.weights.layers) - 1
        for i, (w, b) in enumerate(self.weights.layers):
            x = x @ spec.asarray(w).T + spec.asarray(b)[None, :]
            if i != last:
                x = self._activate(x, xp)

        # reshape rather than ravel: both libraries have it and both mean the
        # same thing by it, which ravel does not guarantee across the two.
        raw = x.reshape(-1)
        out = ops.apply_chain(raw, self.clock.postprocess, ops.POSTPROCESS, xp=xp)
        return pd.Series(np.asarray(spec.tonumpy(out), dtype=np.float64).ravel(),
                         index=data.sample_ids, name=self.clock.id), al

    @classmethod
    def from_registry(cls, registry, clock_id: str) -> NeuralClock:
        c = registry.get(clock_id)
        # A network whose weights ship. AltumAge is the first: its authors
        # publish under MIT, and python/tools/build_altumage_weights.py folds
        # the published Keras model and its scaler into the safetensors file
        # named here. The pickle stays in that build step, where a human runs
        # it once, and never reaches score time.
        src = c.coefficient_source.file or ""
        if src.endswith(".safetensors"):
            from ..registry.registry import DATA_DIR

            path = DATA_DIR / src
            if not path.exists():
                raise WeightsUnavailableError(
                    clock_id, f"{clock_id}: the registry declares {path.name} "
                              "and it is missing from the installed package")
            weights = read_neural_weights(path)
            return cls(clock=c, weights=weights)
        raise WeightsUnavailableError(
            clock_id,
            f"{clock_id} is a neural clock and no weights ship with FALCONAge.\n"
            "  The architecture is implemented and tested. Supply a "
            ".safetensors file and build the model directly:\n"
            "    w = fa.models.read_neural_weights(path, features=[...])\n"
            "    m = fa.models.NeuralClock(clock=reg.get(%r), weights=w)\n"
            "  %s" % (clock_id, registry.unavailable_message(clock_id)
                      if c.availability == "C" else ""))
