#!/usr/bin/env python3
"""Turn the published AltumAge model into weights this package will load.

WHY THIS SCRIPT EXISTS. AltumAge's authors publish the trained model under MIT,
so the licence was never the obstacle. The format was. What they ship is a
Keras ``.h5``, a scikit-learn ``RobustScaler`` pickle and a pandas pickle of
the CpG order, and this package refuses to load a pickle at score time because
unpickling executes arbitrary code and a clock's weights arrive by download
from a third party. So the conversion happens here, once, deliberately, and
what ships is a ``.safetensors`` file that cannot execute anything.

WHAT IT DOES ABOUT THE PICKLES. It reads them with an Unpickler whose
``find_class`` accepts an explicit list of names and raises on everything else.
That is not the same as trusting the file: a pickle that tries to reach for
anything outside the list stops the build rather than running. The list is
below, and it is short on purpose. Writing this was also how the scaler turned
out to be a ``RobustScaler`` -- median and interquartile range -- rather than
the ``StandardScaler`` the repository's own notebook calls it.

WHAT IT DOES ABOUT THE ARCHITECTURE. AltumAge is

    input -> BN -> Dense(32) -> selu -> BN -> Dense(32) -> selu -> ... -> BN -> Dense(1)

with a robust scaler in front. Batch normalisation at inference is an affine
map per feature, and so is the scaler, and every one of them sits immediately
before a dense layer. An affine map followed by a dense layer is a dense layer:

    x' = a * x + c          (scaler, or batch norm at inference)
    y  = x' @ W + b  =  x @ (a[:, None] * W) + (c @ W + b)

So the six batch-norm layers and the scaler fold into the six dense layers, and
what ships is an ordinary feed-forward stack that ``NeuralClock`` already runs.
Nothing is approximated: the fold is exact arithmetic, and the script checks it
against an unfolded reference implementation before writing anything.

Usage
-----
    python python/tools/build_altumage_weights.py            # download and build
    python python/tools/build_altumage_weights.py --check    # verify the shipped file
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "python" / "src" / "falconage" / "registry" / "data" / "coefficients"
RAW = "https://raw.githubusercontent.com/rsinghlab/AltumAge/main/example_dependencies/"
FILES = ("AltumAge.h5", "scaler.pkl", "multi_platform_cpgs.pkl")
EPS = 1e-3          # the epsilon in the model's own batch-norm config


class Restricted(pickle.Unpickler):
    """A pickle reader that refuses anything not on this list.

    Every name here is a container or an array: numpy's array reconstruction,
    the two pandas types the CpG list is stored in, the scaler class itself,
    and `slice`, which pandas uses to describe a block's extent. There is no
    callable on the list that does anything but build data.
    """

    ALLOWED = {
        ("builtins", "slice"),
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy.core.numeric", "_frombuffer"),
        ("numpy.dtypes", "Float64DType"),
        ("numpy.dtypes", "ObjectDType"),
        ("pandas._libs.internals", "_unpickle_block"),
        ("pandas.core.indexes.base", "Index"),
        ("pandas.core.indexes.base", "_new_Index"),
        ("pandas.core.indexes.range", "RangeIndex"),
        ("pandas.core.internals.blocks", "new_block"),
        ("pandas.core.internals.managers", "SingleBlockManager"),
        ("pandas.core.series", "Series"),
        ("sklearn.preprocessing._data", "RobustScaler"),
    }

    def find_class(self, module: str, name: str):
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(
                f"{module}.{name} is not on the allow-list. The file changed, "
                "or it is not the file it claims to be; either way this build "
                "stops rather than running it.")
        return super().find_class(module, name)


def fetch(cache: Path) -> dict[str, Path]:
    cache.mkdir(parents=True, exist_ok=True)
    got = {}
    for name in FILES:
        p = cache / name
        if not p.exists():
            print(f"  fetching {name}")
            with urllib.request.urlopen(RAW + name, timeout=120) as r:
                p.write_bytes(r.read())
        got[name] = p
    return got


def affine_from_batchnorm(g, b, mm, mv):
    """Batch norm at inference, as a scale and an offset per feature."""
    inv = g / np.sqrt(mv + EPS)
    return inv, b - inv * mm


def build(cache: Path, out: Path) -> None:
    import h5py
    from safetensors.numpy import save_file

    files = fetch(cache)

    with open(files["scaler.pkl"], "rb") as fh:
        scaler = Restricted(fh).load()
    assert type(scaler).__name__ == "RobustScaler", type(scaler).__name__
    centre = np.asarray(scaler.center_, dtype=np.float64)
    spread = np.asarray(scaler.scale_, dtype=np.float64)

    with open(files["multi_platform_cpgs.pkl"], "rb") as fh:
        cpgs = [str(c) for c in np.asarray(Restricted(fh).load()).ravel()]
    assert len(cpgs) == len(set(cpgs)) == 20318, len(cpgs)
    assert len(centre) == len(cpgs)

    with h5py.File(files["AltumAge.h5"], "r") as f:
        w = f["model_weights"]

        def dense(i):
            g = w[f"dense_{i}"][f"dense_{i}"]
            return (np.asarray(g["kernel:0"], dtype=np.float64),
                    np.asarray(g["bias:0"], dtype=np.float64))

        def bn(i):
            g = w[f"batch_normalization_{i}"][f"batch_normalization_{i}"]
            return tuple(np.asarray(g[k], dtype=np.float64) for k in
                         ("gamma:0", "beta:0", "moving_mean:0", "moving_variance:0"))

        denses = [dense(i) for i in range(84, 90)]
        bns = [bn(i) for i in range(84, 90)]

    # ---- fold ------------------------------------------------------------
    layers = []
    for k, ((W, b), (g, beta, mm, mv)) in enumerate(zip(denses, bns)):
        a, c = affine_from_batchnorm(g, beta, mm, mv)
        if k == 0:
            # The scaler sits in front of the first batch norm, so its affine
            # composes with it before the dense layer sees anything.
            a, c = a / spread, c - a * centre / spread
        layers.append((a[:, None] * W, c @ W + b))

    # ---- check the fold against the unfolded chain ------------------------
    rng = np.random.default_rng(0)
    x0 = rng.uniform(0.0, 1.0, size=(4, len(cpgs)))

    def selu(x):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * np.where(x > 0, x, alpha * (np.exp(np.minimum(x, 0.0)) - 1.0))

    ref = (x0 - centre) / spread
    for k, ((W, b), (g, beta, mm, mv)) in enumerate(zip(denses, bns)):
        a, c = affine_from_batchnorm(g, beta, mm, mv)
        ref = ref * a + c
        ref = ref @ W + b
        if k != len(denses) - 1:
            ref = selu(ref)
    folded = x0
    for k, (W, b) in enumerate(layers):
        folded = folded @ W + b
        if k != len(layers) - 1:
            folded = selu(folded)
    gap = float(np.abs(ref - folded).max())
    print(f"  fold checked against the unfolded chain: max |delta| = {gap:.3e}")
    if gap > 1e-8:
        raise SystemExit("the folded network does not match the unfolded one")

    # ---- write -----------------------------------------------------------
    # (out, in), which is the orientation NeuralClock multiplies in; Keras
    # stores the transpose.
    tensors = {}
    for i, (W, b) in enumerate(layers):
        tensors[f"layer{i}.weight"] = W.T.astype(np.float32)
        tensors[f"layer{i}.bias"] = b.astype(np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out), metadata={
        "features": "\n".join(cpgs),
        "activation": "selu",
        "source": "https://github.com/rsinghlab/AltumAge (MIT)",
        "note": ("batch norm and the published RobustScaler are folded into the "
                 "dense layers; both are affine at inference"),
    })
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"  wrote {out.name}: {out.stat().st_size / 1e6:.1f} MB, "
          f"{sum(t.size for t in tensors.values()):,} parameters")
    print(f"  sha256 {digest}")


def check(out: Path) -> int:
    if not out.exists():
        print(f"{out} is missing; run this script without --check")
        return 1
    from safetensors import safe_open

    with safe_open(str(out), framework="numpy") as f:
        meta = f.metadata()
        feats = meta["features"].split("\n")
        keys = sorted(f.keys())
    print(f"{out.name}: {len(feats)} features, {len(keys) // 2} layers, "
          f"activation {meta['activation']}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--cache", default=None, help="where to keep the downloads")
    args = ap.parse_args()
    target = OUT / "altumage.safetensors"
    if args.check:
        sys.exit(check(target))
    cache = Path(args.cache) if args.cache else Path.home() / ".cache" / "falconage-altumage"
    build(cache, target)
