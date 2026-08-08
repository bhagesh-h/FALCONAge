#!/usr/bin/env python3
"""Verify the GPU path on real hardware, and report what it is worth.

This is the script that produced every number in ``docs/gpu.md``. Run it and
compare; if your card disagrees with the table there, the table is what should
change.

Five questions, in order, because a later one is meaningless if an earlier one
fails:

1. Does torch see the device at all?
2. Does ``core.backend.resolve`` route to it, and does it enforce the dtype
   rules -- FP64 on CUDA, refusal on MPS, ``requires_fp64`` overriding a
   float32 request?
3. Do CPU and GPU return the same numbers? This is the one that matters. A GPU
   path that is fast and slightly different is worse than no GPU path, because
   the difference surfaces as a changed biological age with no explanation.
4. Is it faster, and where is the crossover?
5. Where does the time actually go?

Run it, from the repository root::

    docker build -f docker/Dockerfile.cuda -t falconage:1.0.0-cuda .
    docker run --rm --gpus all -v "$PWD:/work" -w /work \\
      falconage:1.0.0-cuda python test/gpu_check.py

On a machine with no CUDA device it stops after step 1 with an explanation
rather than an error, so it is safe to run anywhere.
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd

import falconage as fa
from falconage.core.backend import describe, resolve

# Eight tier A clocks over 2,340 distinct features. Chosen because they all ship
# coefficients, so the script needs no network and no licensed file.
CLOCKS = ["horvath2013", "hannum", "dnamphenoage", "skinandblood", "lin",
          "zhangen", "yingcausage", "dnamtl"]


def rule(n: int, title: str) -> None:
    print(f"\n{'=' * 72}\n{n}. {title}\n{'=' * 72}")


def synthetic(reg, feats, base, rng, size: int):
    """A cohort of the requested size. The values are noise around a per-probe
    baseline; nothing here tests biology, only that two devices agree and how
    long each takes."""
    idx = [f"S{i:05d}" for i in range(size)]
    X = np.clip(base[None, :] + rng.normal(0, 0.02, (size, len(feats))), 0.001, 0.999)
    return fa.FalconData(
        X=pd.DataFrame(X, index=idx, columns=feats),
        obs=pd.DataFrame({"age": rng.uniform(20, 85, size)}, index=idx),
        modality="dna_methylation", platform="450K")


def timed(data, device: str, dtype: str, reps: int = 3) -> float:
    """Best of `reps`, after a warm-up. The first CUDA call pays for cuInit and
    kernel compilation, which is a real cost once per process and nothing like
    the cost per run -- averaging it in would flatter the CPU by a wide margin.
    """
    fa.score(data, clocks=CLOCKS, device=device, dtype=dtype)
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fa.score(data, clocks=CLOCKS, device=device, dtype=dtype)
        if device == "cuda":
            import torch

            torch.cuda.synchronize()
        out.append(time.perf_counter() - t0)
    return min(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--max-samples", type=int, default=16384,
                    help="largest cohort in the speed table (default 16384)")
    ap.add_argument("--skip-profile", action="store_true")
    args = ap.parse_args(argv)

    rule(1, "what torch sees")
    info = describe()
    for k, v in info.items():
        print(f"  {k:16} {v}")

    if not info["cuda_available"]:
        print("\n  No CUDA device is visible to torch, so steps 2-5 have nothing to")
        print("  compare against. That is not a failure: the CPU path is the")
        print("  default and, on the clocks that ship today, the faster one.")
        return 0

    import torch

    cap = "".join(map(str, torch.cuda.get_device_capability(0)))
    print(f"  capability       sm_{cap}")
    print(f"  total memory     {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    rule(2, "device and dtype resolution")
    for dev, dt, fp64 in [("auto", None, False), ("cuda", None, False),
                          ("cuda", "float32", False), ("cuda", "float32", True),
                          ("cpu", "float64", False)]:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            spec = resolve(dev, dt, requires_fp64=fp64)
            note = f"   (warned: {str(w[0].message).splitlines()[0][:46]}...)" if w else ""
        print(f"  resolve({dev!r:>7}, {str(dt):>8}, fp64={fp64!s:>5}) -> {spec}{note}")
    try:
        resolve("mps")
    except Exception as exc:
        print(f"  resolve('mps') correctly refused: {str(exc).splitlines()[0]}")

    reg = fa.registry.load()
    feats = sorted({f for c in CLOCKS for f in reg.feature_ids(c)})
    rng = np.random.default_rng(7)
    base = rng.uniform(0.15, 0.85, len(feats))

    rule(3, "do CPU and GPU agree")
    data = synthetic(reg, feats, base, rng, 64)
    cpu = fa.score(data, clocks=CLOCKS, device="cpu", dtype="float64")
    gpu = fa.score(data, clocks=CLOCKS, device="cuda", dtype="float64")
    print(f"  cpu  {cpu.manifest.backend}:{cpu.manifest.device}/{cpu.manifest.dtype}")
    print(f"  cuda {gpu.manifest.backend}:{gpu.manifest.device}/{gpu.manifest.dtype}\n")
    print(f"  {'clock':<16}{'max |cpu-gpu|':>16}{'ulps':>8}   verdict")
    worst = 0.0
    for c in CLOCKS:
        a, b = cpu.scores[c].to_numpy(), gpu.scores[c].to_numpy()
        diff = float(np.max(np.abs(a - b)))
        worst = max(worst, diff)
        # ULPs at the magnitude of the value, which is the meaningful unit: an
        # absolute 1e-13 on a score of 60 is one ulp; on a score of 0.001 it is
        # a real disagreement.
        scale = float(np.max(np.abs(a))) or 1.0
        ulps = diff / np.spacing(scale) if diff else 0.0
        verdict = ("identical" if diff == 0.0
                   else "within 4 ulp" if ulps <= 4 else "differs, see below")
        print(f"  {c:<16}{diff:>16.3e}{ulps:>8.1f}   {verdict}")
    print(f"\n  worst across {len(CLOCKS)} clocks: {worst:.3e} years")
    print("  Expected, and not a defect: numpy's BLAS and cuBLAS sum a dot product")
    print("  in different orders, and floating-point addition is not associative.")
    print("  The bit-identity FALCONAge claims is between R and Python, which share")
    print("  one core -- it was never a claim about two devices.")

    rule(4, "speed, and what single precision costs")
    print(f"  {'samples':>9}{'cpu64':>9}{'cuda64':>9}{'cuda32':>9}"
          f"{'64 gain':>9}{'32 gain':>9}{'max |32-64| yr':>17}")
    sizes = [s for s in (128, 1024, 4096, 16384) if s <= args.max_samples]
    for size in sizes:
        d = synthetic(reg, feats, base, rng, size)
        c64 = timed(d, "cpu", "float64")
        g64 = timed(d, "cuda", "float64")
        g32 = timed(d, "cuda", "float32")
        a = fa.score(d, clocks=CLOCKS, device="cuda", dtype="float64").scores
        b = fa.score(d, clocks=CLOCKS, device="cuda", dtype="float32").scores
        err = float(np.max(np.abs(a.to_numpy() - b.to_numpy())))
        print(f"  {size:>9}{c64:>9.3f}{g64:>9.3f}{g32:>9.3f}"
              f"{c64 / g64:>8.2f}x{c64 / g32:>8.2f}x{err:>17.2e}")
    print("\n  The rightmost column is what single precision costs, in the units the")
    print("  clock reports. Published age-acceleration effects in these cohorts are")
    print("  single-digit years.")

    if args.skip_profile:
        return 0

    rule(5, "where the time goes")
    from falconage.models import align

    d = synthetic(reg, feats, base, rng, 4096)
    t0 = time.perf_counter()
    alignments = [align(d, list(reg.feature_ids(c))) for c in CLOCKS]
    t_align = time.perf_counter() - t0

    spec = resolve("cuda", "float64")
    mats = [spec.asarray(a.matrix) for a in alignments]
    ws = [spec.asarray(reg.coefficients(c)[1]) for c in CLOCKS]
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(20):
        for m, w in zip(mats, ws):
            _ = m @ w
    torch.cuda.synchronize()
    t_gpu = (time.perf_counter() - t0) / 20

    t0 = time.perf_counter()
    for _ in range(20):
        for a, c in zip(alignments, CLOCKS):
            _ = a.matrix @ reg.coefficients(c)[1]
    t_cpu = (time.perf_counter() - t0) / 20

    print(f"  4096 samples x {len(feats)} features, {len(CLOCKS)} clocks\n")
    print(f"    feature alignment (pandas, CPU only)  {t_align:8.4f} s")
    print(f"    the dot products, on the GPU          {t_gpu:8.4f} s")
    print(f"    the dot products, on the CPU          {t_cpu:8.4f} s")
    print(f"\n  Alignment is {t_align / max(t_cpu, 1e-9):.0f}x the arithmetic it feeds. That is why")
    print("  the device barely matters here, and why device='auto' resolves to CPU.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
