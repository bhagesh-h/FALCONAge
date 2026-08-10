# GPU support: what was tested, on what, and what it is worth

Verified on real hardware on 2026-08-08. The short version: the GPU path works,
it is numerically sound, and for the clocks that ship today it is **slower than
the CPU**, so `device="auto"` resolves to CPU and the GPU is opt-in.

## The machine

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop, 8 GB, compute capability sm_89 (Ada) |
| Driver | 610.88, CUDA UMD 13.3 |
| Container runtime | Docker 29.6.1, GPU passthrough working via `--gpus all` |
| Second adapter | AMD Radeon 780M (integrated; not used - FALCONAge has no ROCm path) |

## What was already present, and what had to be installed

Nothing was installed on the host. Everything went into a throwaway container.

| Component | State | Action |
|---|---|---|
| NVIDIA driver 610.88 | already installed | none |
| CUDA user-mode driver 13.3 | already installed | none |
| Docker GPU passthrough | already working | none, `docker run --gpus all nvidia/cuda:12.4.1-base nvidia-smi` returned the card first try |
| CUDA toolkit on the host | **not installed, and not needed** | none, the torch wheel bundles its own runtime; only the driver has to be on the host |
| `torch` with CUDA | **missing** - the CPU image ships without it on purpose | installed `torch 2.6.0+cu124` inside a container, from `docker/Dockerfile.cuda`. Nothing was added to the host |
| `nvidia-container-toolkit` | already configured | none |

The driver reports CUDA 13.3 and the wheel is built against 12.4. That is the
supported direction: a newer driver runs an older runtime. The reverse does not
work, which is why the shipping `docker/Dockerfile.cuda` pins the runtime rather
than tracking the driver.

## 1. Device and dtype resolution

```
resolve('auto',    None,     fp64=False) -> numpy:cpu/float64
resolve('cuda',    None,     fp64=False) -> torch:cuda/float64
resolve('cuda',    float32,  fp64=False) -> torch:cuda/float32
resolve('cuda',    float32,  fp64=True)  -> torch:cuda/float64   + warning
resolve('mps')                            -> refused: no MPS device
```

The `requires_fp64` override fires as designed: a clock the registry flags as
ill-conditioned gets double precision back with a warning, rather than silently
honouring a `float32` request that would move its answer.

## 2. CPU and GPU do not agree bit for bit

This is the finding that matters most, and it qualifies a claim made elsewhere
in the documentation.

| clock | max abs difference | ulps |
|---|---:|---:|
| dnamphenoage | 9.9e-14 | 14 |
| horvath2013 | 1.3e-13 | 9 |
| lin | 6.4e-14 | 9 |
| skinandblood | 5.7e-14 | 4 |
| yingcausage | 8.5e-14 | 3 |
| hannum | 2.8e-14 | 2 |
| dnamtl | 1.8e-15 | 2 |
| zhangen | 1.4e-14 | 1 |

Worst case across eight clocks: **1.3e-13 years**, about four femtoseconds of
biological age.

The cause is ordinary: numpy's BLAS and cuBLAS sum a dot product in different
orders, and floating-point addition is not associative. It is not a defect and
it is not fixable without giving up the vendor kernels.

**What this means for the conformance guarantee.** FALCONAge claims R and Python
return the same bits. That claim is intact, both go through the same Python
core, and the R suite asserts equality at tolerance exactly zero. It is a
statement about the two *languages*, not about two *devices*. A CPU result and a
GPU result agree to about 1e-13 and are not bit-identical. The run manifest
records `device` and `dtype` for exactly this reason: a number is reproducible
against the manifest that produced it, not against every possible manifest.

Single precision costs far more: **7e-5 years** between `float32` and `float64`
on the same device. Still negligible against effects measured in single-digit
years, but four orders of magnitude worse than the device difference, which is
the right way round.

## 3. Speed: the GPU makes the shipping clocks slower

Eight clocks, 2,340 distinct features, RTX 4060, best of three runs, measured
inside `falconage:1.1.0-cuda` - the image the command at the foot of this page
builds. These are numbers you can reproduce, not numbers from a throwaway
environment.

| samples | CPU float64 | CUDA float64 | CUDA float32 | verdict |
|---:|---:|---:|---:|---|
| 128 | **0.009 s** | 0.011 s | 0.013 s | CPU wins |
| 1,024 | **0.032 s** | 0.053 s | 0.051 s | CPU wins by 1.6x |
| 4,096 | **0.143 s** | 0.307 s | 0.190 s | CPU wins by 2.1x |
| 16,384 | **0.506 s** | 2.328 s | 1.726 s | CPU wins by 4.6x |

The gap *widens* with size, which is the opposite of the usual shape. Profiling
says why:

```
4096 samples x 2340 features, 8 clocks
  feature alignment (pandas, CPU only)    0.134 s
  the dot products, on the GPU            0.010 s
  the dot products, on the CPU            0.005 s
```

Alignment is 28x the arithmetic it feeds, and at this size that arithmetic is
*faster on the CPU*: the GPU's 10 ms is mostly transfer, not multiply. At
16,384 samples it is 2.4 GB moved across PCIe to save nothing. A linear clock
over a few thousand features is simply too small a matrix multiplication to be
worth a device.

An earlier revision of this page reported 6.5x at 16,384 rather than 4.6x,
measured in a stripped test image with a different numpy. Neither the shape of
the answer nor the conclusion changed, but the figures above are what the
shipping image gives, and that is the only version worth quoting.

So `device="auto"` resolves to **CPU**, even when a GPU is present. Picking CUDA
because a card exists would make the common case several times slower on every
machine that has one, silently. The GPU is opt-in: `device="cuda"` or
`FALCONAGE_DEVICE=cuda`.

### When the GPU will be worth it

Not for the 23 tier A clocks. It should pay for itself on:

- **the PC clocks** - 78,464 features and coefficient tensors of 78 MB to 1.2 GB.
  Thirty times the feature count of the current largest, so the matmul stops
  being a rounding error.
- **neural architectures** - AltumAge is a multilayer perceptron, not a dot
  product, and has real depth to parallelise.
- **CpGPT** - a transformer, where the GPU is not optional.

None of those are tier A yet, which is exactly why this document can only report
that the path is correct rather than that it is fast.

## A performance bug this found

Profiling for the GPU comparison turned up something unrelated to GPUs. Feature
alignment was looping `X.iloc[:, i]` once per feature - 2,666 pandas indexing
calls for one eight-clock run, 76% of total runtime, against 0.5% for the
arithmetic they fed. Replacing the loop with a single `reindex` made the **CPU
path 2.1× faster** at 4,096 samples (0.358 s to 0.171 s), which benefits every
user whether or not they own a GPU.

It also means the earlier GPU figures were flattering: they were measured
against a slow CPU baseline. The table above is after the fix.

## Reproducing this

Every number on this page comes from one script, and it is in the repository:

```bash
docker build -f docker/Dockerfile.cuda -t falconage:1.1.0-cuda .
docker run --rm --gpus all -v "$PWD:/work" -w /work falconage:1.1.0-cuda \
  python test/gpu_check.py
```

`test/gpu_check.py` runs the five steps in the order above, what torch sees,
device and dtype resolution, CPU-versus-GPU agreement, the speed and precision
table, and the profile that explains it. Each step is meaningless if the one
before it failed, so it stops rather than continuing. On a machine with no CUDA
device it stops after step 1 with an explanation, which makes it safe to run
anywhere and safe to put in a bug report.

`--max-samples` caps the speed table for a card with less memory than the 8 GB
this was measured on; `--skip-profile` drops the last step.

If your card disagrees with the tables above: the tables are what should change.
