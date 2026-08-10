# GPU support: what was tested, on what, and what it is worth

Verified on real hardware on 2026-08-08. The short version: the GPU path works,
it is numerically sound, and for the clocks that ship today it is **slower than
the CPU**, so `device="auto"` resolves to CPU and the GPU is opt-in.

Section 2 was added on 2026-08-10, after an audit found that three of the 23
scoring clocks never reached the device at all and the manifest did not say so.
The routing was fixed and checked against the torch CPU backend; no hardware
measurement was repeated, so every timing on this page is still the 2026-08-08
run.

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

## 2. Which clocks the device actually reaches

`device="cuda"` is a request, and until 2026-08-10 it was granted more often in
the manifest than in the arithmetic. All six model classes take a `DeviceSpec`.
Of the five that can produce a score, three computed in numpy regardless, and
the scoring loop then recorded the request rather than the fact.

| Model class | Reaches the device | Registry entries it serves |
|---|---|---|
| `LinearClock` | yes | 20 of the 23 that score offline |
| `PCLinearClock` | yes | 19, none shipping weights |
| `AggregationClock` | yes, since 2026-08-10 | 6, none shipping weights |
| `NeuralClock` | yes, since 2026-08-10 | AltumAge |
| `ClinicalClock` | **no, by declaration** | PhenoAge, KDM, homeostatic dysregulation |
| `ScaffoldClock` | not applicable | 28 tier C entries; it raises rather than scoring |

**Why the clinical three decline.** A methylation clock reduces thousands of
probes. A clinical clock reduces nine markers: PhenoAge sums ten terms, KDM fits
one univariate regression per marker, HD inverts a 9×9 covariance. That is less
arithmetic than a CUDA kernel launch costs to dispatch, so a device
implementation would be slower, and it would pull torch into the one modality
that otherwise needs nothing beyond numpy. They set `CPU_ONLY = True`, which
`falconage.models.effective_spec()` reads, so the refusal is a declared property
rather than an argument quietly dropped.

**Why the other two were routed anyway.** `NeuralClock` is the one architecture
here where a device should pay, for the reason section 4 makes measurable: a
linear clock is a single dot product over a few thousand features and loses to
the CPU because the transfer costs more than the multiply, while AltumAge is
dense layers over 20,318 inputs with depth to parallelise. `AggregationClock`
takes a mean or a 95th percentile over a probe set and will not repay a device
on its own; it was routed so that what the manifest records is what happened.

### What the manifest says now

Per clock, not per run. This is a real mixed run: twenty methylation clocks and
one clinical clock, combined, on the torch backend.

::: {.falcon-output}
```
device / backend : cpu / mixed
compute_summary  : torch:cpu/float64 (20 clocks), numpy:cpu/float64 (1 clock)
compute["METH:horvath2013"] : {device: cpu, dtype: float64, backend: torch}
compute["CLIN:phenoage"]    : {device: cpu, dtype: float64, backend: numpy}
```
:::

A CUDA run has the same shape with `cuda` in place of `cpu` on the torch rows.
The backend split is what makes this one mixed, and it is the same split a card
produces, which is why it can be checked without one.

The three scalar fields are derived from `compute`: the shared value when every
clock agreed, and `"mixed"` when they did not. Before this they were assigned
once per clock inside the scoring loop, so a run reported whichever clock ran
last. The same overwrite made `dtype` wrong for any run combining a
`requires_fp64` clock with a `float32` request, which is all six PC clocks.

Two things make a run legitimately mixed. A `CPU_ONLY` model where the others
reached the device, as above; and `combine()` across datasets scored on
different machines, which is ordinary in a benchmark. The combined manifest
merges the per-clock records rather than copying the first run's device onto
all of them.

**This changes no score.** It changes what the run says about itself, which is
the whole of the reproducibility claim the next section rests on.

## 3. CPU and GPU do not agree bit for bit

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

## 4. Speed: the GPU makes the shipping clocks slower

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

Both of the first two now route through the device (section 2). That is a
correctness statement, not a speed one: no weights ship for either, so there is
nothing here to time. The point of routing them before the weights arrive is
that the alternative is discovering on the day somebody supplies a
`.safetensors` file that `device="cuda"` had been doing nothing all along.

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

The coverage table in section 2 is not from that script, because it needs no
hardware. It is asserted by `python/tests/unit/test_device_contract.py`, which
hands every model class a spec that counts how often it is reached and requires
either a non-empty count or a `CPU_ONLY` declaration. The torch arithmetic is
checked against numpy in the same file, on the CPU torch backend, so a runner
with no card still verifies everything except the transfer:

```bash
docker run --rm -e PYTHONPATH=/work/python/src -v "$PWD:/work" -w /work \
  falconage:1.1.0-cuda python -m pytest python/tests/unit/test_device_contract.py -q
```

`--max-samples` caps the speed table for a card with less memory than the 8 GB
this was measured on; `--skip-profile` drops the last step.

If your card disagrees with the tables above: the tables are what should change.
