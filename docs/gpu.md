# GPU support: what was tested, on what, and what it is worth

Verified on real hardware, re-measured in full on 2026-08-10. The short version:
the GPU path works, it is numerically sound, and for the clocks that ship today
it is **slower than the CPU**, so `device="auto"` resolves to CPU and the GPU is
opt-in.

Every figure below comes from `falconage:1.1.0-cuda`, built from the shipping
`docker/Dockerfile.cuda`. The 2026-08-08 run of this page reported the same
shape on an earlier build; where a number moved, the current one is here and the
old one is named next to it.

## The machine

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop, 8 GB, compute capability sm_89 (Ada) |
| Driver | 610.88, CUDA UMD 13.3 |
| Container runtime | Docker 29.6.2, GPU passthrough working via `--gpus all` |
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

Per clock, not per run. This is a real run on the card: twenty methylation
clocks and one clinical clock, both scored with `device="cuda"` and combined.

::: {.falcon-output}
```
device / backend : mixed / mixed
device_requested : cuda
compute_summary  : torch:cuda/float64 (20 clocks), numpy:cpu/float64 (1 clock)
compute["METH:horvath2013"] : {device: cuda, dtype: float64, backend: torch}
compute["CLIN:phenoage"]    : {device: cpu,  dtype: float64, backend: numpy}
```
:::

Before this, that run reported `device="cuda"` flat, for a manifest in which
one of the twenty-one clocks had never touched the card.

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
| yingcausage | 8.5e-14 | 3 |
| horvath2013 | 5.7e-14 | 4 |
| hannum | 4.3e-14 | 3 |
| lin | 4.3e-14 | 6 |
| skinandblood | 4.3e-14 | 3 |
| zhangen | 1.4e-14 | 1 |
| dnamtl | 1.8e-15 | 2 |

Worst case across eight clocks: **9.9e-14 years**, about three femtoseconds of
biological age.

The cause is ordinary: numpy's BLAS and cuBLAS sum a dot product in different
orders, and floating-point addition is not associative. It is not a defect and
it is not fixable without giving up the vendor kernels.

The per-clock figures moved between the two runs, worst case 1.3e-13 on
2026-08-08 against 9.9e-14 now, horvath2013 the largest mover at 1.3e-13 to
5.7e-14. No code changed for these eight, which are all `LinearClock`; the
rebuilt image carries a different numpy and therefore a different BLAS, which
sums in a different order again. Numbers that move when the linear algebra
library moves are what this section is about, not an exception to it.

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

Eight clocks, 2,340 distinct features, RTX 4060, best of three timings inside a
run and the best of three runs, measured in `falconage:1.1.0-cuda` - the image
the command at the foot of this page builds. These are numbers you can
reproduce, not numbers from a throwaway environment.

| samples | CPU float64 | CUDA float64 | CUDA float32 | verdict |
|---:|---:|---:|---:|---|
| 128 | **0.007 s** | 0.013 s | 0.011 s | CPU wins by 1.9x |
| 1,024 | **0.026 s** | 0.048 s | 0.048 s | CPU wins by 1.8x |
| 4,096 | **0.109 s** | 0.333 s | 0.216 s | CPU wins by 3.1x |
| 16,384 | **0.445 s** | 3.275 s | 2.397 s | CPU wins by 7.4x |

The gap *widens* with size, which is the opposite of the usual shape. Profiling
says why:

```
4096 samples x 2340 features, 8 clocks
  feature alignment (pandas, CPU only)    0.132 s
  the dot products, on the GPU            0.010 s
  the dot products, on the CPU            0.005 s
```

Alignment is 28x the arithmetic it feeds, and at this size that arithmetic is
*faster on the CPU*: the GPU's 10 ms is mostly transfer, not multiply. At
16,384 samples it is 2.4 GB moved across PCIe to save nothing. A linear clock
over a few thousand features is simply too small a matrix multiplication to be
worth a device.

**Read the CUDA columns as a range, not a figure.** Three consecutive runs of
the same script on the same idle machine gave 3.275, 3.482 and 3.664 s at
16,384 samples, a 12% spread, while the CPU column moved by under 7% (0.445 to
0.477). This is a laptop card that clocks down as it warms. Each column above
is the best of the three, so the columns are not necessarily from one run;
computed within a run instead, the CPU margin at 16,384 was 6.9x, 7.7x and
7.8x. A desktop card with a power budget would be steadier and would still
lose, because what it is losing to is the transfer.

Two earlier revisions of this page reported 6.5x and then 4.6x at 16,384. The
gap has widened again to 7.4x, and the reason is not that the GPU got worse:
the CPU column improved from 0.506 s to 0.445 s on a rebuilt image with a newer
numpy, while the CUDA column got slower on the same hardware. The shape of the
answer has never changed across three measurements on two image builds, which
is worth more than any one of the figures.

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
docker run --rm -v "$PWD:/work" -w /work falconage:1.1.0-cuda \
  python -m pytest python/tests/unit/test_device_contract.py -q
```

**If the build fails at the dependency layer**, the pinned uv is older than the
`revision` field in `python/uv.lock` and refuses to parse it. That is what broke
both images between the 2026-08-08 and 2026-08-10 runs of this page.
`test/check_docker_lock.py` catches it in a second, and CI runs it, but the
symptom is worth recognising: ``error: Failed to parse `uv.lock` ``, six
minutes into a build.

`--max-samples` caps the speed table for a card with less memory than the 8 GB
this was measured on; `--skip-profile` drops the last step.

If your card disagrees with the tables above: the tables are what should change.
