<!-- The Docker Hub repository overview for bhagesh/falconage.

     Docker Hub cannot generate this from the GitHub repository, so it is kept
     here and updated alongside the images. Paste it into the repository's
     Overview tab, or push it with docker/push_dockerhub.sh. Keep it short:
     it is a landing page, not the documentation, and the documentation is one
     link away. -->

# FALCONAge

Biological age and aging-clock scoring from DNA methylation, clinical
chemistry, proteomic and transcriptomic data. 161 catalogued clocks, 23 of them
scoring offline with coefficients bundled in the image. Python and R return the
same numbers, because the R package delegates every calculation to the same
Python core.

- **Source:** https://github.com/bhagesh-h/FALCONAge
- **Documentation:** https://bhagesh-h.github.io/FALCONAge/
- **Clock catalogue:** https://bhagesh-h.github.io/FALCONAge/clocks.html
- **Getting started:** https://bhagesh-h.github.io/FALCONAge/guide/FALCONAge.html

## Tags

| Tag | Base | Size |
|---|---|---|
| `latest`, `1.0.0-cpu` | `python:3.12-slim-bookworm` | 2.5 GB |
| `1.0.0-cuda` | `nvidia/cuda:12.4.1-runtime-ubuntu22.04` | 14.6 GB |

Both carry Python 3.12, R, and the package for each. **Take the CPU image
unless you know why you want the other one:** on the clocks that ship, CUDA is
*slower*, measured at 1.8x behind the CPU at 1,024 samples and 7.4x at 16,384,
because the matrix costs more to move than to multiply. `device="auto"` is CPU
even where a card exists. Details: https://bhagesh-h.github.io/FALCONAge/gpu.html

## Use it

```bash
docker pull bhagesh/falconage:1.0.0-cpu
```

The entrypoint is the `falconage` CLI, so the first argument is a verb:

```bash
# what this installation resolved to
docker run --rm bhagesh/falconage:1.0.0-cpu config

# score a beta matrix and write an HTML report with 31 figures
docker run --rm -v "$PWD:/work" -w /work bhagesh/falconage:1.0.0-cpu \
  report betas.csv --outdir results/

# scores only
docker run --rm -v "$PWD:/work" -w /work bhagesh/falconage:1.0.0-cpu \
  score --input betas.csv --outdir results/ --clocks compatible
```

`python`, `R`, `Rscript`, `pytest` and `bash` are passed through instead:

```bash
docker run --rm -it -v "$PWD:/work" -w /work bhagesh/falconage:1.0.0-cpu python
docker run --rm -it -v "$PWD:/work" -w /work bhagesh/falconage:1.0.0-cpu R
```

GPU, which needs the NVIDIA container toolkit on the host and no CUDA toolkit:

```bash
docker run --rm --gpus all -v "$PWD:/work" -w /work bhagesh/falconage:1.0.0-cuda \
  score --input betas.csv --outdir results/ --device cuda
```

On Windows PowerShell write `"${PWD}"`; on `cmd.exe`, `"%cd%"`.

## In Python

```python
import falconage as fa

data = fa.read_betas("betas.csv")
res  = fa.score(data, clocks="compatible")
res.interpretation()   # what each number is, and what may be done with it
res.manifest           # versions, device per clock, a digest of every coefficient file
```

## What it refuses to do

Refusals are the design. Scoring a clock off its training tissue is refused or
warned about, because saliva and buffy coat from the same people differ by 3.83
to 16.46 years while still correlating at Spearman 0.45 to 0.69. A cohort-centred
clock on a single sample is refused, because centring one row against itself
returns the intercept for everybody. Coverage is enforced twice, on the fraction
of probes present and on the fraction of coefficient mass they carry, because a
dataset can pass the first and fail the second.

Aging clocks are population-research instruments. A score is interpretable
against a comparison group; it is not a diagnostic statement about one person,
and no clock here is validated for clinical use.

GPL-3.0-or-later.
