# FALCONAge <img src="logo/logo.png" alt="FALCONAge logo" width="180" align="right"/>
<!-- badges: start -->
[![python-test](https://github.com/bhagesh-h/FALCONAge/actions/workflows/python-test.yaml/badge.svg)](https://github.com/bhagesh-h/FALCONAge/actions/workflows/python-test.yaml)
[![R-CMD-check](https://github.com/bhagesh-h/FALCONAge/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/bhagesh-h/FALCONAge/actions/workflows/R-CMD-check.yaml)
[![docs](https://github.com/bhagesh-h/FALCONAge/actions/workflows/docs.yaml/badge.svg)](https://bhagesh-h.github.io/FALCONAge/)
[![Docker Hub](https://img.shields.io/docker/v/bhagesh/falconage?label=docker&sort=semver)](https://hub.docker.com/r/bhagesh/falconage)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![R 4.1+](https://img.shields.io/badge/R-4.1%2B-blue)](https://www.r-project.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
<!-- badges: end -->

**F**ramework for **A**ging **CL**ocks, **O**mics **N**ormalisation and **Age** scoring.

FALCONAge applies 68 published human aging clocks to DNA methylation, clinical chemistry and NMR
metabolomics data, in Python and R. Every score comes back with its unit, where its coefficients
came from, and how much of it is measurement noise.

📖 **[Documentation](https://bhagesh-h.github.io/FALCONAge/)**

## Run it in three commands

You need [Docker](https://docs.docker.com/get-docker/) and nothing else.

```bash
# 1. get the image (once)
docker pull bhagesh/falconage:1.0.0-cpu

# 2. see it work on a small synthetic dataset, in about a minute
docker run --rm -v "$PWD:/work" bhagesh/falconage:1.0.0-cpu demo

# 3. run your own data
docker run --rm -v "$PWD:/work" bhagesh/falconage:1.0.0-cpu run betas.csv --samples samples.csv
```

Then open `betas_falconage/report.html`. On Windows PowerShell write `"${PWD}"` for `"$PWD"`; on
`cmd.exe`, `"%cd%"`.

`run` takes a beta matrix (`.csv`, `.tsv`, `.parquet`, `.h5ad`), a GEO series matrix, a folder of
raw Illumina IDATs, a clinical chemistry table (`--kind clinical`) or a Nightingale NMR export.
The sample sheet is optional: its first column is the sample id, then any of `age`, `sex`,
`tissue` and `group`.

→ [Your first run](https://bhagesh-h.github.io/FALCONAge/start/quickstart.html) explains every
file a run writes. [Install](https://bhagesh-h.github.io/FALCONAge/start/install.html) covers
Docker itself, the smaller Python-only image, and installing without Docker.

## What you get

- A coverage map, drawn first: one genome track per clock, showing which of its CpGs your data
  carries and which it shares with other clocks
- A score per sample for every one of the 68 human clocks your data supports, all of which ship
  their coefficients and run offline
- A **standard error on every score**, propagated from published per-probe reliability, and a
  prediction interval
- Age acceleration where the sample's age is known and the clock's scale permits it
- Whether a group difference holds across clocks or rests on one
- Publication-ready figures, each a `<title>.png` with no text on the canvas but axes and
  legend, and a `<title>.txt` beside it holding the title, the numbers and the description
- One HTML report, and a manifest recording versions, every coefficient file's SHA-256 and
  every warning, enough to reproduce the run

## From Python or R

```python
import falconage as fa

d   = fa.prepare(fa.read_betas("betas.csv"))   # or fa.read_idat_dir("idats/")
res = fa.score(d, clocks="compatible")
res.interpretation()          # scale, unit, permitted arithmetic, reliability, caveats
fa.technical_se(res, d)       # how much of each score is the assay
```

```r
library(FALCONAge)
d   <- prepare(read_betas("betas.csv"))
res <- score(d, clocks = "compatible")
interpretation(res)
technical_se(res, d)
```

The R package calls the same Python core, so R and Python give identical numbers, asserted at
tolerance zero in CI.

## What it refuses, and why

Each refusal is enforced in code and names the measurement behind it.

| Refused | Why, measured |
|---|---|
| Age acceleration on a pace, a division count, a log-hazard or a relative score | The subtraction is undefined for the scale. 29 of the 52 clocks that ship are on one of those scales, and a pace is already a rate |
| `predicted − chronological` on DamAge/AdaptAge | Slope against age is 0.967, but the offset swings **162 years** between cohorts against Horvath's 15, so the difference is not a quantity |
| A whole-blood clock on saliva | Saliva clock ages ran **3.83–16.46 years** above buffy coat in the same 91 people while still correlating at Spearman 0.45–0.69. Correlation is not agreement |
| Any array clock on cell-free DNA | Not a tissue but a fragment population shed from many. 12 clocks refuse a specimen outright; 127 more warn |
| A clock below the coverage **or** the coefficient-mass floor | 96% of probes present can be 61% of the model |
| Clinical chemistry with undeclared units | PhenoAge has an SI and a conventional variant that disagree by years, so `units=` has no default and `units="SI"` is not a unit |
| A `.pt` or `.pkl` file of weights | Unpickling a file executes arbitrary code. AltumAge ships because its weights were converted to safetensors once, in a build step, not because the refusal was relaxed |
| A per-probe weight from a network | AltumAge has layer matrices, not one weight per feature, so the probe-level standard error and the coefficient-mass floor say so instead of computing something else |

→ [About and design](https://bhagesh-h.github.io/FALCONAge/start/about.html)

## Known limitations

| Not implemented | Why |
|---|---|
| Cross-platform liftover (`mLiftOver`) | Mapping tables encode concordance measured on paired samples; not derivable from array manifests |
| Dye-bias correction **on by default** | Ships opt-in. On real IDATs it moves the median beta by +0.10 to +0.12, and a correct version needs control probes absent from the fetchable manifest |
| Proteomic or transcriptomic **clocks** | Readers and preparation chains ship, and `fit_clock` fits one; no published model may be redistributed |
| A foundation-model imputation backend | `NeuralClock` ships; CpGPT/MethylGPT as zero-shot probe imputation does not |
| meQTL awareness at score time | A clock CpG under strong genetic control carries variance fixed at conception. `coefficient_mass()` now measures each clock's exposure against a meQTL list you supply, but the registry stores no meQTL status and no score is adjusted for it, so a number still cannot say which part of its spread is acquired |
| Allele-specific methylation | A beta value averages both alleles, so an array cannot resolve it. Sequencing can; nothing here reads that |
| Co-methylation modules | Probes are treated independently. Module-level testing is what makes an epigenome-wide scan tractable, and is not implemented |

## Documentation

| Page | What is on it |
|---|---|
| [Get started](https://bhagesh-h.github.io/FALCONAge/) | Install, the first run, what the tool refuses |
| [Guide](https://bhagesh-h.github.io/FALCONAge/data/loading.html) | Loading data, preprocessing, acceleration, uncertainty, reports, building your own clocks |
| [Clocks](https://bhagesh-h.github.io/FALCONAge/clocks/choosing.html) | Choosing a clock, what a score means, the catalogue of 68 |
| [Methods](https://bhagesh-h.github.io/FALCONAge/methods/formulas.html) | The formulas FALCONAge implements, the raw-array chain, uncertainty, benchmarking |
| [Python](https://bhagesh-h.github.io/FALCONAge/reference/) · [R](https://bhagesh-h.github.io/FALCONAge/r/reference/) | Full API references |
| [Developers](https://bhagesh-h.github.io/FALCONAge/dev/index.html) | Architecture, testing, and how the images are built |

The site is also downloadable as a PDF or a markdown ZIP.

## Claude skill

The repository ships a [Claude](https://claude.com/claude-code) skill at
[`.claude/skills/falconage/`](.claude/skills/falconage/) carrying the commands, the clock catalogue
and every refusal, so Claude runs FALCONAge the way this documentation does. It is active inside a
clone; to use it anywhere, copy it into your personal skills folder:

```bash
cp -r .claude/skills/falconage ~/.claude/skills/
```

## Citation

Hunakunti, B. (2026). FALCONAge: Multiomic biological age and aging clock scoring in Python and R
(Version 1.0.0) [Computer software]. https://github.com/bhagesh-h/FALCONAge

In R, `citation("FALCONAge")`. The clock you scored is somebody else's work, and citing FALCONAge
does not cite it: `falconage clocks cite <clock>` prints its primary reference.

GPL-3.0-or-later.
