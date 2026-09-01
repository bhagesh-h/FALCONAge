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

Aging-clock scoring from DNA methylation, clinical chemistry, proteomic and transcriptomic data,
in Python and R, on CPU or GPU.

📖 **[Full documentation](https://bhagesh-h.github.io/FALCONAge/)**. This file is a summary; every
section links to the page that covers it properly.

## Overview

Reads raw or public data, applies published aging clocks, and returns a score per sample **with
the unit, the provenance, and how much of the number is measurement noise**.

The R and Python interfaces call one numerical core, so results are bit-identical, asserted at
tolerance zero in CI rather than approximately.

| | What ships today |
|---|---|
| **Clocks catalogued** | 175. **46 score offline**: 43 ship a weights file, 3 are formulas with none to ship. 40 are `licensed`, implemented but awaiting a file you supply; 89 are `untraced`, with no primary coefficient source established |
| **Inputs** | Raw Illumina IDATs (27K/450K/EPIC v1/v2), beta matrices, GEO series matrices, ComputAgeBench studies, RRBS, nanopore bedMethyl, targeted panels, clinical chemistry, Olink NPX, SomaScan RFU, bulk RNA-seq counts. [How to load each](https://bhagesh-h.github.io/FALCONAge/guide/loading.html) |
| **Normalisation** | pOOBAH detection, noob background correction, BMIQ, published probe masks |
| **Uncertainty** | Technical standard error per score, distribution-free prediction intervals, sample-size calculation |
| **Tested** | 457 Python tests and 52 R tests passing, 5 skipped, including the R-to-Python conformance pass at tolerance zero |

Every clock algorithm is implemented from its published description; no clock implementation is
imported from another package. Coefficients are fitted data rather than a procedure, and 40 clocks
have coefficients that are research-use-only. Those ship as tested scaffolds and take a file you
supply.
→ [Clock catalogue](https://bhagesh-h.github.io/FALCONAge/clocks.html) ·
[Choosing a clock](https://bhagesh-h.github.io/FALCONAge/guide/clocks.html)

## Installation

**Docker is the supported path.** One image carries Python, R, the CLI and all 43 bundled weight files
at pinned versions, so the same input gives the same numbers on any machine. Nothing else to
install.

```bash
docker pull bhagesh/falconage:1.0.0-cpu
```

Or build the same image from source, which needs a clone and about ten minutes:

```bash
git clone https://github.com/bhagesh-h/FALCONAge.git
cd FALCONAge
docker build -f docker/Dockerfile.cpu -t bhagesh/falconage:1.0.0-cpu .
```

Score a dataset and get an HTML report with 31 figures, no code written:

```bash
docker run --rm -v "$PWD:/work" -w /work bhagesh/falconage:1.0.0-cpu \
  report betas.csv --outdir results/
```

Python or R inside the same image:

```bash
docker run --rm -it -v "$PWD:/work" -w /work bhagesh/falconage:1.0.0-cpu python
docker run --rm -it -v "$PWD:/work" -w /work bhagesh/falconage:1.0.0-cpu R
```

The image's entrypoint is `falconage` itself, so the first argument is a verb.
Write `report`, not `falconage report`.

On Windows PowerShell write `"${PWD}"`; on `cmd.exe`, `"%cd%"`. For GPU, pull
`bhagesh/falconage:1.0.0-cuda` and add `--gpus all`. Measured, CUDA is
[slower than CPU](docs/gpu.md) for the linear clocks that ship today.

→ [Step-by-step Docker walkthrough](https://bhagesh-h.github.io/FALCONAge/guide/FALCONAge.html),
written for someone who has not used a terminal.

<details>
<summary><b>Native install, if you already manage your own environment</b></summary>

```bash
pip install "falconage @ git+https://github.com/bhagesh-h/FALCONAge.git#subdirectory=python"
```

```r
remotes::install_github("bhagesh-h/FALCONAge", subdir = "r")
FALCONAge::falconage_install()   # builds the Python environment R calls into
```

The R package delegates every number to the Python core, so it needs a Python environment either
way. `falconage_install()` creates a pinned one; on reticulate ≥ 1.41 an ephemeral one is built on
first use instead.

</details>

## Quick start

```python
import falconage as fa

d   = fa.prepare(fa.read_betas("betas.csv"))   # or fa.read_idat_dir("idats/")
res = fa.score(d, clocks="compatible")

res.interpretation()          # scale, unit, legal operations, reliability, caveats
fa.technical_se(res, d)       # how much of each score is the assay

fa.entropy(d)                 # disorder, without predicting an age
```

```r
library(FALCONAge)

d   <- prepare(read_betas("betas.csv"))
res <- score(d, clocks = "compatible")

interpretation(res)
technical_se(res, d)
```

## Claude skill

The repository ships a [Claude](https://claude.com/claude-code) skill at
[`.claude/skills/falconage/`](.claude/skills/falconage/). It carries the Docker commands, the
clock catalogue routed by the question each clock answers, and every refusal with the measurement
behind it, so Claude scores a dataset the way the documentation says to rather than inventing an
API.

It is already active for anyone working inside a clone. To use it anywhere on your machine, copy
the folder into your personal skills directory:

```bash
cp -r .claude/skills/falconage ~/.claude/skills/            # macOS, Linux
```

```powershell
Copy-Item -Recurse .claude\skills\falconage $HOME\.claude\skills\   # Windows
```

Then ask in plain language. *"Score these IDATs and tell me which clocks you refused and why"*
loads the skill. Every `fa.*` name and CLI verb in it is checked against the running package by
`docs/check_api_docs.py` on each push, because a wrong name in a skill does not mislead a reader
into checking; it becomes a command.

## Out-of-scope use

Refusals are the design, not the edge cases. Each names the measurement or the
mechanism behind it, and each is enforced in code rather than documented as advice.

| Refused | Why, measured |
|---|---|
| Age acceleration on a pace, a division count, a log-hazard or a relative score | The subtraction is undefined for the scale. 23 of the 46 clocks that ship are on one of those scales, and a pace is already a rate |
| `predicted − chronological` on DamAge/AdaptAge | Slope against age is 0.967, but the offset swings **162 years** between cohorts against Horvath's 15, so the difference is not a quantity |
| A whole-blood clock on saliva | Saliva clock ages ran **3.83–16.46 years** above buffy coat in the same 91 people while still correlating at Spearman 0.45–0.69. Correlation is not agreement |
| Any array clock on cell-free DNA | Not a tissue but a fragment population shed from many. 12 clocks refuse a specimen outright; 127 more warn |
| A clock below the coverage **or** the coefficient-mass floor | 96% of probes present can be 61% of the model |
| Clinical chemistry with undeclared units | PhenoAge has an SI and a conventional variant that disagree by years, so `units=` has no default and `units="SI"` is not a unit |
| A `.pt` or `.pkl` file of weights | `torch.load` executes arbitrary code while unpickling. AltumAge ships because its weights were converted to safetensors once, in a build step, not because the refusal was relaxed |
| A per-probe weight from a network | AltumAge has layer matrices, not one weight per feature, so the probe-level standard error and the coefficient-mass floor say so instead of computing something else |

Two more are implemented and quiet today. `requires_cohort` refuses a
cohort-centred clock handed a single sample, and no clock currently ships with
it set. And every clock that disagrees with its own paper carries the
discrepancy as a warning at score time: Weidner's third CpG is a substitution
the paper did not make, and it says so on every run.

→ [The science, §19](https://bhagesh-h.github.io/FALCONAge/science.html)

## Release highlights

- **Raw IDATs end to end**, validated at **r = 0.99928** against the published betas for the same
  physical samples (median |Δ| = 0.011, 99.7% of probes within 0.05).
- **Technical standard error on every score.** Horvath 2013 comes out at **±1.58 years** on the
  test corpus (implied cohort ICC 0.98); DunedinPoAm38 is least repeatable at 0.72.
- **Frozen-reference batch correction.** Standard ComBat moves already-reported scores by up to
  2.20 years when a plate is added; this makes earlier plates bit-identical.
- **Probe loss priced in years** per clock per platform. `hrsinchphenoage` shifts +16.7 years on
  EPIC v2; two clocks that lose nothing shift exactly 0.00.
- **Readouts that are not clocks.** Methylome entropy, per-sample drift, the Mei noise barometer
  and the Brown-Forsythe selection behind it. Tong et al. showed 66 to 75% of Horvath's accuracy
  is reproducible by drift alone, so the stochastic part is worth reading directly.
- **Repertoire structure**, the covariate no blood clock carries. Clone-size diversity metrics, and
  a simulator that holds cell fractions fixed while clone structure varies. Across 42 clocks the
  measured spread falls as `N_eff^-0.494` against a derived `-0.5`, and `dnamphenoage` moves
  **4.35 years** on clone structure alone.
- **Trait, state and technical variance separated.** `variance_components()` returns an ICC and an
  age-adjusted ICC; the gap between them is how much of a published ICC was the cohort's age range.
- Conformal prediction intervals, `power()`, `consensus()`, probe masks, BMIQ, `AggregationClock`,
  `NeuralClock`, `scAge`, proteomic and transcriptomic chains.

→ [CHANGELOG](CHANGELOG.md) · [Architecture §15](https://bhagesh-h.github.io/FALCONAge/architecture.html)
· [Beyond the clock score](https://bhagesh-h.github.io/FALCONAge/beyond-clocks.html)

## Known limitations

| Not implemented | Why |
|---|---|
| Cross-platform liftover (`mLiftOver`) | Mapping tables encode concordance measured on paired samples; not derivable from array manifests |
| Dye-bias correction **on by default** | Ships opt-in. On real IDATs it moves the median beta by +0.10 to +0.12, and a correct version needs control probes absent from the fetchable manifest |
| Proteomic or transcriptomic **clocks** | Readers and preparation chains ship; no catalogue entry, because organAging and tAge are both licence-restricted |
| A foundation-model imputation backend | `NeuralClock` ships; CpGPT/MethylGPT as zero-shot probe imputation does not |
| 89 untraced coefficient sources | A per-clock literature hunt; some have no public supplement |
| meQTL awareness at score time | A clock CpG under strong genetic control carries variance fixed at conception. `coefficient_mass()` now measures each clock's exposure against a meQTL list you supply, but the registry stores no meQTL status and no score is adjusted for it, so a number still cannot say which part of its spread is acquired |
| Allele-specific methylation | A beta value averages both alleles, so an array cannot resolve it. Sequencing can; nothing here reads that |
| Co-methylation modules | Probes are treated independently. Module-level testing is what makes an epigenome-wide scan tractable, and is not implemented |

## Figures

26 figure types, generated from the public corpus rather than drawn by hand.
→ [**Figure gallery**](https://bhagesh-h.github.io/FALCONAge/gallery.html)

![Every clock, every pooled study, one figure](test/output_figures/gallery/clock_atlas.png)

## Choosing clocks by what they were trained on

[`overlap.csv`](overlap.csv) is one row per clock, 39 columns, built to answer the question the
catalogue answers backwards: *which clocks were fitted on the kind of data I have?*

```python
import pandas as pd

d = pd.read_csv("overlap.csv")
d[(d.tissue_class == "whole_blood")
  & (d.species == "Homo sapiens")
  & (d.population_class.isin(["adults", "older_adults"]))
  & (d.target_class == "chronological_age")
  & (d.ships_coefficients == "yes")][["clock_id", "n_features", "verify_url"]]
```

Read it with a CSV parser rather than `awk -F,` or `cut`: citations and several
metadata fields contain commas, so a naive split silently shifts the columns.

[`overlap_col_desc.csv`](overlap_col_desc.csv) is the data dictionary: one row per column with its
description, where the value came from, its closed vocabulary where it has one, how many cells are
blank, and **what a blank means**. That last field is load-bearing rather than decoration. Five
columns use an empty cell to carry information and in each case the wrong reading is the plausible
one: a blank `feature_overlap_jaccard` means the features are not distributed, not that the overlap
is zero, and a blank `training_cohort_note` means nothing was written down, not that the cohort was
healthy. Both files are generated together, and a test asserts the dictionary describes exactly the
columns the table has.

`tissue_class`, `population_class` and `target_class` are coarsenings of the registry's own
vocabulary, with the raw value kept beside each one. `profile_key` groups clocks fitted on the same
kind of data, and `peers_same_profile` names them.

Two senses of overlap are in the file. **Training overlap** is registry metadata and exact.
**Feature overlap** is the literal shared CpGs between two clocks as a Jaccard index, computable
for the 42 clocks that ship coefficients; the rest are blank rather than zero, because *shares no
CpGs* and *we cannot see this clock's CpGs* are different facts. Nearby clocks come out where they
should: `leerobust` and `leerefinedrobust` share 395 probes at J = 0.71.

Every row carries a `verify_url` and every one is checked. 131 resolve directly; the other 44 are
DOIs whose publisher blocks automated requests, confirmed registered against Crossref rather than
reported as broken. 66 clocks are cross-referenced against
[TranslAGE](https://www.translage.io), [biolearn](https://github.com/bio-learn/biolearn) and
[methylCIPHER](https://github.com/HigginsChenLab/methylCIPHER), so the training target can be
checked against someone who is not us.

**One column is deliberately absent.** Health status of the training cohort is not a registry
field and is not recoverable from one: 5 of 175 entries mention it in free text. There is no
"trained on healthy individuals" column, because the honest value for nearly every row would be
*unstated*. `target_class == disease` identifies the five clocks fitted on a disease endpoint,
which is the part that is knowable; for the rest, read the paper behind `verify_url`.

Regenerate with `python python/tools/build_overlap.py --check-urls`.

## Reproducibility

Every run writes a manifest: package and registry versions, the SHA-256 of every coefficient file,
the array manifest that decoded the IDATs, the reliability table behind the intervals, the device
and floating-point precision, the imputation policy, and every warning raised.

→ [Reproducibility](https://bhagesh-h.github.io/FALCONAge/architecture.html) ·
[test/README.md](test/README.md), the benchmark corpus and how to read each output

## Documentation

| Page | What is on it |
|---|---|
| [About](https://bhagesh-h.github.io/FALCONAge/) | What a clock is, what this computes, what it refuses |
| [Getting started](https://bhagesh-h.github.io/FALCONAge/guide/FALCONAge.html) | Docker first, then Python and R; score, interpret, reproduce |
| [Choosing a clock](https://bhagesh-h.github.io/FALCONAge/guide/clocks.html) | All 175 routed by the question they answer |
| [Clock catalogue](https://bhagesh-h.github.io/FALCONAge/clocks.html) | Every clock with its scale, tissue, platform and paper |
| [The science](https://bhagesh-h.github.io/FALCONAge/science.html) | The algorithms, the failure modes, the citations, the genetics under the probe, and how to fit a clock of your own |
| [Architecture](https://bhagesh-h.github.io/FALCONAge/architecture.html) | Which file computes what, and why |
| [Figure gallery](https://bhagesh-h.github.io/FALCONAge/gallery.html) | Every figure type with its interpretation |
| [Python](https://bhagesh-h.github.io/FALCONAge/reference/) · [R](https://bhagesh-h.github.io/FALCONAge/r/reference/) | Full API references |
| [GPU](docs/gpu.md) | Measured, including where CUDA is *slower* |
| [Docker Hub](https://hub.docker.com/r/bhagesh/falconage) | `bhagesh/falconage:1.0.0-cpu` and `:1.0.0-cuda`, built from the Dockerfiles here |
| [Claude skill](.claude/skills/falconage/) | Drive all of the above from Claude; CI-checked against the running package |
| [test/README.md](test/README.md) | Every published number and the command that regenerates it |

The site is also downloadable as a single PDF or a markdown ZIP.

## Citation

[`CITATION.cff`](CITATION.cff) is machine-readable and is what GitHub's *Cite this repository*
button reads. In R, `citation("FALCONAge")`.

**Citing FALCONAge does not cite the clock it computed for you.** Every registry entry carries its
primary reference:

```python
res.registry.get("horvath2013").cite("bibtex")
```

## Contributing

Most valuable first: tracing an untraced clock's coefficients to a primary source.
→ [PUBLISHING.md](PUBLISHING.md) for release process, [CHANGELOG.md](CHANGELOG.md) for history.

## How the documentation was written

Commit messages and parts of the documentation (this README, the vignettes, the roxygen comments
and the pages on the site) were drafted with Anthropic's Claude, mostly Claude Opus 5 through
Claude Code, with earlier sections written using Opus 4.1 and Sonnet 4.5.

The analysis code and its results are the author's, and every number quoted in the documentation
comes from a run rather than from a draft. Prose written this way can still describe the code
incorrectly. If something here does not match what the package does, please open an issue: a
documentation error is a bug and is worth reporting as one.

## Licence

GPL-3 for the code. Clock coefficients keep their own licences. The registry records the licence,
source URL and redistribution status per clock, and FALCONAge prints the restriction at score time.

## See also

[pyaging](https://github.com/rsinghlab/pyaging) · [biolearn](https://bio-learn.github.io/) ·
[ComputAge](https://github.com/ComputationalAgingLab/ComputAge) ·
[methylCIPHER](https://github.com/HigginsChenLab/methylCIPHER) ·
[BioAge](https://github.com/dayoonkwon/BioAge) ·
[sesame](https://bioconductor.org/packages/sesame/)
