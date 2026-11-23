# Running FALCONAge

Every command runs from the repository root. On Windows PowerShell write `"${PWD}"` in place of
`"$PWD"`; on `cmd.exe`, `"%cd%"`. A path error on Windows is almost always this and nothing else.

## Get the image

```bash
docker pull bhagesh/falconage:1.0.0-cpu
```

Or build the same image from a clone:

```bash
git clone https://github.com/bhagesh-h/FALCONAge.git && cd FALCONAge
docker build -f docker/Dockerfile.cpu -t bhagesh/falconage:1.0.0-cpu .
```

One image carries Python, R, the CLI and every bundled weight file. `bhagesh/falconage:1.0.0-py`
is a smaller Python-only image (no R, no parquet or `.h5ad` reader). Check either with:

```bash
docker run --rm bhagesh/falconage:1.0.0-cpu config
```

## The two commands most people need

```bash
# the whole pipeline on synthetic data, to see every output: about a minute
docker run --rm -v "$PWD:/work" bhagesh/falconage:1.0.0-cpu demo

# the whole pipeline on real data: scores, uncertainty, acceleration, figures, report.html
docker run --rm -v "$PWD:/work" bhagesh/falconage:1.0.0-cpu run betas.csv --samples samples.csv
```

`run` takes a beta matrix (`.csv`, `.tsv`, `.parquet`, `.h5ad`), a GEO series matrix, a folder of
IDATs, a clinical table (`--kind clinical`) or a Nightingale `.xlsx`. The sample sheet is optional:
first column the sample id, then any of `age`, `sex`, `tissue`, `group`. Results land in
`<input>_falconage/`; per-clock notes go to `warnings.csv` rather than the console.

## Three ways in

The entrypoint **is** the `falconage` CLI, so a verb goes straight after the image name. Writing
`... bhagesh/falconage:1.0.0-cpu falconage score ...` runs `falconage falconage score` and fails.
`python`, `R`, `Rscript`, `sh` and `bash` are the exceptions, passed through as given.

```bash
# a session
docker run --rm -it -v "$PWD:/work" bhagesh/falconage:1.0.0-cpu python
docker run --rm -it -v "$PWD:/work" bhagesh/falconage:1.0.0-cpu R

# a single CLI command: the verb, with no `falconage` in front of it
docker run --rm -v "$PWD:/work" bhagesh/falconage:1.0.0-cpu report betas.csv --outdir results/

# a script in the working tree
docker run --rm -v "$PWD:/work" bhagesh/falconage:1.0.0-cpu python my_analysis.py
```

The mount is what makes this work: `-v "$PWD:/work"` puts the current directory inside the
container at `/work`, so inputs are read from and outputs written to the real filesystem. Anything
written outside `/work` disappears when the container exits.

## CLI verbs

Thirteen verbs, and `--help` on any of them is authoritative. Every verb also takes
`--threads N`, `--log-level {quiet,info,debug}` and `--log-file PATH`:

```bash
falconage demo                         # everything, on synthetic data
falconage run betas.csv --samples samples.csv   # everything, on your data
falconage versions                     # versions as YAML (a pipeline's versions.yml)
falconage config                       # versions, registry size; run this first when confused
falconage clocks list --tier bundled   # browse the registry; list, info and cite are the actions
falconage clocks list --search mortality
falconage clocks info horvath2013
falconage download GSE40279 --dry-run  # what it would fetch, and how many bytes, before fetching
falconage download GSE40279 --want both
falconage cache ls                     # what has been downloaded; `cache rm` clears it
falconage preprocess --input idats/ --output data.h5ad    # raw or public data to a scoreable file
falconage score --input data.h5ad --outdir results/ --clocks compatible
falconage bench --input results/       # the AA1/AA2 benchmark
falconage power --clock horvath2013 --effect 0.5 --sd 6   # how many samples you need
falconage consensus results/ --group-col condition        # does a group difference survive across clocks
falconage report betas.csv --outdir results/              # read, QC, score, quantify, write HTML
falconage report betas.csv --outdir results/ --quarto     # and falconage_report.qmd, every output under its step
```

Which arguments are positional and which are flags is not guessable, and it
differs by verb: `report` and `consensus` take their input positionally,
`score`, `bench`, `preprocess` and `power` take named flags, and `clocks`
requires an action before any flag. `--help` on a verb is authoritative, and
`docs/check_api_docs.py` runs every command written above through the real
parser so this block cannot drift from it.

`--quarto` writes `falconage_report.qmd`; Quarto is not in the image, so render it with the
Quarto container from the output directory:
`docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD:/w" -w /w ghcr.io/quarto-dev/quarto quarto render falconage_report.qmd`.

`--dry-run` on `download` transfers nothing and prints every URL and byte count. Use it before
committing to a GEO series; some are several hundred megabytes.

## Inputs it accepts

| Input | Reader |
|---|---|
| Raw Illumina IDATs (27K, 450K, EPIC v1, EPIC v2) | `fa.read_idat_dir("idats/")` |
| Beta matrix, CSV or parquet | `fa.read_betas("betas.csv")` |
| GEO series matrix | `fa.read_series_matrix("GSE*_series_matrix.txt.gz")` |
| Nanopore bedMethyl | `fa.read_bedmethyl(...)`, `fa.read_bedmethyl_dir(...)` |
| RRBS coverage files | `fa.read_rrbs_dir(...)` |
| Targeted panel | `fa.read_panel(...)` |
| ComputAgeBench study | `fa.read_computage_bench("GSE107143")` |
| Olink NPX | `fa.preprocess.read_olink(...)` |
| SomaScan RFU | `fa.preprocess.read_somascan(...)` |
| Bulk RNA-seq counts | `fa.preprocess.read_counts(...)` |
| Clinical chemistry | `fa.read_clinical(..., units=...)`, `units` has no default, on purpose |

The proteomic and transcriptomic readers live under `fa.preprocess` rather than at top level,
which reflects their status: the preparation chains ship and are tested, but no clock in the
catalogue declares either data type, so scoring one refuses with "no clocks to score".

## From raw IDATs

The only path FALCONAge controls end to end. Everything else is somebody else's beta matrix,
normalised by choices that were not recorded.

```python
d   = fa.read_idat_dir("idats/")        # decode, map addresses, infer the platform
res = fa.score(fa.prepare(d), clocks="compatible")
```

The chain runs detection (pOOBAH) before background correction (noob), because pOOBAH's null *is*
the uncorrected out-of-band signal. Undetected probes become `NA` rather than a number. The array
manifest is fetched from Illumina's public bucket on first use and cached, the one step that
needs a network.

Validated against the published betas for the *same physical samples*: r = 0.99928,
median |Δ| = 0.011, 99.7% of probes within 0.05.

A published probe mask is a separate, deliberate step rather than a default, because what an EWAS
should drop and what a clock should drop are different questions:

```python
d = fa.preprocess.apply_mask(d, kind="general")   # platform inferred from the data
fa.preprocess.mask_report(d)                      # what it would remove, before removing it
```

Dye-bias correction ships **switched off**. On real IDATs the implementation moves the median beta
by +0.10 to +0.12, because a correct version needs normalisation control probes whose addresses
are not in the fetchable manifest. Turn it on only if you know why you want it.

## Adding a plate later

Standard ComBat re-estimates from every sample at once, so adding a batch silently moves scores
you already reported, measured at up to 2.20 years. Freeze the reference instead:

```python
ref = fa.fit_batch_reference(d, batch_col="plate")   # save this alongside the results
d2  = fa.apply_batch_reference(d2, ref)              # earlier plates stay bit-identical
```

It freezes ComBat's global parameters *and* its empirical-Bayes hyperparameters, which is the part
a naive "just refit on the reference" gets wrong.

## Clocks of your own

```python
fit = fa.fit_clock(d, target="age", learner="elasticnet", groups="person")
fit.to_clock("mycohortclock", "my_clocks/")
fa.add_clock("my_clocks/mycohortclock.yaml")
fa.score(d, clocks=["mycohortclock", "horvath2013"])
```

On the command line, `run --clocks-dir my_clocks` scores them alongside the catalogue, and the
coverage map gives each one its own track.

## Native install, if Docker is not available

```bash
pip install "falconage @ git+https://github.com/bhagesh-h/FALCONAge.git#subdirectory=python"
```

```r
remotes::install_github("bhagesh-h/FALCONAge", subdir = "r")
FALCONAge::falconage_install()   # builds the Python environment R calls into
```

Every code example works unchanged; only the `docker run` prefix goes away. The R package
delegates all arithmetic to the Python core, so it needs a Python environment either way.

## Reproducibility

Every run writes `run_manifest.json`: package and registry versions, the SHA-256 of every
coefficient file, the array manifest that decoded the IDATs, the reliability table behind the
intervals, the floating-point precision, the imputation policy, and every warning raised.
Two runs reporting the same score either used the same coefficients or the manifest says they
did not. Keep it with the results.

```bash
falconage score --input d.h5ad --outdir r/
```

```bash
falconage score --input d.h5ad --outdir r/
```

```bash
falconage score --input d.h5ad --outdir r/
```
