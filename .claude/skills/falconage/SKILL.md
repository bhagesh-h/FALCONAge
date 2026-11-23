---
name: falconage
description: Score biological aging clocks from DNA methylation, clinical chemistry or NMR metabolomics data using FALCONAge via Docker. Use when asked to compute epigenetic age, DNAm age, biological age, age acceleration, pace of aging, or a named clock (Horvath, Hannum, PhenoAge, GrimAge, DunedinPACE, ...) from a preprocessed beta matrix, GEO series matrix, clinical chemistry table or Nightingale NMR table; to build, add or combine (stack) a clock on one's own cohort; to run FALCONAge inside a Nextflow or Snakemake pipeline; or to answer which clock answers a given question and how far a clock score can be trusted.
---

# FALCONAge

Aging-clock scoring with the unit, the provenance, and the measurement error attached
to every number. 68 human clocks, every one scoring offline, plus any clock you add.

**Docker is the supported path.** Nothing else needs installing, and the image pins every
dependency so the same input gives the same numbers on any machine.

## First, orient

Before running anything, establish three things, because each one changes the answer:

1. **What question is being asked?** "How old does this look" and "who is at risk of dying
   sooner" are different questions with different clocks. See `reference/clocks.md`.
2. **What specimen is it?** A whole-blood clock on saliva was measured at 3.83–16.46 years of
   error in the same 91 people. FALCONAge refuses or warns, but only if `obs["tissue"]` is set.
3. **What platform?** Coverage decides which clocks can run at all. A 27K array supports about
   four of them.

If any of the three is unknown, ask rather than guess. Guessing is the failure mode this
package exists to prevent.

## Setup, once

```bash
docker pull bhagesh/falconage:1.0.0-cpu
```

Or build the same image from a clone:

```bash
git clone https://github.com/bhagesh-h/FALCONAge.git && cd FALCONAge
docker build -f docker/Dockerfile.cpu -t bhagesh/falconage:1.0.0-cpu .
```

On Windows PowerShell write `"${PWD}"` wherever `"$PWD"` appears below; on `cmd.exe`, `"%cd%"`.

## The one-command path

Scores, uncertainty, age acceleration, the figures and an HTML report, no code written. Each
figure is `figures/<title>.png` (clock appended for per-clock figures) with no title or caption
on the canvas, and a `<title>.txt` beside it holding the title, the figure's numbers and the
description for a figure legend. The
entrypoint is the CLI, so the verb goes straight after the image name with no `falconage` in
front of it:

```bash
docker run --rm -v "$PWD:/work" bhagesh/falconage:1.0.0-cpu run betas.csv --samples samples.csv
```

The sample sheet is optional (first column the sample id, then any of `age`, `sex`, `tissue`,
`group`); without `age` there is no acceleration and without `tissue` no specimen check. To show
someone the outputs before their data is ready, `demo` runs the same pipeline on synthetic data.

## The scripted path

```bash
docker run --rm -it -v "$PWD:/work" bhagesh/falconage:1.0.0-cpu python
```

```python
import falconage as fa

d = fa.read_betas("betas.csv")          # preprocessed, QC-passed betas
d.obs["tissue"] = "whole blood"         # do not skip this
d.obs["age"] = ages                     # needed for acceleration
d = fa.prepare(d)

res = fa.score(d, clocks="compatible")

res.interpretation()      # scale, unit, legal operations, reliability, caveats
res.coverage              # what ran, what was skipped, and why
fa.technical_se(res, d)   # how much of each score is the assay
```

`clocks="compatible"` scores what the data supports and reports the rest as skipped with a
reason. Naming clocks explicitly is stricter: every one must work, because an explicit request
should never be dropped quietly.

### When a clock score is not the right answer

Three readouts return no age and are gated by no `scale_type`. Reach for them when the question
is about variance rather than about level:

```python
fa.entropy(d)             # per sample: how far the methylome is from committed
fa.drift(d)               # per sample: distance from the cohort centroid, leave-one-out
fa.variable_sites(d)      # per site: which cytosines widen with age
fa.noise_barometer(d)     # per GROUP: the Mei barometer over those sites

fa.repertoire_diversity(clones)   # clone structure, to pass to acceleration(adjust=[...])
fa.coefficient_mass(annotation)   # where a clock's weight sits, given a list you supply
fa.variance_components(res, subject_col="donor", occasion_col="visit")
```

`noise_barometer` is per group, not per sample, and asking it for a per-person number is the
mistake to avoid: a standard deviation needs more than one observation. Use `drift` for that.

`variance_components` returns both `icc` and `icc_age_adjusted`. Quote the adjusted one for any
individual-level claim: a raw ICC on a cohort spanning decades mostly reports that the clock
tracks age, which was never in question. With one draw per visit and no technical replicates,
`var_state` holds day-to-day and assay variance together and `var_tech` is NaN; `design` says so.

R is the same verbs in the same image: `docker run --rm -it -v "$PWD:/work"
bhagesh/falconage:1.0.0-cpu R`, then `prepare()`, `score()`, `interpretation()`. Both languages call one
numerical core, so the results are bit-identical rather than approximately equal.

## Your own clocks

```python
cmp = fa.compare_learners(d, target="age", groups="person")   # several learners, out of fold
cmp.selected          # the nested error of "pick the best": the number to report
fit = fa.fit_clock(d, target="age", learner="pc_elasticnet", groups="person")
fit.to_clock("mycohortclock", "my_clocks/")                  # yaml + csv
fa.add_clock("my_clocks/mycohortclock.yaml")                 # score it like a catalogue clock
st = fa.stack_clocks(res, target="age", method="nnls")       # combine clocks, out of fold
st.comparison         # against each clock alone and the plain mean
```

`groups=` keeps a person's repeated visits on one side of every split. Quote `cmp.selected`,
never the best row of `cmp.table`. A plain mean across scale types is refused; never fit a stack
on the samples you then compare between groups. On the command line, `run --clocks-dir
my_clocks` scores user clocks alongside the catalogue.

## In a pipeline (Nextflow, Snakemake)

```bash
falconage run betas.csv --samples samples.csv --outdir results --format tsv \
  --threads ${task.cpus} --log-file results.log
falconage versions --process "${task.process}" > versions.yml
```

Exit status: 0 success, 1 internal error, 2 usage error, 3 input data error, 4 scoring or analysis
error (5 only from `consensus --fail-unsupported`). Errors print one line to stderr;
`--log-level debug` adds the traceback. The image passes `/bin/bash .command.sh` straight through,
has `ps`, and runs as any user id. Everything runs on the CPU.

## Rules to hold to

- **Never report a score without its scale.** `res.interpretation()` gives it. An age in years, a
  pace ratio, a cell-type proportion and a log-hazard are not interchangeable, and the package
  will refuse operations that mix them.
- **Never compute `predicted − chronological` by hand.** Use `fa.acceleration()`, which knows
  which clocks that is defined for and refuses the rest.
- **Never present a point estimate alone** for a decision about one person. Attach
  `technical_se()` (assay noise) or `conformal_interval()` (distance from the truth). They answer
  different questions and the second is much larger.
- **A refusal is the answer.** When FALCONAge declines to score something, report the reason
  rather than working around it with `min_coverage=0` or by dropping the tissue column.
- **Coverage is not validity.** High feature coverage says the probes are present. It says
  nothing about whether coefficients fitted in one tissue, species or population transfer.
- **Entropy, drift and the barometer do not travel between datasets.** They move with
  normalisation, probe panel and detection-p filtering, so the only defensible comparison is
  within one processed matrix. Never quote an entropy from one paper against one you computed.

## Reference

- `reference/docker.md`, the full command set: accepted inputs, GEO accessions,
  benchmarking, and the native pip/R install for anyone who wants it.
- `reference/clocks.md`, the clock families, which one answers which question,
  the nine scale types, and the traps.
- `reference/interpreting.md`, reading the outputs, every refusal and the measurement behind
  it, and what the package deliberately does not do.

Full documentation: <https://bhagesh-h.github.io/FALCONAge/>
