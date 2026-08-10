---
name: falconage
description: Score biological aging clocks from DNA methylation, clinical chemistry, proteomic or transcriptomic data using FALCONAge via Docker. Use when asked to compute epigenetic age, DNAm age, biological age, age acceleration, pace of aging, or a named clock (Horvath, Hannum, PhenoAge, GrimAge, DunedinPACE, ...); to process raw Illumina IDATs; or to answer questions about what an aging clock measures, which clock answers a given question, and how far a clock score can be trusted.
---

# FALCONAge

Aging-clock scoring with the unit, the provenance, and the measurement error attached
to every number. 161 clocks catalogued; 23 score offline.

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
git clone https://github.com/bhagesh-h/FALCONAge.git && cd FALCONAge
docker build -f docker/Dockerfile.cpu -t falconage:1.1.0-cpu .
```

On Windows PowerShell write `"${PWD}"` wherever `"$PWD"` appears below; on `cmd.exe`, `"%cd%"`.

## The one-command path

An HTML report with 31 figures, no code written:

```bash
docker run --rm -v "$PWD:/work" -w /work falconage:1.1.0-cpu \
  falconage report betas.csv --outdir results/
```

## The scripted path

```bash
docker run --rm -it -v "$PWD:/work" -w /work falconage:1.1.0-cpu python
```

```python
import falconage as fa

d = fa.read_betas("betas.csv")          # or fa.read_idat_dir("idats/")
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

R is the same verbs in the same image: `docker run --rm -it -v "$PWD:/work" -w /work
falconage:1.1.0-cpu R`, then `prepare()`, `score()`, `interpretation()`. Both languages call one
numerical core, so the results are bit-identical rather than approximately equal.

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

## Reference

- `reference/docker.md`, the full command set: IDATs, GEO accessions, batch correction,
  benchmarking, GPU, and the native pip/R install for anyone who wants it.
- `reference/clocks.md`, what a clock is, the six families, which one answers which question,
  the nine scale types, and the traps.
- `reference/interpreting.md`, reading the outputs, every refusal and the measurement behind
  it, and what the package deliberately does not do.

Full documentation: <https://bhagesh-h.github.io/FALCONAge/>
