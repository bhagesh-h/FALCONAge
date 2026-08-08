# FALCONAge test corpus

Public data for exercising every algorithm in the [clock catalogue](../../README.md#5-clock-catalogue).
**586 MB across 33 files**, against a 1 GB ceiling the fetcher refuses to cross.

Nothing here is committed. [datasets.yaml](datasets.yaml) holds the URLs, the expected sizes and
the publishers' checksums; the two fetchers turn that into files on disk; [.gitignore](.gitignore)
keeps the files out of the history. The corpus is reproducible from the manifest, not stored
beside it.

## Quick start

```bash
# from the repository root
docker build -f docker/Dockerfile.testdata -t falconage-testdata:1.0.0 .

docker run --rm -v "$PWD/test/data:/data" falconage-testdata:1.0.0 plan     # what it costs
docker run --rm -v "$PWD/test/data:/data" falconage-testdata:1.0.0 python   # fetch it
docker run --rm -v "$PWD/test/data:/data" falconage-testdata:1.0.0 verify   # check it
```

On Windows PowerShell use `${PWD}` instead of `$PWD`. Through compose the mount is set for you:

```bash
docker compose -f docker/docker-compose.yml run --rm testdata plan
```

Without Docker, either script runs standalone — Python needs `PyYAML`, R needs
`yaml`, `jsonlite`, `curl`, `digest`:

```bash
python test/data/fetch_test_data.py --dry-run
Rscript test/data/fetch_test_data.R  --groups bench,clinical
```

## Two implementations, one output

`fetch_test_data.py` and `fetch_test_data.R` are the same program in two languages, reading the
same manifest, and they write a **byte-identical** `checksums.sha256` and `provenance.json`.

That is not decoration. FALCONAge proper wraps one Python numerical core from R, so the two
languages cannot disagree about a clock score. The corpus is the one artefact that sits *outside*
that guarantee — it is what the package is tested against, so a fetcher that depended on the
package would be circular. Two independent implementations that agree byte for byte is the
substitute, and `docker build` asserts it:

```dockerfile
RUN python3 /opt/testdata/fetch_test_data.py --self-test \
 && Rscript   /opt/testdata/fetch_test_data.R  --self-test
```

`fetch-test-data both` runs the pair end to end: fetch with Python, verify with R.

## What is in it, and what it is for

| Group | Files | Size | Samples | Platform | Exercises |
|---|---:|---:|---:|---|---|
| [`bench`](#bench) | 11 | 368.4 MB | 261 | 27K, 450K, EPICv1 | the AA1/AA2 benchmark and most of the 161 clocks |
| [`idat`](#idat) | 8 | 60.4 MB | 4 | EPICv1, EPICv2 | the raw IDAT → noob → BMIQ path |
| [`gestational`](#gestational) | 1 | 50.8 MB | 22 | 450K | the 7 gestational and paediatric clocks |
| [`mouse`](#mouse) | 5 | 50.2 MB | 4 | RRBS | Petkovich, Meer, Stubbs, Thompson |
| [`epicv2`](#epicv2) | 1 | 41.2 MB | 8 | EPICv2 | probe-suffix aggregation |
| [`mammalian`](#mammalian) | 4 | 9.3 MB | 29 | MammalMethylChip40 | MammalianLifespan, MammalianFemale, EnsembleAge ×3 |
| [`clinical`](#clinical) | 3 | 5.4 MB | tens of thousands | blood chemistry | PhenoAge, KDM, homeostatic dysregulation |
| **total** | **33** | **585.7 MB** | | | |

Fetch a subset with `--groups bench,clinical`. Every group is on by default.

### bench

The ten smallest studies in the published 65-study ComputAgeBench set, plus its sample table.
Eight conditions with matched controls, two case-only cohorts, three array generations.

| Study | n | Platform | Condition | Ages |
|---|---:|---|---|---|
| GSE182991 | 27 | EPICv1 | Hutchinson-Gilford progeria vs HC | 0–41 |
| GSE130030 | 28 | 450K | multiple sclerosis vs HC | 20–65 |
| GSE71841 | 24 | 450K | rheumatoid arthritis vs HC | 23–57 |
| GSE118468 | 21 | 450K | COPD vs HC (15 vs 6) | 49–82 |
| GSE214297 | 16 | EPICv1 | congenital generalised lipodystrophy vs HC | 1–48 |
| GSE151355 | 20 | 450K | Parkinson's, **no controls** | 74–91 |
| GSE107143 | 16 | 450K | ankylosing spondylitis vs HC | 32–88 |
| GSE56606 | 90 | 27K | type 1 diabetes vs HC | 16–68 |
| GSE49909 | 49 | 27K | obesity vs HC | 29–85 |
| GSE62867 | 6 | 27K | ischaemic heart disease, **no controls** | 48–67 |

Chosen against three constraints at once. **Progeria** is the hard case, not the easy one:
running the corpus shows every clock in the registry failing to separate HGPS from age-matched
healthy, which is the published result rather than a bug — Horvath's own work found progeroid
syndromes have near-normal DNAm age. It is in the corpus because a benchmark made only of
conditions the clocks already detect measures nothing. **The two case-only sets** exist because
AA1 and AA2 are different tests and a corpus with only matched designs never executes the AA1
path. **COPD at 15 versus 6** is the unbalanced design where a benchmark that ignores group sizes
starts reporting significance it has not earned.

What the corpus actually produces, running all ten studies through the 20 tier A clocks: 13
significant comparisons out of 102, concentrated in ankylosing spondylitis, with DNAmPhenoAge
leading on AA2 and every clock scoring zero on progeria.

The 27K sets are the cheapest files here and do the least glamorous job: most modern clocks lose
most of their features on a 27,578-probe array, and the per-clock coverage report has to say so
rather than impute its way to a confident wrong answer.

`computage_bench_meta.tsv` carries `SampleID, DatasetID, PlatformID, Tissue, CellType, Gender,
Age, Condition, Class` for all 65 studies, so condition labels never have to be scraped out of
GEO characteristics strings.

> **float32.** These parquets store betas in single precision. Fine for benchmarking; wrong for
> the 1e-6 FP64 gold-standard vectors in the scoring specification (§10.1).
> Those need a source that carries full double precision, and this is not one.

### idat

Two EPIC v1 samples from GSE182991 and two EPIC v2 samples from GSE330325, Grn and Red channels
each. Every other methylation file in the corpus is somebody else's beta matrix, normalised by a
pipeline whose choices are not recorded. This is the only path FALCONAge controls end to end, and
it is the one that decides whether two labs' numbers are comparable.

Two samples per platform is enough for correctness and not enough for a normalisation that
estimates parameters across a plate — funnorm on n=2 is not funnorm. The full `RAW.tar` for either
series is several hundred megabytes and does not fit the budget.

### gestational

GSE66459: 11 preterm and 11 term umbilical cord blood samples on 450K. A series matrix, so the
characteristics and the betas arrive in one file — which also makes it the fixture for the
series-matrix reader, the path roughly 60% of GEO methylation series need because they publish no
IDATs.

> **Days, not weeks.** Gestational age is recorded here in days (185–280); all seven gestational
> clocks predict weeks. That silent factor of seven is exactly what
> §2.6 refuses to guess at.

### mouse

Four whole-blood RRBS samples from GSE80672 (Petkovich), two at 0.67 and two at 35, plus the
12.7 kB series matrix covering all 255 samples in the study.

The mouse clocks are not array clocks: they key on genomic coordinate, not `cg` identifier, and
no array file here can stand in. This is also the corpus's only sequencing input, so it is the
only thing that exercises coverage-weighted beta estimation — a site read four times and one read
four hundred times are not the same measurement.

> **Months, labelled years.** GEO records the age field as `age (years)` and the values run 0.67
> to 35. They are months. A mouse does not live 35 years, and a clock handed 35 as a year value
> reports a 33-year acceleration and looks precise doing it. Best argument in the corpus for
> declaring units instead of inferring them, and it is a real published series rather than a
> contrived example.

GSM2132960 is titled `M3503R` against `M3503` above it, with identical age, sex, strain, diet and
genotype — a re-run of the same animal. GEO does not say so outright, so treat it as a
near-duplicate rather than a certified technical replicate. It is still the only thing here that
can put a number on a clock's own measurement noise instead of quoting the published ICC.

### epicv2

GSE330325, eight samples. EPIC v2 renamed its probes: `cg00000029` became `cg00000029_TC21`, and
a clock matching on exact identifiers finds zero of its features and returns a confident number
computed entirely from imputed values. Suffix aggregation is mandatory, not an optimisation, and
the replicate structure — some probes appear two or three times with different suffixes — cannot
be faked convincingly. This file also carries the metadata for the two EPIC v2 IDAT pairs above.

### mammalian

GSE184222 (wild ass and Grevy's zebra, 12 samples, ages 2.4–18.5 y) and GSE184224 (human cells,
17 samples), each as a normalised beta CSV plus a 2 kB series matrix. Human 450K data cannot
exercise the mammalian clocks: the probe namespace is different and the species lookup has
nothing to look up. GSE184224 is the cross-check that separates a species effect from an array
effect — same probes, same pipeline, a species the human clocks also cover.

> **Three species, not 348.** The Mammalian Methylation Consortium's full matrix (GSE223748) is a
> single 4.2 GB file — four times this entire corpus for one group. It is out of budget, and named
> here rather than quietly omitted.

### clinical

NHANES III and NHANES IV as extracted by the [BioAge](https://github.com/dayoonkwon/BioAge)
package, plus the healthy-young reference subset. PhenoAge, Klemera–Doubal and homeostatic
dysregulation take blood chemistry, were fitted on NHANES III and validated on NHANES IV, and
nothing in the methylation groups can exercise them.

`NHANES3_HDTrain` is not optional. Homeostatic dysregulation is a Mahalanobis distance *to a
reference population*; substituting the sample's own distribution silently changes what is being
measured.

## What this corpus cannot test

Stated here rather than discovered later.

- **The 28 scaffold-only clocks.** GrimAge, GrimAge2 and its ten sub-clocks, PCGrimAge, the twelve
  SystemsAge variants and DunedinPACE need coefficients FALCONAge does not distribute
  ([README §6](../../README.md#6-clocks-that-need-author-permission)). The data here exercises
  their feature alignment and tensor shapes; it cannot produce a number from them. Register a
  licensed coefficient file and the same corpus scores them.
- **FP64 gold-standard vectors.** The bench parquets are float32. Double-precision reference
  values come from the papers' own worked examples, not from here.
- **Plate-level normalisation.** Four IDATs is not a plate.
- **Transcriptomic and proteomic clocks.** Out of scope for v1.0; groups get added when v1.1 and
  v1.2 land.
- **345 of 348 mammalian species.** See above.

## Checksums, and what they are worth

| Source | Files | Digest | On mismatch |
|---|---:|---|---|
| Hugging Face (ComputAgeBench) | 10 | published as the LFS object SHA-256 | **hard error**, file deleted |
| GEO | 20 | none published | size warning, digest recorded on first fetch |
| GitHub raw (BioAge) | 3 | none published | size warning, digest recorded on first fetch |

Twenty-three of the thirty-three files are trust-on-first-use, and that is stated rather than
dressed up: GEO publishes sizes but no digests, and a submitter can replace a supplementary file
in place without the name changing. What the fetcher guarantees is that the bytes have not
changed *since you fetched them*, which is a weaker claim than verification and a real one.

Sizes were read from the live listings on 2026-08-07. A size mismatch is a warning, not a
failure — the honest response to a re-uploaded file is "the manifest is stale", not a refusal to
work.

## Files the fetcher writes

```
test/data/
├── bench/  clinical/  epicv2/  gestational/  idat/  mammalian/  mouse/
├── checksums.sha256     # sha256sum format, sorted; `sha256sum -c` works
└── provenance.json      # URL, source, licence and both digests per file
```

Neither carries a timestamp. A provenance file that changes on every run cannot be diffed against
the last one, which is the only thing anybody ever wants to do with it.

## Licences

The corpus is fetched, never redistributed. Terms belong to the publishers:

- **ComputAgeBench** — CC-BY-SA-4.0. Cite Kriukov et al. 2024, bioRxiv 2024.06.06.597715.
- **GEO** — public; individual submitters' and source publications' terms apply. Several of these
  series are human subject data reused under the consent the original study obtained.
- **BioAge** — GPL-3.0 for the extraction and harmonisation. NHANES itself is US federal public
  domain. Cite Kwon & Belsky 2021, GeroScience 43:2795–2808.

## Adding a dataset

1. Add an entry to `datasets.yaml` with its real `bytes:` (an HTTP `HEAD` gives it) and a
   `sha256:` if the publisher provides one.
2. Update the group's `bytes:`, and `expected_total_bytes` / `expected_total_files` at the top.
3. Run `--self-test`. It checks the arithmetic, rejects duplicate destinations, non-HTTPS URLs,
   short digests and paths that escape the output directory. Both languages must pass.
4. Say in the entry's `note:` which clocks it exercises that nothing else does. A dataset that
   only overlaps what is already here costs budget and buys nothing.
