# Reading what comes back

## The result object

```python
res.scores            # samples x clocks
res.long()            # one row per sample per clock, carrying the scale
res.coverage          # per clock: feature coverage, mass coverage, skip reason
res.interpretation()  # scale, unit, legal operations, reliability, caveats
res.manifest          # versions, device, SHA-256 of every coefficient file, every warning
```

Read `coverage` before `scores`. A clock absent from the table was refused, and the reason is
more informative than any number it would have produced.

## Two uncertainties, and they are not the same

```python
fa.technical_se(res, d)        # would a repeat of this DNA give a different number?
fa.conformal_interval(res)     # how far is this number from the truth?
```

**Technical SE** propagates each probe's published test-retest reliability through the clock's own
weights: `Var = f'(raw)² · Σ wⱼ² sⱼ² (1 − ICCⱼ)`. Per clock and per sample, because a sample with
more imputed features earns a wider interval. On the reference corpus Horvath 2013 comes out at
±1.58 years; DunedinPoAm38 is least repeatable; the 319,607-probe BLUP clock is the most, because
a model spread over the whole array averages the noise away.

That sum treats probe errors as independent, and they are not: chip position and plate move many
probes together. **The reported SE is therefore a lower bound.** Where a published clock-level ICC
exists, `source="clock"` gives a figure that does contain the correlated part; if the two disagree
by a lot, believe the clock-level one.

**Conformal intervals** answer the other question and the answer is much larger. Split conformal
on healthy blood samples with known ages: the half-width is an order statistic of the absolute
residuals, so on any sample exchangeable with the calibration cohort the interval contains the
truth at the stated rate, with no distributional assumption. Horvath 2013's 90% half-width is
about 12 years. That is the honest width of an individual age prediction, and it is why these
are population-research instruments.

The guarantee is conditional on exchangeability. The calibration cohort is public blood data,
adult, and overwhelmingly of European ancestry. On a paediatric or non-European cohort the
guarantee does not transfer, and the package says so rather than widening by an arbitrary factor.

## Before you commit to a design

```python
fa.power("horvath2013", effect=0.5, result=pilot)
```

Reliability is part of the answer: at ICC 0.9 a tenth of the sample size exists purely to average
out the instrument. It refuses to default the standard deviation, because n scales with its
square, so a guessed SD is a guessed answer printed to three significant figures.

## Every refusal, and the measurement behind it

| Refused | Why, measured |
|---|---|
| Age acceleration on a pace-of-aging clock | A pace is already a rate. Subtracting age from it is a units error. |
| A whole-blood clock on saliva | Saliva clock ages ran 3.83–16.46 years above buffy coat in the same 91 people, while still correlating with them at Spearman 0.45–0.69. Correlation is not agreement. |
| Any array clock on cell-free DNA | Not a tissue, but a fragment population shed from many. Array clocks applied directly perform poorly. |
| A placenta, cord-blood, buccal or brain-cortex clock on blood | A category error, not an offset. Twelve clocks refuse outright; the rest warn with the measured discordance. |
| A cohort-centred clock given one sample | Centring one row against itself zeroes every feature; the model returns its intercept for anybody. |
| `predicted − chronological` on DamAge or AdaptAge | Slope against age is 0.967, but the offset swings 162 years between cohorts against Horvath's 15. |
| A clock below the coverage **or** the coefficient-mass floor | 96% of probes present can be 61% of the model. |
| A `.pt` file of neural weights | `torch.load` executes arbitrary code while unpickling. Safetensors only. |
| An interval on a clock with no established reliability | It would be invented rather than measured. `None` means "not established", never "fine". |

When a refusal appears, report the reason. Working around it with `min_coverage=0`, by dropping the
tissue column, or by computing the difference by hand produces a number that looks identical to a
valid one and is not.

## Outputs that are not scores

Four readouts return no age and carry no `scale_type`, so none of the acceleration machinery
applies to them and none is comparable across datasets processed differently.

| Call | Unit | Per what | Read it as |
|---|---|---|---|
| `fa.entropy(d)` | 0 to 1 | sample | 0 = every site committed, 1 = every site at beta 0.5. Report `n_sites` alongside: it is a mean over sites, so a mean over *different* sites is not a comparison. |
| `fa.drift(d)` | beta units | sample | Distance from the cohort centroid, leave-one-out. The one to regress against an outcome. |
| `fa.noise_barometer(d)` | summed SD | **group** | Mei's statistic. Needs a cohort in the hundreds; on 27 samples nothing survives FDR and the function says so rather than returning a number. |
| `fa.variable_sites(d)` | test result | site | `rising` is the conjunction of significance *and* an upward trend. A site whose variance falls is equally significant and must not be counted. |

`fa.repertoire_diversity(clones)` returns clone-structure metrics to join onto `obs` and pass to
`acceleration(adjust=[...])`. Prefer `clonality`, `simpson` and `effective_clones` over `richness`
and `shannon`, which both rise with sequencing depth; pass `rarefy="min"` when depths differ.

`fa.variance_components(res, subject_col=..., occasion_col=...)` returns `icc` and
`icc_age_adjusted`. **Quote the adjusted one for anything about an individual.** A raw ICC on a
cohort spanning decades mostly reports that the clock tracks age.

## What FALCONAge does not do

| Not implemented | Why |
|---|---|
| Cross-platform liftover | Mapping tables encode concordance measured on paired samples; not derivable from array manifests. |
| Dye-bias correction on by default | Ships opt-in. Moves the median beta by +0.10 to +0.12 on real IDATs, because the control probes are not in the fetchable manifest. |
| Proteomic or transcriptomic **clocks** | Readers and preparation chains ship. No catalogue entry, because every published model in both families is licence-restricted. |
| A foundation-model imputation backend | `NeuralClock` ships, safetensors only. CpGPT and MethylGPT as zero-shot probe imputation do not. |
| Coefficients for 89 untraced clocks | A per-clock literature hunt; some have no public supplement. |
| Silent correction of anything | Batch reference, platform offset and interval are all *reported*. A score adjusted by an untraceable factor destroys the provenance that is the reason to use this package. |
| Percentiles against an unnamed population | The same person reads as accelerated or decelerated depending on the comparator. |
| Diagnostic thresholds | None exist in the literature. Nothing is coloured red above a cut-off. |

## Citing

Citing FALCONAge does not cite the clock it computed. Every registry entry carries its primary
reference:

```python
res.registry.get("horvath2013").cite("bibtex")
```

Coefficients keep their own licences. The registry records the licence, source URL and
redistribution status per clock, and the package prints the restriction at score time.
