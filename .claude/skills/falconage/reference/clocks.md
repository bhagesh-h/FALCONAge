# Aging clocks: what they are, and which one answers what

## What a clock is

A supervised model mapping molecular features to a scalar. What the scalar *means* is set by the
outcome it was regressed on, not by the units it is reported in. Six families, trained on six
different targets, all commonly reported as "biological age":

| Family | Trained on | Examples | What the number is |
|---|---|---|---|
| **First-generation** | calendar age | Horvath 2013, Hannum, SkinAndBlood | an estimate of age, with prediction error against a known truth |
| **Second-generation** | a survival-weighted composite, then rescaled to years | PhenoAge, GrimAge | years as a unit; age was never the target |
| **Pace-of-aging** | the *rate of change* in organ-system biomarkers, tracked longitudinally | DunedinPACE, DunedinPoAm | a rate: biological ageing per calendar year |
| **Mitotic** | cumulative stem-cell divisions | epiTOC2, StemTOC, HypoClock | a count, not elapsed time |
| **Causality-enriched** | CpGs with Mendelian-randomisation support | CausAge, DamAge, AdaptAge | damaging vs adaptive change, separated |
| **Deconvolution** | cell-type reference profiles | the Salas 12-cell panel | proportions, constrained to sum to one |

Reporting all six as "biological age" throws away the information needed to interpret any of them.

**Chronological accuracy and usefulness are different axes.** Across 39 biomarkers in more than
20,000 people, the correlation between a clock's chronological-age accuracy and its mortality
prediction was R = 0.12, P = 0.67, uncorrelated
([Nat Aging 2025](https://doi.org/10.1038/s43587-025-00987-y)). A clock that predicts age perfectly
has no acceleration left with which to detect anything.

## Route by the question

| The question | Clocks that ship and score offline | Scale |
|---|---|---|
| How old does this sample look? | `horvath2013`, `hannum`, `skinandblood`, `lin`, `pedbe`, `vidalbralo`, `yingcausage`, `zhangblup`, `zhangen`, `altumage` (20,318-CpG network), `weidner` (three CpGs) | `age_years` |
| How old is this newborn, gestationally? | `knight`, `leecontrol`, `leerobust`, `leerefinedrobust` | `gestational_weeks` |
| How fast is this person aging? | `dunedinpoam38` | `pace_ratio` |
| Who is at risk of dying sooner, or is frailer? | `dnamphenoage`, `phenoage`, `hrsinchphenoage`, `kdm`, `hd`, `zhangmortality` | mixed |
| Is damage separable from adaptation? | `yingdamage`, `yingadaptage` | `age_years_relative` |
| How long are the telomeres? | `dnamtl` | `telomere_kb` |
| How many times has this tissue divided, relative to another sample? | `epitoc1` (mean over 385 polycomb-target CpGs), `hypoclock` (1 minus the mean over 678 solo-WCGWs), `stemtoc` (95th percentile over 371), `stemtocvitro` (over 629), `epicmithyper` (mean over 184), `epicmithypo` (1 minus the mean over 1,164), `replitali` (87 CpGs, linear) | `divisions` |
| How many times has it divided, in divisions per stem cell? | `epitoc2` (163 sites), `epitoc3` (170 sites) | `divisions` |
| Which organ system is aging fastest? | none ship. SystemsAge is licence-restricted | — |
| What is the blood's cell composition? | none ship; the Salas panels are tier C | `proportion` |
| How old is this **mouse**? | `meer` (435 RRBS sites, whole-lifespan multi-tissue) | `age_years`, reported in months |

**The two questions in that pair are different questions.** The seven relative
scores are summaries of a probe set, bounded by 0 and 1, and they compare
samples to each other: higher means more divisions accumulated, or deeper PMD
hypomethylation for the two that take a complement. They do not convert to a
number of divisions and they do not convert to years.

epiTOC2 and epiTOC3 do return a count. Each site carries a de-novo methylation
rate and a fetal ground state, and the model inverts them site by site to
estimate cumulative divisions per stem cell, with a real zero at the fetal
stage. Expect four figures. `acceleration()` refuses on all nine: a division
count is not elapsed time.

Two things worth knowing before you read one of these numbers. epiTOC2 and
epiTOC3 warn when measured betas fall below the fitted fetal ground state,
which makes a site contribute a negative count; that is usually a
normalisation difference and the estimate is reported unchanged rather than
clipped. And `weidner`'s third CpG is a substitution the paper did not make:
its published equation uses an unnamed site upstream of `cg17861230`, measured
by pyrosequencing, which carries 164.7 of the model against 26.4 and 23.7 for
the two real probes. It warns at score time.

**The mouse clock is keyed differently from every other entry.** Its features
are mm10 `chromosome:position`, not `cg` probe ids, which is what
`read_bedmethyl` produces and what an mm10-aligned RRBS pipeline gives you.
Data whose contigs are named some other way will match nothing, and the
coverage floor is what stops that being scored anyway. Its output is months,
converted from the days the model returns.

`fa.registry.load().compatible_with(d)` answers this against a real dataset, which beats any table.

## The nine scales, and what each permits

`scale_type` is enforced, not annotated. `acceleration()` refuses on a scale that does not
support it rather than returning a units error dressed as a number.

| Scale | n | Permitted | Refused, and why |
|---|---:|---|---|
| `age_years` | 73 | acceleration, residual, difference, mean, correlate | — |
| `relative_score` | 44 | correlate, rank | no external unit to difference or average |
| `proportion` | 18 | compositional, difference, mean, correlate | anything ignoring sum-to-one |
| `divisions` | 10 | difference, mean, rank, correlate | acceleration: a division count is not elapsed time |
| `gestational_weeks` | 8 | acceleration, residual, difference, mean | mixing with an age in years |
| `telomere_kb` | 2 | acceleration, difference, mean, correlate | comparison with an age in years; and **higher is younger** here, unlike every age clock beside it |
| `pace_ratio` | 2 | difference, mean, rank, correlate | acceleration: it is already a rate |
| `mortality_log_hazard` | 2 | hazard ratio, mean, rank, correlate | acceleration: a log-hazard has no zero on the age scale |
| `age_years_relative` | 2 | residual, difference, mean, correlate | acceleration: slope against age is near one but the origin moves between cohorts |

`age_years_relative` exists because of a measurement: DamAge tracks age with a pooled slope of
0.967, better than DNAmPhenoAge, but its offset swings **162 years** between cohorts against
Horvath's 15. The residual and group differences are defined; `predicted − chronological` is not.

## Availability, which is about rights and not about quality

| Tier | n | What it means |
|---|---:|---|
| **A** | 46 | Scores offline. 43 ship a coefficient file; 3 (PhenoAge, KDM, homeostatic dysregulation) are formulas with none to ship. |
| **B** | 87 | Catalogued, no traced coefficient source yet. Deliberately not copied out of another package, because that is how the field's paper-versus-implementation discrepancies spread. |
| **C** | 28 | Architecture implemented and tested; coefficients are research-use-only. Supply a licensed file and the same code scores them. |

Every architecture is implemented and tested regardless of tier.

## Traps that bite

**Coverage is not validity.** The mammalian array carries 96% of Horvath2013's 353 probes, so a
zebra scores at *higher* coverage than many human 450K datasets and returns a confident number
from a clock fitted on people. Nothing in the arithmetic notices. Set `species`.

**Counting probes is not weighing them.** An elastic net's weights are nothing like uniform.
96% of probes present can be 61% of the model. `probe_loss()` reports both count coverage and
coefficient-mass coverage; the scoring floor applies to each.

**EPIC v2 renamed the probes.** `cg00000029` became `cg00000029_TC21`. A clock matching on exact
identifiers finds *zero* features and then returns a plausible number computed entirely from
imputed values. Suffix aggregation is mandatory and `prepare()` does it. Even done correctly,
probe loss shifts `hrsinchphenoage` by +16.7 years on EPIC v2 and `hannum` by −7.7.

**Eleven clocks disagree with their own papers.** Bohlin (96 CpGs published, 251 in circulation),
CVDWesterman (1,305 vs 235), ZhangMortality (an integer count published, a continuous sum
implemented), the three Ying clocks (off by one each), and others. Every one carries the note and
raises it as a warning at score time rather than silently inheriting one side.

**Reliability means two things.** Technical ICC repeats the same DNA; biological ICC re-samples
the same person days later. They do not track together, and GrimAge2 and DunedinPACE, the two
most often used to claim an intervention worked, are among the most biologically fragile.

**One significant clock is probably a false positive.** Re-analysing six intervention datasets,
exactly one clock reached significance in five of them, a first-generation clock every time, and
four of those five lost it under multiple-testing correction. `fa.consensus()` implements the
resulting decision rule.
