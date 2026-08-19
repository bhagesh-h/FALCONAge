# Clock audit: provenance, correctness, and what is missing

An audit of the 161 catalogued clocks against what is published, asking three
questions: is any entry an artefact rather than a real published model, is any
implemented incorrectly, and is any implementable from public material but left
unimplemented. Everything below is measured against the shipped registry and
the public corpus, not recalled.

First pass 2026-08-18; literature survey and corrections 2026-08-19.
Registry schema 1.1.0.

## 1. Are any entries artefacts?

No, on the evidence available. Every clock that scores was run against real
methylation data (GSE107143, 16 samples, ages 32 to 88) and correlated with
chronological age. A fabricated coefficient set cannot do this.

| clock | scale | Pearson r | Spearman | p |
|---|---|---:|---:|---:|
| dnamphenoage | age_years | 0.837 | 0.836 | 5.3e-05 |
| lin | age_years | 0.826 | 0.834 | 8.0e-05 |
| zhangblup | age_years | 0.812 | 0.831 | 1.3e-04 |
| zhangen | age_years | 0.812 | 0.799 | 1.3e-04 |
| hrsinchphenoage | age_years | 0.809 | 0.781 | 1.5e-04 |
| hannum | age_years | 0.795 | 0.787 | 2.3e-04 |
| skinandblood | age_years | 0.790 | 0.764 | 2.7e-04 |
| vidalbralo | age_years | 0.789 | 0.805 | 2.8e-04 |
| yingcausage | age_years | 0.760 | 0.736 | 6.3e-04 |
| horvath2013 | age_years | 0.760 | 0.759 | 6.4e-04 |
| yingadaptage | age_years_relative | 0.659 | 0.599 | 5.5e-03 |
| dnamtl | telomere_kb | **-0.645** | -0.614 | 7.0e-03 |
| yingdamage | age_years_relative | 0.628 | 0.630 | 9.2e-03 |
| zhangmortality | mortality_log_hazard | 0.614 | 0.658 | 1.1e-02 |
| dunedinpoam38 | pace_ratio | 0.081 | -0.034 | 0.76 |

Two rows are worth reading rather than skimming. **DNAmTL is negative**, which
is correct: telomeres shorten with age, and a positive correlation would have
been the defect. **DunedinPoAm38 does not correlate**, which is also correct: it
estimates a rate of change, not a level, and a cross-sectional cohort gives it
nothing to track. Its `scale_type` is `pace_ratio` and the registry already
refuses age acceleration on it. Reading that row as a failure is the mistake
the scale-type system exists to prevent.

Caveat on strength of evidence: n = 16 in one dataset. This rules out
fabrication; it does not establish agreement with each paper's own reported
performance. See item 4.

## 2. Provenance is weaker than the tier system implies

This is the main finding.

| | traced to primary source | not traced |
|---|---:|---:|
| Tier A, scores offline | 7 | **18** |
| Tier B | 0 | 108 |
| Tier C | 0 | 28 |

Seven, not five, since `epitoc1` and `hypoclock` were shipped from the record
their method's author published (section 3). The 18 untraced entries are
unchanged: nothing in this pass re-traced one of them.

All 18 untraced tier A entries carry the same provenance string: *"third-party
package copy; primary source not re-traced"*. Their coefficients were taken
from another package rather than from the paper or its supplement. That is
recorded honestly in the registry and is **not** what the documentation
conveys: the landing page and README frame tier B as the untraced tier, which
leaves a reader to assume tier A is verified against its publication.

Seven entries are traced: `horvath2013`, `dnamphenoage`, `phenoage`, `kdm`, `hd`, and the two shipped in this pass, `epitoc1` and `hypoclock`.

Eleven clocks additionally carry `known_discrepancies` where the paper and the
circulating coefficients disagree, and these are raised as warnings at score
time rather than left on a page: `bocklandt`, `bohlin`, `cvdwesterman`,
`phenoage`, `senchronoage`, `sencultureage`, `senmortalityage`, `yingadaptage`,
`yingcausage`, `yingdamage`, `zhangmortality`.

## 3. Implementable from public material, not implemented

The first pass said several tier B entries were "demonstrably obtainable" and
named them from their model shapes. Going after the material rather than
reasoning about it changed the picture in both directions: more is obtainable
than that list implied, and two of the three clocks it called trivial cannot be
implemented at all.

The mitotic clocks turn out to be one find. Teschendorff publishes the epiTOC2
parameters, the epiTOC1 probe list and the HypoClock probe list as a single
6 kB R object on Zenodo under CC-BY 4.0, and the group's `EpiMitClocks` package
carries the rest under GPL-2, with a reference implementation per clock:

| clock | status now | material | what is in the way |
|---|---|---|---|
| `epitoc1` | **ships** | Zenodo 2632938, `dataETOC2.Rd` element 2, 385 CpGs, CC-BY 4.0 | nothing |
| `hypoclock` | **ships** | same record, element 3, 678 solo-WCGWs | nothing |
| `stemtoc` | obtainable | `EpiMitClocks` `epiTOCcpgs3.rda`, 371 CpGs, GPL-2 | GPL-2 into a GPL-3 tree, or re-extract from Zhu 2024's supplement |
| `stemtocvitro` | obtainable | `cugpmitclockCpG.rda`, 629 CpGs, GPL-2 | as above |
| `epicmithyper` | obtainable | `EpiCMITcpgs.rda`, the 184 rows classed hyper, GPL-2 | as above |
| `epicmithypo` | obtainable | the same file's 1,164 hypo rows | as above |
| `replitali` | obtainable | `Replitali.rda`, an intercept and 87 weights, GPL-2 | as above; it is an ordinary linear model |
| `epitoc2` | data in hand | Zenodo 2632938, 163 CpGs with a de-novo rate and a ground state, CC-BY 4.0 | a model class, not data |
| `epitoc3` | data public | `dataETOC3.rda`, 170 CpGs, GPL-2 | the same class, plus the licence question |
| `altumage` | licence resolved | `rsinghlab/AltumAge`, **MIT** | the weights ship as `.pt` and `.pkl`, which this package refuses to load; `AltumAge.h5` read with h5py converts to safetensors |
| `weidner` | printed in the paper | Genome Biology 15:R24 | two equations exist, one fitted on pyrosequencing percentages and one on beta values; the primary text was not readable in this pass |
| `bocklandt` | **not implementable as published** | PLoS ONE 6:e14821 | no equation in beta units exists |
| `garagnani` | **not implementable as published** | Aging Cell 11:1132 | no age model was ever published |

Two corrections to the first pass, both in the direction of less optimism:

**`bocklandt` and `garagnani` are not "the equation is the paper".** They were
grouped with `weidner` as models small enough to be printed rather than
supplied. Reading them says otherwise. Bocklandt regresses age on EDARADD
methylation, EDARADD squared and NPTX2, measured by Sequenom MassArray and by
pyrosequencing, and reports an R-squared instead of coefficients; there is no
equation in Illumina beta units to implement. Garagnani establishes that ELOVL2
methylation tracks age and publishes no predictor at all. What other packages
ship under both names is a raw beta with weight 1, which is a probe readout and
not an age. Only `weidner` is genuinely a printed model.

**The two mitotic clocks that need a division model still need one.** epiTOC2 is
`2 x mean_i[(beta_i - beta0_i) / (delta_i (1 - beta0_i))]` over the CpGs
present. The divisor is the count of represented CpGs and each CpG carries two
parameters, so neither `LinearClock` nor `AggregationClock` can express it. That
is now a missing class with the data sitting beside it, rather than a missing
data set.

One thing the survey found that nobody was looking for: **the author's own two
implementations of HypoClock disagree in sign.** The 2019 `epiTOC2.R` script
returns the mean beta over the 678 solo-WCGWs; the later `EpiMitClocks` package
returns one minus that mean. FALCONAge ships the later definition, which is also
what pyaging returns, and the registry note now says so instead of describing
pyaging's version as an inversion of the published one.

## 4. Does code align with logic, results and plots?

Checked and holding: the op chain each clock declares is applied in the order
the registry states, `scale_type` gates every downstream operation, the plots
are generated from the same result objects the tables come from, and no figure
is a screenshot. Not yet checked: whether each clock reproduces the accuracy
its own paper reports, which is the difference between "runs and correlates"
and "is correct".

## Todo

Ordered by evidence value per unit of work. Struck items are done.

- [x] ~~**Ship the aggregation clocks whose probe list is public.**~~ `epitoc1`
      and `hypoclock` ship, from the author's CC-BY record, verified to score
      exactly the mean and one-minus-the-mean their papers define.
- [x] ~~**Resolve the AltumAge licence question.**~~ MIT. The block is the
      pickle format, not the licence.
- [x] ~~**Record a `why` for every tier B entry that has been investigated.**~~
      Eleven now carry the source, the licence and the obstacle, and
      `unavailable_message()` prints them instead of the generic "no primary
      source has been established".
- [ ] **Settle GPL-2 material in a GPL-3 tree.** Five clocks are one decision
      away from shipping: `stemtoc`, `stemtocvitro`, `epicmithyper`,
      `epicmithypo`, `replitali`. Either take the lists from the papers'
      supplements instead, or record a decision that a factual probe list
      carries no licence claim.
- [ ] **Build the division-model class** and ship `epitoc2` from the CC-BY
      parameters already traced. `epitoc3` follows if the GPL-2 question above
      is settled. This is the only genuinely new architecture the survey found.
- [ ] **Convert AltumAge's weights safely.** Read `AltumAge.h5` with h5py, write
      safetensors, check against the authors' example data. `NeuralClock` scores
      it as it stands.
- [ ] **Read the Weidner equation out of the primary text.** Three coefficients
      and an intercept, but two published fits: on pyrosequencing percentages
      and on beta values. Getting the wrong one wrong by a factor of a hundred
      is the whole risk, so this needs the paper rather than a quotation of it.
- [ ] **Correct the tier framing in the documentation.** README, `docs/index.qmd`
      and the catalogue imply tier A is source-verified. State that 18 of 25
      are third-party copies.
- [ ] **Trace the 18 untraced tier A clocks to their primary sources.** Start
      with the four that already carry a discrepancy warning (`yingadaptage`,
      `yingcausage`, `yingdamage`, `zhangmortality`), since a mismatch there is
      most likely.
- [ ] **Add a per-clock accuracy check against published performance.** Record
      each paper's reported r or MAE and assert the corpus result is within a
      stated tolerance. This is what turns "correlates" into "agrees with the
      publication".
- [ ] **Investigate the tier B entries nobody has looked at yet.** 97 of the 108
      have no route recorded. `reedbmi` is the cheapest of them: a weighted mean
      whose weights are the model.
- [ ] **Re-run this audit on a larger cohort.** n = 16 rules out fabrication and
      little else. The corpus holds ten studies; use them all and report per
      clock per study.

## Sources consulted in the 2026-08-19 pass

- Teschendorff, *Epigenetic Timer of Cancer-2*, Zenodo, doi:10.5281/zenodo.2632938,
  CC-BY 4.0 (`dataETOC2.Rd`, `epiTOC2.R`)
- `github.com/aet21/EpiMitClocks`, GPL-2 (per-clock reference implementations and
  the probe lists for stemTOC, stemTOCvitro, epiCMIT, RepliTali, epiTOC3)
- `github.com/rsinghlab/AltumAge`, MIT (weights, scaler, CpG list)
- Bocklandt et al. 2011, PLoS ONE 6:e14821
- Garagnani et al. 2012, Aging Cell 11:1132
- Weidner et al. 2014, Genome Biology 15:R24 (abstract and secondary summaries;
  the full text was not reachable in this pass)
