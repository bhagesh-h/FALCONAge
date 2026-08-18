# Clock audit: provenance, correctness, and what is missing

An audit of the 161 catalogued clocks against what is published, asking three
questions: is any entry an artefact rather than a real published model, is any
implemented incorrectly, and is any implementable from public material but left
unimplemented. Everything below is measured against the shipped registry and
the public corpus, not recalled.

Date: 2026-08-18. Registry schema 1.1.0.

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
| Tier A, scores offline | 5 | **18** |
| Tier B | 0 | 110 |
| Tier C | 0 | 28 |

All 18 untraced tier A entries carry the same provenance string: *"third-party
package copy; primary source not re-traced"*. Their coefficients were taken
from another package rather than from the paper or its supplement. That is
recorded honestly in the registry and is **not** what the documentation
conveys: the landing page and README frame tier B as the untraced tier, which
leaves a reader to assume tier A is verified against its publication.

Five entries are traced: `horvath2013`, `dnamphenoage`, `phenoage`, `kdm`, `hd`.

Eleven clocks additionally carry `known_discrepancies` where the paper and the
circulating coefficients disagree, and these are raised as warnings at score
time rather than left on a page: `bocklandt`, `bohlin`, `cvdwesterman`,
`phenoage`, `senchronoage`, `sencultureage`, `senmortalityage`, `yingadaptage`,
`yingcausage`, `yingdamage`, `zhangmortality`.

## 3. Implementable from public material, not implemented

All 110 tier B entries record no `why`, no `url` and no `obtain` text. There is
no record of whether the coefficients were sought, found, or refused. Several
are demonstrably obtainable.

**Aggregation clocks, needing only a probe list.** `AggregationClock` ships and
is tested; these need the published CpG set and nothing else, which makes them
the cheapest gap in the catalogue: `epitoc1`, `epicmithyper`, `hypoclock`,
`stemtoc`, `stemtocvitro`, `reedbmi`.

**Clocks whose entire model is printed in the paper.** A single-CpG or
three-CpG model has no supplementary file to obtain; the equation is the paper:
`bocklandt` (one CpG in EDARADD), `garagnani` (one CpG in ELOVL2), `weidner`
(three CpGs).

**Architectures specified but unbuilt.** `epitoc2` and `epitoc3` use a mitotic
division model rather than a linear combination, and the authors publish R
code. `MultiStageCoxClock`, `DeconvolutionClock` and `CompositeClock` are named
in the architecture page as specified and unbuilt, and 18 deconvolution entries
depend on the third.

**Weights public but licence unverified.** `altumage` has weights in the
authors' repository and `NeuralClock` ships and is tested against synthetic
weights. Whether the licence permits redistribution has not been recorded.

## 4. Does code align with logic, results and plots?

Checked and holding: the op chain each clock declares is applied in the order
the registry states, `scale_type` gates every downstream operation, the plots
are generated from the same result objects the tables come from, and no figure
is a screenshot. Not yet checked: whether each clock reproduces the accuracy
its own paper reports, which is the difference between "runs and correlates"
and "is correct".

## Todo

Ordered by evidence value per unit of work.

- [ ] **Correct the tier framing in the documentation.** README, `docs/index.qmd`
      and the catalogue imply tier A is source-verified. State that 18 of 23
      are third-party copies. This is a wording fix and it removes a false
      impression from the most-read pages.
- [ ] **Trace the 18 untraced tier A clocks to their primary sources.** Each
      needs the paper's supplement fetched, the coefficient set compared to the
      shipped file, and `primary_source_traced` set with a checksum. Start with
      the four that carry a discrepancy warning already (`yingadaptage`,
      `yingcausage`, `yingdamage`, `zhangmortality`), since a mismatch there is
      most likely.
- [ ] **Add a per-clock accuracy check against published performance.** For each
      tier A clock, record the paper's reported r or MAE and assert the corpus
      result is within a stated tolerance. This is what closes item 4 and turns
      "correlates" into "agrees with the publication".
- [ ] **Ship the six aggregation clocks.** Obtain the published probe lists for
      `epitoc1`, `epicmithyper`, `hypoclock`, `stemtoc`, `stemtocvitro` and
      `reedbmi`. The architecture is already implemented and tested.
- [ ] **Ship the three printed-in-the-paper clocks:** `bocklandt`, `garagnani`,
      `weidner`. No supplement needed.
- [ ] **Record a `why` for every tier B entry.** 110 entries say nothing about
      whether their coefficients are obtainable. Even "not yet attempted" is
      more informative than silence, and it converts the tier into a work list.
- [ ] **Resolve the AltumAge licence question** and ship the weights if it
      permits, since the architecture already exists.
- [ ] **Decide on epiTOC2 and epiTOC3.** They need a division model rather than
      a dot product. Either build it or record the decision not to.
- [ ] **Re-run this audit on a larger cohort.** n = 16 rules out fabrication and
      little else. The corpus holds ten studies; use them all and report per
      clock per study.
