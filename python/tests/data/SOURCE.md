# Test fixtures derived from published data

## `nhanes3_kdm0_fixture.csv.gz`

The NHANES III rows BioAge trains its `kdm0` Klemera-Doubal model on, with BioAge's own `kdm0`
value for each, so that FALCONAge's KDM can be checked against the reference implementation on
every test run rather than only where the full corpus has been downloaded.

| | |
|---|---|
| Source file | `NHANES3.rda` from the BioAge R package, `https://raw.githubusercontent.com/dayoonkwon/BioAge/master/data/NHANES3.rda` |
| Source SHA-256 | `c1b940fc9614add6effaa3ea6d189677c5cd8532f46be79284dce695ae3e0b92` |
| Rows kept | `age` 30 to 75 and `pregnant == 0`, both sexes: 11,363 rows, 9,583 with a `kdm0` |
| Columns kept | `sampleID, gender, age`, the nine `kdm0` biomarkers `fev, sbp, totchol, hba1c, albumin, creat, lncrp, alp, bun`, and `kdm0` |
| Precision | six significant figures |
| Fixture SHA-256 | `37c54b4a70186cc537c2bd57e10c6f972db8edcf62b19369d69ea089ec80b663` |

The filter and the biomarker list are the ones BioAge's `data-raw/nhanes_all.R` uses to produce
`kdm0` (lines 944-951 at the time of extraction): trained separately by sex on NHANES III aged 30
to 75, non-pregnant. `lncrp` in BioAge is `log(1 + crp)` with CRP in mg/dL.

Rounding to six significant figures does not change the result: FALCONAge fitted on this file
reproduces `kdm0` to 0.0007 years mean absolute difference (maximum 0.007 years), with the same
rows missing.

**Licence and citation.** NHANES is a US federal public-domain dataset. The extraction,
harmonisation and variable naming are the BioAge authors' work, distributed under GPL-3, which
FALCONAge's GPL-3.0-or-later covers. Cite: Kwon D, Belsky DW. A toolkit for quantifying aging in
humans. GeroScience 2021;43:2795-2808. https://doi.org/10.1007/s11357-021-00480-5

## `cox_ties.csv`

400 synthetic subjects with heavily tied event times (295 of 400 times shared), for checking the
Cox fit against R's `survival::coxph(..., ties = "breslow")`, the reference implementation of
Breslow's handling of ties. Generated once with NumPy `default_rng(7)`: `x`, `age`, `sex`,
exponential event times with hazard `exp(0.5 x + 0.04 (age - 55) + 0.3 sex)` rounded up to whole
months, uniform censoring. Stored rather than regenerated because NumPy does not promise
identical random streams across versions.

Reference values, R 4.5.3 with `survival` 3.8.6, `ties = "breslow"`:

| Model | Coefficients | Standard errors |
|---|---|---|
| `~ x` | 0.424830 | 0.069674 |
| `~ x + age + sex` | 0.512266, 0.039324, 0.419025 | 0.071065, 0.004888, 0.127651 |
