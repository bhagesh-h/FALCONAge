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

## `epitoc_betas.csv` and `epitoc_reference.csv`

Eight synthetic samples over the epiTOC2 (163) and epiTOC3 (170) sites, for checking FALCONAge's
transmission model against the authors' own code. Each beta is the site's ground state plus a
per-sample drift, with Gaussian noise (NumPy `default_rng(11)`), rounded to six decimals. Forty
sites are dropped from the file, leaving 142 of epiTOC2's and 150 of epiTOC3's, and four values
inside present sites are set missing.

`epitoc_reference.csv` holds, per sample, the full (`tnsc`) and simplified (`tnsc2`, all ground
states zero) estimates from Teschendorff's `epiTOC2()` and `epiTOC3()`, and the two estimates from
dnaMethyAge's `epiTOC2()`, all run unmodified in R 4.5.3:

| Source | Files | SHA-256 |
|---|---|---|
| EpiMitClocks 0.1.0 (GPL-2), `aet21/EpiMitClocks` at `2c236cd` | `R/epiTOC2.R` | `a3df6cc838cb9f75ebcc9b84fd6cf045c99bff93e38266bc01738f4de9ac1bac` |
| | `R/epiTOC3.R` | `6149d9875a3a8c470c05946c5a578c4109bec610e7887efe2f156783745acaed` |
| | `data/dataETOC3.rda` | `2720cf457277e4c6831ae365c700aed304d502d6338b3e9c5c1f9b1eb05b9e88` |
| dnaMethyAge 0.2.0 (GPL-3), `yiluyucheng/dnaMethyAge` at `0d40c9b` | `R/epiTOC2.R` | `27f8d0aac95bf944cba24743baf9c195708fb3ae31eda95a2c85ef6aae281f0f` |
| | `data/epiTOC2.rda` | `6d209ee449fc01ae170b082244970945ebcff3b7a11d9bebef463a0e5e05fe09` |

Both packages carry the same per-site `delta` and `beta0` as FALCONAge's `epitoc2.csv` and
`epitoc3.csv`, to the last digit. Both average over the sites present. They differ in one case:
Teschendorff's `colMeans(diag(w) %*% M, na.rm = TRUE)` returns NA for a sample with any missing
value at a present site, because the matrix product turns `0 * NA` into NA before `na.rm` can drop
it; dnaMethyAge's element-wise rewrite skips the value. FALCONAge agrees with Teschendorff wherever
his code returns a number, and with dnaMethyAge everywhere, to a relative 1e-9.

| File | SHA-256 |
|---|---|
| `epitoc_betas.csv` | `94a7c3f4f9684bd9982e4ccc2ecd7558e9b78177bf1f1890d3b6ec8c21ac5617` |
| `epitoc_reference.csv` | `ced4aae7d542c71fb56099e2a331515b40d03c126f0ee4f3bbf4258eb4439935` |

## `nhanes3_phenoage0_fixture.csv.gz`

The 8,924 NHANES III rows for which BioAge ships `phenoage0`, with the ten PhenoAge inputs in the
paper's units and BioAge's value, so that `phenoage(df, crp_transform="log1p",
coefficients="bioage")` is checked against the reference implementation on every run.

| | |
|---|---|
| Source file | `NHANES3.rda` from the BioAge R package (as for the `kdm0` fixture above), SHA-256 `c1b940fc9614add6effaa3ea6d189677c5cd8532f46be79284dce695ae3e0b92` |
| Rows kept | every row with a non-missing `phenoage0` |
| Columns kept | `sampleID`, `age`, and BioAge's `albumin_gL`, `creat_umol`, `glucose_mmol`, `crp` (mg/dL), `lymph`, `mcv`, `rdw`, `alp`, `wbc` renamed to FALCONAge's marker names, and `phenoage0` |
| Precision | ten significant figures, so the reproduction is exact rather than approximate |
| Fixture SHA-256 | `343738820be2978bd8a0e3daa61c4880a7ff6003b965834b28becafccb01f0eb` |

BioAge computes `phenoage0` in `R/phenoage_calc.R` from the full-precision weights and `lncrp =
log(1 + crp)`. FALCONAge reproduces it to below 0.00001 years on every row. With Table 1's printed
weights instead, it is a mean 0.074 years higher. Licence and citation as for the `kdm0` fixture.

## `zhang_betas.csv.gz` and `zhang_reference.txt`

Eight synthetic samples for checking FALCONAge's Zhang et al. 2019 predictors against the
authors' own script. The 4,494 probes are 494 of the elastic-net model's 514 (20 left out), 2,500
of the BLUP model's other probes and 1,500 probes neither model uses, drawn from a Beta(0.6, 0.6)
per probe with Gaussian noise (NumPy `default_rng(19)`), rounded to six decimals. Four values are
missing and one probe is missing in every sample, so the per-probe mean fill and the all-NA
filter are both exercised.

`zhang_reference.txt` is the output of `pred.R` from `qzhang314/DNAm-based-age-predictor` at
`18f20a4`, run unmodified in R 4.5.3 on this matrix (`Rscript pred.R -i input.rds -o
zhang.pred -a ages`): per sample, the chronological age given to it and the elastic-net and BLUP
predictions.

| File | SHA-256 |
|---|---|
| `pred.R` | `9c2a45aa89cc223e76af5654e82c5ceb19c9a8225ffa7c20aef57fc71a07e5e3` |
| `en.coef` | `c8096eadc79dd3f9dc44544293150ed24043616332130bac2cc7dd7d604a4bc0` |
| `blup.coef` | `d4802729177da3e5f3506ce0e652549594ad2d2b74b21d544b9e6572c3175777` |
| `zhang_betas.csv.gz` | `b780e5c9e356507c13a7e64a21c6b803c1015ad46ae34c33496412d7dc0c1130` |
| `zhang_reference.txt` | `9470dbb02bc90fea96a27c6ec1ef873d25e621d3fc0ae5ccad06445030370b1c` |

FALCONAge agrees to 5e-14 years. On the authors' own example, `data.rds` (10 samples by the full
485,512 probes of the 450K array; SHA-256 `7a82f8fed2a7cddf09f228a6eebc755ce248f5e48ab4c54eef45279376ef914e`),
it agrees to 1e-13 years for the elastic net and 5e-12 for BLUP with the default coverage floor;
that file is 38 MB and is not kept here.

## `knight_testdataset.csv.gz`

Knight et al.'s demonstration dataset for their gestational-age clock, so that the clock and
Horvath's gold-standard normalisation can be checked against the output the authors state for it:
37.366, 38.346 and 39.324 weeks (Additional file 7, the instructions).

| | |
|---|---|
| Source file | Additional file 6 (`TestDataset.csv`) of Knight AK et al., Genome Biology 2016;17:206, `https://static-content.springer.com/esm/art%3A10.1186%2Fs13059-016-1068-z/MediaObjects/13059_2016_1068_MOESM6_ESM.csv` |
| Source SHA-256 | `629c4dd3106263f19fffa4395da97fb5c718f77c5626b6dc2964713aa4c919a8` |
| Rows kept | the 21,265 of its 27,235 probes that are in Horvath's goldstandard2, the only ones the normalisation and the clock read |
| Values | as published, line endings converted from CR to LF |
| Fixture SHA-256 | `38e0a03114163c824de6a432cf36758862e15fdac5b53b6cac6e6d4c0d70fa6c` |

**Licence and citation.** CC BY 4.0. Cite: Knight AK, Craig JM, Theda C, et al. An epigenetic
clock for gestational age at birth based on blood methylation data. Genome Biol 2016;17:206.
https://doi.org/10.1186/s13059-016-1068-z
