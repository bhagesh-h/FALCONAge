<!-- The Docker Hub repository overview for bhagesh/falconage.

     Docker Hub cannot generate this from the GitHub repository, so it is kept
     here and updated alongside the images. Paste it into the repository's
     Overview tab. Keep it short: it is a landing page, not the documentation,
     and the documentation is one link away. -->

# FALCONAge

Biological age and aging-clock scoring from DNA methylation, clinical
chemistry and NMR metabolomics data. 68 human clocks, every one scoring
offline with coefficients bundled in the image. Python and R return the same
numbers, because the R package delegates every calculation to the same Python
core.

- **Documentation:** https://bhagesh-h.github.io/FALCONAge/
- **Your first run:** https://bhagesh-h.github.io/FALCONAge/start/quickstart.html
- **Source:** https://github.com/bhagesh-h/FALCONAge

## Tags

| Tag | Contains | Size |
|---|---|---|
| `1.0.0-cpu`, `latest` | Python, R, the `falconage` command, every reader | 1.55 GB |
| `1.0.0-py` | Python and the `falconage` command; no R, no parquet or `.h5ad` reader | 0.84 GB |

Sizes are uncompressed.

## Run it

```bash
# the whole pipeline on a small synthetic dataset, in about a minute
docker run --rm -v "$PWD:/work" bhagesh/falconage:1.0.0-cpu demo

# the whole pipeline on your data: scores, uncertainty, figures, report.html
docker run --rm -v "$PWD:/work" bhagesh/falconage:1.0.0-cpu run betas.csv --samples samples.csv

# a Python or R session in the same environment
docker run --rm -it -v "$PWD:/work" bhagesh/falconage:1.0.0-cpu python
docker run --rm -it -v "$PWD:/work" bhagesh/falconage:1.0.0-cpu R
```

On Windows PowerShell write `"${PWD}"`; on `cmd.exe`, `"%cd%"`. The sample
sheet is optional: first column the sample id, then any of `age`, `sex`,
`tissue`, `group`.

## What it refuses to do

Refusals are the design. Scoring a clock off its training tissue is refused or
warned about, because saliva and buffy coat from the same people differ by 3.83
to 16.46 years while still correlating at Spearman 0.45 to 0.69. A cohort-centred
clock on a single sample is refused, because centring one row against itself
returns the intercept for everybody. Coverage is enforced twice, on the fraction
of probes present and on the fraction of coefficient mass they carry, because a
dataset can pass the first and fail the second.

Aging clocks are population-research instruments. A score is interpretable
against a comparison group; it is not a diagnostic statement about one person,
and no clock here is validated for clinical use.

GPL-3.0-or-later.
