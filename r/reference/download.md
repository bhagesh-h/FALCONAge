# Download public data by accession

Dispatches on the shape of the accession: `GSE*`/`GSM*` to GEO,
`E-MTAB-*` to ArrayExpress, a DOI to Zenodo, `owner/name` to Hugging
Face, or a full https URL fetched directly.

## Usage

``` r
download(accession, want = NULL, dry_run = FALSE)
```

## Arguments

- accession:

  An accession or URL.

- want:

  For GEO series: `"matrix"` (the default, metadata and values in one
  file), `"suppl"`, or `"both"`.

- dry_run:

  List what would be fetched, and how much, without fetching. Worth
  using first on a large series: a supplementary directory can be
  several gigabytes.

## Value

A list with `files` (paths), `samples` (a data frame, when the source
provides one) and `notes`.

## Details

Credentialed archives – dbGaP, EGA, Synapse, UK Biobank – are documented
rather than automated. The access agreement is between you and the
archive, and a tool that made it one function call would be inviting
people to breach it. Once the files are local, read them with
[`read_betas()`](https://bhagesh-h.github.io/FALCONAge/r/reference/read_betas.md)
or
[`read_series_matrix()`](https://bhagesh-h.github.io/FALCONAge/r/reference/read_series_matrix.md).

## Examples

``` r
if (FALSE) { # \dontrun{
d <- download("GSE182991")
d$files
head(d$samples)

download("GSE182991", want = "suppl", dry_run = TRUE)
} # }
```
