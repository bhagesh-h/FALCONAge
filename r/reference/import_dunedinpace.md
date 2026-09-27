# Register DunedinPACE from the authors' R package

DunedinPACE (Belsky et al. 2022, eLife 11:e73420) is research use only
and is not distributed with FALCONAge. Given the authors' package,
installed or as a source checkout, or its `R/sysdata.rda`, this reads
the weights, the 20,000-probe background and its means, and registers
the clock. Scoring then follows the package's `PACEProjector()`:
quantile normalisation of every sample to the background, its
missing-value rules, and its 0.8 coverage threshold (0.7 on EPIC v2).

## Usage

``` r
import_dunedinpace(path, out_dir = NULL)
```

## Arguments

- path:

  The installed package directory, a source checkout, or `sysdata.rda`.

- out_dir:

  Where to write the extracted parameters; by default a cache folder in
  the home directory.

## Value

The SHA-256 of the registered weight file, invisibly.

## Examples

``` r
if (FALSE) { # \dontrun{
import_dunedinpace(system.file(package = "DunedinPACE"))
score(d, clocks = "dunedinpace")
} # }
```
