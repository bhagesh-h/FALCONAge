# Register the PC clocks from the authors' data file

The PC clocks (Higgins-Chen et al. 2022, Nature Aging 2:644) are scored
in their authors' code from one file, `CalcAllPCClocks.RData`, which the
PC-Clocks repository links on Yale Box and which carries no licence, so
it is not distributed with FALCONAge. Given a downloaded copy, this
registers PCHorvath 2013, PCSkinAndBlood, PCHannum, PCPhenoAge and
PCDNAmTL, each collapsed exactly to one weight per CpG plus a constant,
with the authors' fill values for CpGs the data lacks. It reproduces the
authors' `run_calcPCClocks.R`. PCGrimAge is not imported: it needs a
composite model.

## Usage

``` r
import_pc_clocks(path, out_dir = NULL)
```

## Arguments

- path:

  The downloaded `CalcAllPCClocks.RData`.

- out_dir:

  Where to write the extracted weights; by default a folder next to
  `path`.

## Value

The SHA-256 of each registered file, by clock, invisibly.

## Examples

``` r
if (FALSE) { # \dontrun{
import_pc_clocks("~/PC-Clocks/CalcAllPCClocks.RData")
score(d, clocks = c("horvath2013", "pchorvath2013"))
} # }
```
