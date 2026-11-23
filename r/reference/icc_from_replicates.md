# Per-probe reliability from your own technical replicates

Preferred over the bundled table whenever it is available, because it is
this laboratory's noise on this platform rather than a published
cohort's. A one-way random-effects single-measurement ICC is the right
model when the repeated measurements are interchangeable array positions
rather than named assessors.

## Usage

``` r
icc_from_replicates(data, subject_col)
```

## Arguments

- data:

  A `falcon_data` containing replicates.

- subject_col:

  Column naming the subject each sample came from.

## Value

A named numeric vector of ICCs, one per feature.

## Details

Negative values are kept rather than clipped: a negative ICC means the
within-subject spread exceeded the between-subject spread, which is a
real and reportable state of affairs for a probe that measures nothing.
