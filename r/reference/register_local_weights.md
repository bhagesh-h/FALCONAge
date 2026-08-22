# Supply a coefficient file for a clock FALCONAge does not distribute

Twenty-eight clocks ship as scaffolds: the model, the feature list, the
preprocess and postprocess chain, the expected shapes – everything
except the numbers, which are research-use-only. Once you hold a
licensed file, this registers it.

## Usage

``` r
register_local_weights(clock_id, path, sha256 = NULL)
```

## Arguments

- clock_id:

  A clock identifier, e.g. `"grimage2"`.

- path:

  A CSV with `feature_id,coefficient` columns.

- sha256:

  Optional expected digest; a mismatch is an error.

## Value

The file's SHA-256, invisibly.

## Details

Registration validates the file against the scaffold and rejects a
mismatch with the discrepancy named, which also makes it a way to check
a coefficient set somebody handed you. The digest goes into the run
manifest as `user_supplied`, so a result computed from a licensed copy
is distinguishable from one computed from a redistributed set.

## Examples

``` r
if (FALSE) { # \dontrun{
register_local_weights("grimage2", "~/licensed/grimage2_coefs.csv")
score(d, clocks = "grimage2")
} # }
```
