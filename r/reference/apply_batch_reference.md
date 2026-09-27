# Apply a frozen batch-correction reference

Apply a frozen batch-correction reference

## Usage

``` r
apply_batch_reference(data, reference, batch_col)
```

## Arguments

- data:

  A `falcon_data` to correct.

- reference:

  A `falcon_batch_reference` from
  [`fit_batch_reference()`](https://bhagesh-h.github.io/FALCONAge/r/reference/fit_batch_reference.md).

- batch_col:

  Column naming the batch.

## Value

A corrected `falcon_data`, recording the reference's digest.
