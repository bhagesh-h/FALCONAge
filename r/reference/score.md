# Score a dataset against one or more clocks

Score a dataset against one or more clocks

## Usage

``` r
score(
  data,
  clocks = "compatible",
  device = "auto",
  dtype = NULL,
  imputation = c("reference", "mean", "none"),
  min_coverage = 0.8,
  reference = NULL
)

# S3 method for class 'falcon_result'
print(x, ...)

# S3 method for class 'falcon_result'
summary(object, ...)
```

## Arguments

- data:

  A `falcon_data`.

- clocks:

  `"compatible"` scores everything this dataset can support and reports
  the rest as skipped with a reason. `"all"` attempts every clock of the
  right modality and fails loudly on the ones that cannot run. A
  character vector names them explicitly, and then every name must work
  – an explicit request is never silently dropped.

- device:

  `"auto"`, `"cpu"`, `"cuda"` or `"mps"`. Naming a device that is not
  present is an error rather than a silent downgrade: a run that was
  asked for a GPU and quietly used a CPU looks like a very slow success.
  `"auto"` is CPU even where CUDA exists, which is measured rather than
  cautious: on the clocks that ship, the transfer costs more than the
  dot product it feeds. A named device is not granted to every clock
  either – the three clinical formulas compute in numpy whatever is
  asked, because nine markers are not worth a kernel launch.
  `manifest(res)$compute` records what each clock actually used.

- dtype:

  `NULL` (float64), `"float64"` or `"float32"`. Clocks flagged
  `requires_fp64` in the registry override a float32 request, with a
  warning.

- imputation:

  How to fill a clock feature the data does not carry. `"reference"`
  uses the value the clock's authors published where one exists;
  `"mean"` uses the column mean; `"none"` refuses, so the coverage check
  fails loudly instead of being papered over. Zero is never used – in
  beta space it means completely unmethylated, which is a real and
  extreme measurement.

- min_coverage:

  Fraction of a clock's features that must be present.

- reference:

  A reference fitted with
  [`fit_kdm()`](https://bhagesh-h.github.io/FALCONAge/r/reference/fit_kdm.md)
  or
  [`fit_hd()`](https://bhagesh-h.github.io/FALCONAge/r/reference/fit_hd.md),
  for the two clinical clocks that have no fixed coefficients.

- x:

  A `falcon_result`.

- ...:

  Ignored.

- object:

  A `falcon_result`.

## Value

A `falcon_result`: an S3 object wrapping the scores, the per-clock
coverage and the run manifest.

## Examples

``` r
if (FALSE) { # \dontrun{
res <- score(d, clocks = "compatible")
res <- score(d, clocks = c("horvath2013", "dnamphenoage"), device = "cuda")
} # }
```
