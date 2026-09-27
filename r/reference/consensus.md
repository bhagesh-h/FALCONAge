# Does a group difference hold up across clocks?

Implements the decision rule from *When to Trust Epigenetic Clocks*
(PMC11526921). Re-analysing six intervention datasets, the authors found
that in five of them exactly one clock reached significance – a
first-generation clock every time – and four of those five lost it under
multiple-testing correction. In no case did the principal-component
version of the same clock corroborate the finding. Their conclusion,
stated plainly: a single significant clock after an intervention is
likely a false positive.

## Usage

``` r
consensus(
  x,
  group_col,
  reference = NULL,
  alpha = 0.05,
  design = "independent",
  subject_col = NULL
)
```

## Arguments

- x:

  A `falcon_result`.

- group_col:

  Column with exactly two levels, or with two or more visits for
  `design = "mixed"`.

- reference:

  Which level is the comparison group.

- alpha:

  Significance level.

- design:

  `"independent"` (Welch's t-test), `"paired"` or `"mixed"`.

- subject_col:

  Column naming the person each sample came from; needed by the paired
  and mixed designs.

## Value

A list with `verdict` (`supported`, `unsupported`, `inconclusive`),
`why` – which always carries the counts it was computed from – and the
per-clock `table`.

## Details

Each clock is tested on its acceleration residual where that is a legal
operation for its scale, and on the raw score where it is not. A pace of
aging has no residual to take.

Repeated measurements on the same people take `design = "paired"` (two
visits, a paired t-test on each person's change) or `design = "mixed"`
(two or more visits, `y ~ visit + (1 | person)` by REML with nlme's
containment degrees of freedom, so a person with a missed visit still
counts). Both add `mdc95`, the minimum detectable change
\\1.96\sqrt{2}\\SEM\\ (Weir 2005, doi:10.1519/15184.1), and flag
significant mean changes smaller than it.
