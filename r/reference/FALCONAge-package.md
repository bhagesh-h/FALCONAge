# FALCONAge: Multiomic Biological Age and Aging Clock Scoring

Scores DNA methylation and clinical chemistry data against a catalogue
of 175 published aging clocks, and provides the downstream statistics
the field expects: age acceleration in its three conventions, survival
and association models, technical reliability, and the AA1/AA2
benchmark. Numerical work is delegated to the 'falconage' Python package
through 'reticulate', so results from R and from Python are the same
bits rather than two implementations that agree approximately. Forty-six
clocks carry coefficients inside the package and run offline; forty are
implemented as scaffolds whose research-use-only coefficients the user
supplies.

## Why this package is a wrapper

Two implementations of the same clock disagree. Not in the first decimal
– in the fourth, because one of them centred before scaling and the
other did not, or because one filled an absent probe with a cohort mean
and the other with the value the clock's authors published. Those
disagreements are the field's reproducibility problem in miniature, and
a package that shipped an R port alongside a Python one would be adding
to it.

So there is one numerical core, written in Python, and this package
calls it through reticulate. An R result and a Python result are the
same bits. What R gets is not a translation but an idiom: data frames
in, data frames out, S3 classes with `print`, `summary` and `plot`
methods, and ggplot2 rendering of the same numbers matplotlib draws on
the other side.

## Getting started


    library(FALCONAge)
    falconage_install()          # one-off: creates the managed Python env
    falconage_config()           # what resolved: versions, devices, registry

    data <- read_betas("betas.csv")
    res  <- score(data, clocks = "compatible")
    summary(res)
    acceleration(res)

## What ships

175 catalogued clocks. 46 carry coefficients inside the package and run
offline; 40 are scaffolds whose coefficients are research-use-only and
are not ours to distribute – `list_clocks(tier = "licensed")` names each
one, why, and where to obtain a file. The remaining 89 are catalogued
with metadata and await a traced extractor.

## Scales, and what may be done with them

Every clock carries a `scale_type`, and it governs which downstream
operations are defined. Age acceleration is a residual against
chronological age: meaningful for a clock that returns years, undefined
for one that returns a mortality log-hazard, and actively misleading for
a pace of aging, which is already a rate.
[`acceleration()`](https://bhagesh-h.github.io/FALCONAge/r/reference/acceleration.md)
refuses rather than computing, because a wrong number in a column of
numbers is invisible.

## See also

Useful links:

- <https://github.com/bhagesh-h/FALCONAge>

- <https://bhagesh-h.github.io/FALCONAge/>

- <https://bhagesh-h.github.io/FALCONAge/r>

- Report bugs at <https://github.com/bhagesh-h/FALCONAge/issues>

## Author

**Maintainer**: Bhagesh Hunakunti <bhunakun@uni-bonn.de>
([ORCID](https://orcid.org/0000-0002-5957-8005))

Authors:

- Bhagesh Hunakunti <bhunakun@uni-bonn.de>
  ([ORCID](https://orcid.org/0000-0002-5957-8005))
