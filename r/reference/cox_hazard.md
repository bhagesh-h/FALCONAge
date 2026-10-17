# Univariable Cox hazard ratio per clock

Breslow-tied partial likelihood by Newton-Raphson. Deliberately minimal:
competing risks and time-varying covariates belong in a survival
package, and pretending otherwise would be worse than saying so.

## Usage

``` r
cox_hazard(x, time_col, event_col, clocks = NULL)
```

## Arguments

- x:

  A `falcon_result`.

- time_col, event_col:

  Columns in the sample annotation.

- clocks:

  Optional clock names.

## Value

A data frame with `hr`, its 95% interval, `p` and `q`.

## Examples

``` r
if (FALSE) { # \dontrun{
cox_hazard(res, time_col = "permth_exm", event_col = "mortstat")
} # }
```
