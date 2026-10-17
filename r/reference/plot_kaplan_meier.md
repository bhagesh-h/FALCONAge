# Survival by age acceleration

Kaplan-Meier curves for the fastest-ageing tail against the slowest,
with the log-rank p-value in the subtitle.

## Usage

``` r
plot_kaplan_meier(
  x,
  clock,
  time_col,
  event_col,
  age_col = "age",
  quantile = 0.1
)
```

## Arguments

- x:

  A `falcon_result`.

- clock:

  Which clock's acceleration to stratify on.

- time_col, event_col:

  Columns of the sample annotation holding follow-up time and the event
  indicator (1 = event, 0 = censored).

- age_col:

  Column holding chronological age.

- quantile:

  Size of each tail; 0.1 gives top and bottom 10%.

## Value

A ggplot.

## Why the tails and not a median split

The middle of the acceleration distribution is where a clock
discriminates least, so pooling it into two halves dilutes whatever
signal the tails carry. The published convention is the extreme deciles.

## Examples

``` r
if (FALSE) { # \dontrun{
plot_kaplan_meier(res, "horvath2013", time_col = "time", event_col = "status")
} # }
```
