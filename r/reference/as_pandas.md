# Convert an R data frame to a pandas DataFrame, keeping row names as the index

Row names matter here in a way they usually do not in R: they are the
sample identifiers, and every join downstream is on them. reticulate's
default conversion drops them.

## Usage

``` r
as_pandas(df)
```

## Arguments

- df:

  A data frame.

## Value

A pandas DataFrame proxy.
