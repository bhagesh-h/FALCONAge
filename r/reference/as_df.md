# Convert a Python pandas object to an R data frame

Goes through plain Python lists rather than letting reticulate convert
the DataFrame directly. That is deliberate: reticulate's pandas
conversion is registered against a class name that has changed between
releases (1.42 reports `pandas.DataFrame` where earlier versions
reported `pandas.core.frame.DataFrame`), so `py_to_r()` on a DataFrame
silently returns the proxy untouched on some installations and a data
frame on others. A package whose return type depends on the user's
reticulate version is not usable, and the failure is quiet –
[`nrow()`](https://rdrr.io/r/base/nrow.html) on the proxy gives `NULL`,
not an error.

## Usage

``` r
as_df(x, row_names = TRUE)
```

## Arguments

- x:

  A pandas DataFrame or Series proxy.

- row_names:

  Keep the pandas index as row names.

## Value

A data frame.

## Details

`to_dict("list")` and `index.tolist()` are stable across every pandas
and reticulate version in use, at the cost of one extra copy.
