# Everything the registry knows about one clock

For a licensed clock this also prints why its coefficients are not
distributed, where to obtain them, and which open clocks answer the same
question.

## Usage

``` r
clock_info(clock_id)
```

## Arguments

- clock_id:

  A clock identifier.

## Value

A named list, invisibly. Printed for reading.

## Examples

``` r
if (FALSE) { # \dontrun{
clock_info("horvath2013")
clock_info("grimage2")
} # }
```
