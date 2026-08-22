# Run a call against the Python core, translating its errors

A Python exception becomes an R error whose message is the Python
message, verbatim, with the class name attached. Those messages are
written to be read – `WeightsUnavailableError` names an open alternative
clock, and `UnitsNotDeclaredError` prints the exact `units=` list to
supply – so replacing them with an R-flavoured summary would throw away
the useful part.

## Usage

``` r
py_do(expr)
```

## Arguments

- expr:

  An expression calling into the Python module.

## Value

Whatever `expr` returns.
