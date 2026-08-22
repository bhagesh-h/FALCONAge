# Install the Python core FALCONAge computes with

Creates a managed virtual environment and installs the `falconage`
Python package into it. Run once per machine;
[`falconage_config()`](https://bhagesh-h.github.io/FALCONAge/r/reference/falconage_config.md)
reports what it resolved to.

## Usage

``` r
falconage_install(
  envname = "r-falconage",
  method = c("auto", "virtualenv", "conda"),
  extras = c("methylation", "plot", "anndata"),
  gpu = FALSE,
  cuda = "cu124",
  version = NULL,
  ...
)
```

## Arguments

- envname:

  Environment name. The default keeps it out of the way of any other
  reticulate project on the machine.

- method:

  `"auto"`, `"virtualenv"` or `"conda"`, passed to reticulate.

- extras:

  Optional extras to install alongside the core. `"methylation"` adds
  the parquet reader; `"plot"` adds matplotlib; `"anndata"` adds the
  `.h5ad` reader. Note that `"gpu"` is deliberately *not* in this list –
  see `gpu` below.

- gpu:

  Install CUDA torch as well. Off by default, and not merely out of
  caution: on the clocks that ship today the GPU is slower than the CPU
  (0.58 s against 3.74 s at 16,384 samples on an RTX 4060), because a
  linear clock over a few thousand features is too small a matrix
  multiplication to pay for the transfer. It earns its place on the PC
  clocks and on neural architectures, none of which are tier A yet.

- cuda:

  CUDA version for the torch wheel index, e.g. `"cu124"`. The `gpu`
  extra alone is not enough: pip's default index serves a CUDA build on
  Linux and a CPU-only build on Windows under the same name, so the
  wheel has to be requested from PyTorch's own index explicitly.

- version:

  Git ref to install from. A version number such as `"1.0.0"` becomes
  the tag `v1.0.0`; anything else – `"main"`, a branch, a commit SHA –
  is used verbatim. Defaults to the version of this R package.

- ...:

  Passed to
  [`reticulate::py_install()`](https://rstudio.github.io/reticulate/reference/py_install.html).

## Value

Invisibly, the resolved configuration from
[`falconage_config()`](https://bhagesh-h.github.io/FALCONAge/r/reference/falconage_config.md).

## Why an environment of its own

The alternative – installing into whatever interpreter reticulate finds
– works until the day something else in that interpreter upgrades numpy,
and then the same script gives a different answer in the seventh decimal
with no change to anything the user can see. A named environment makes
the dependency set a property of the package rather than of the machine.

## Where the core comes from

FALCONAge is not on PyPI, so this installs from the GitHub repository.
The Python package sits in the `python/` subdirectory, which the
`#subdirectory=python` fragment selects, and the ref is pinned to the
tag matching this R package's version rather than left on `main`. Those
two halves have to agree: the R suite asserts bit equality against the
core, and a mismatched pair fails that assertion in a way that reads as
a numerical bug rather than as version skew.

## Examples

``` r
if (FALSE) { # \dontrun{
falconage_install()
falconage_install(extras = c("plot", "anndata"))
falconage_install(gpu = TRUE, cuda = "cu124")
falconage_install(version = "main")   # track the development branch
} # }
```
