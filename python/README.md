# FALCONAge (Python)

Multiomic biological age and aging clock scoring, identical in Python and R.

Not on PyPI. This package is the `python/` subdirectory of the FALCONAge repository, which is what
the `#subdirectory=python` fragment below selects:

```bash
pip install "falconage @ git+https://github.com/bhagesh-h/FALCONAge.git#subdirectory=python"
```

Pin a tag in anything you intend to reproduce - `...FALCONAge.git@v1.0.0#subdirectory=python`.
`main` moves; a tag does not.

```python
import falconage as fa

data = fa.read("betas.csv")
res  = fa.score(data, clocks="compatible")
res.summary()
```

161 catalogued clocks. Twenty-three run offline today - twenty carry a coefficient file, and
PhenoAge, KDM and homeostatic dysregulation are formulas with no coefficients to carry.
Twenty-eight are scaffolds whose coefficients are research-use-only and are not ours to
distribute. The remaining 110 are catalogued and await a traced extractor.

```python
fa.registry.load().filter(availability="A")   # the ones that work today
```

The R package `FALCONAge` wraps this same core through reticulate, so an R result and a Python
result are the same bits rather than two implementations that agree to six decimals.

Full documentation, the clock catalogue, and the list of clocks that need author permission:
<https://bhagesh-h.github.io/FALCONAge/>
