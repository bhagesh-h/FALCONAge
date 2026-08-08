# Publishing FALCONAge

Where each half of this package can be distributed, what each route needs, and — the part most
release documents leave out — what is still missing for the routes not yet taken.

Nothing here is on a registry today. `pip install falconage` and `install.packages("FALCONAge")`
both find nothing, and the README says so. This file is the plan for changing that.

## The routes, in the order they become available

| Route | Status | Gate |
|---|---|---|
| **GitHub tag** | working now | none. `release.yaml` builds both artefacts on a `v*` tag |
| **r-universe** | one commit away | add this repository to a `packages.json` in a `<user>.r-universe.dev` repository |
| **PyPI** | needs a decision | a trusted-publisher claim on the project, then enable the upload step |
| **CRAN** | needs work | see the gap table below |
| **Bioconductor** | probably not | see below |
| **GHCR (Docker)** | needs a decision | no image is published; both build from source |

## 1. GitHub tags — the current channel

A tag is the whole distribution mechanism right now, and it is a real one: both package managers
install from a git ref, and a tag is immutable in the way a branch is not.

```bash
# bump all three version strings first -- release.yaml checks every one
#   python/pyproject.toml   [project] version
#   r/DESCRIPTION           Version:
#   CITATION.cff            version:
git tag v1.0.0 && git push origin v1.0.0
```

The workflow then builds the Python sdist and wheel, builds the R source tarball, checks the
tarball as CRAN would with `error_on = "warning"`, and attaches everything to the release.

The three version checks exist because they have each failed somewhere. A tag that disagrees with
the package version produces a release whose tarball installs as a different version than its
name, which is unfixable afterwards — the tag is what a citation points at. A `CITATION.cff` left
at the previous version is worse than no citation file, because it is confidently wrong and
GitHub renders it in the sidebar.

## 2. r-universe — the next step, and nearly free

r-universe builds and checks the package on Windows, macOS and Linux on every push, and gives
users `install.packages("FALCONAge", repos = "https://bhagesh-h.r-universe.dev")` with binaries
and no compiler. It has no gatekeeper and no review.

Set-up is one file in a separate repository named `bhagesh-h.r-universe.dev`:

```json
{"packages": [{"package": "FALCONAge",
               "url": "https://github.com/bhagesh-h/FALCONAge",
               "subdir": "r"}]}
```

`subdir` matters — the R package is not at the repository root.

Nothing in `.github/workflows/` is needed for this, which is why it is documented here rather
than automated.

The one caveat worth knowing before it surprises you: r-universe will build the package, and its
check will skip every test that needs the Python core, because r-universe has no way to install
one. A green r-universe badge therefore means "the R package builds and its pure-R tests pass",
not "R and Python agree". The conformance gate lives in `R-CMD-check.yaml`, which builds a Python
environment first.

## 3. PyPI

The mechanics are done: `release.yaml` builds and `twine check`s the distributions on every tag.
Publishing needs two things that cannot be done from a workflow file.

1. A **trusted publisher** claim configured on PyPI for this repository and workflow. Trusted
   publishing means no API token in repository secrets, which is the correct configuration and
   also one fewer thing to leak.
2. A decision about the **name**. `falconage` was free at the time of writing; verify before
   claiming, because a name taken between then and now changes the import path in every document
   here.

Then add an upload job gated on `startsWith(github.ref, 'refs/tags/v')` with
`permissions: id-token: write` and `pypa/gh-action-pypi-publish`. Test against TestPyPI first —
a bad upload to PyPI cannot be replaced, only yanked.

## 4. CRAN

The honest gap table. None of these is hard; together they are a week.

| Requirement | State | What is needed |
|---|---|---|
| `R CMD check --as-cran` clean | passes locally and in CI | keep it that way |
| No `Suggests`-only failures | handled | `_R_CHECK_FORCE_SUGGESTS_: false` in CI; the package degrades without ggplot2 by design |
| Examples run in < 5 s each | **not met** | every example is `\dontrun{}` because it needs a Python environment. CRAN tolerates this but reviewers ask; the fix is a tiny bundled fixture that scores one clock on ten samples |
| Tests pass without network | met | the corpus tests skip when `test/data` is absent |
| Tests pass without Python | met | every conformance test skips with a reason rather than failing |
| Package size < 5 MB | met | the R half carries no data; the clock catalogue is in the Python core |
| `\value` on every exported function | met | roxygen `@return` throughout |
| No writing outside `tempdir()` | met | `write_results()` requires an explicit path |
| Reverse-dependency safety | n/a | nothing depends on it yet |

The real question for CRAN is not any of the above. It is whether a package whose numerical core
is a Python dependency belongs there at all. CRAN accepts reticulate packages — tensorflow and
keras are on it — but the review is stricter and the maintenance burden is real: every CRAN check
machine that cannot build the Python environment produces a NOTE you have to explain again at
every release.

## 5. Bioconductor

Plausible on subject matter — the `biocViews` in `DESCRIPTION` are real ones, and methylation
tooling is squarely Bioconductor's area. Two things argue against it.

The package does not use Bioconductor classes. It takes a matrix and returns a data frame; there
is no `SummarizedExperiment` in the interface, and Bioconductor review will ask why not. Adopting
one for the sake of the submission would add a heavy dependency to a package that currently needs
three.

And Bioconductor's release cycle is twice a year against a pinned R version. The clock registry
changes when a coefficient set is traced, which is not on that schedule.

r-universe covers the same audience with none of this, which is why it is step 2 and this is a
maybe.

## 6. Docker images

Both images build from source and neither is published. Publishing to GHCR is a small workflow —
`docker/build-push-action` on a tag, `permissions: packages: write` — and the decision is about
what it commits you to rather than about the difficulty.

The CUDA image is about 9 GB. Publishing it means publishing a 9 GB artefact per release, and
means users pulling a torch build pinned to CUDA 12.4 whether or not it matches their driver. The
CPU image at 2 GB has no such problem and is the one to publish first if either is.

## Release checklist

```
[ ] CHANGELOG.md updated, with the user-visible changes first
[ ] version bumped in python/pyproject.toml, r/DESCRIPTION, CITATION.cff
[ ] date-released in CITATION.cff is today
[ ] uv lock re-run if any dependency changed
[ ] docs/build_docs.py run, both generated configs committed
[ ] full corpus re-run: python test/run_all.py, test/README.md tables current
[ ] both suites green in the shipping image, not just on the dev machine
[ ] git tag v<version> && git push origin v<version>
[ ] release.yaml green; both artefacts attached
[ ] docs.yaml green; the Download menu links resolve
```

The seventh line is the one that catches things. A dev machine has editable installs, a warm
cache and whatever was pip-installed last month; the image has exactly what the lock file says.
Two of the bugs fixed before v1.0 were visible only in the image.
