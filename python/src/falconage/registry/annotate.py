"""Where does a clock's weight actually sit?

A clock is a weight vector over CpGs. Two questions about that vector can be
answered without a single sample, using only the coefficients already bundled
with FALCONAge and a list of annotated sites:

**Is this clock exposed to genotype?** If a clock CpG is under strong meQTL
control, and the allele frequency at that variant differs between ancestral
populations, then a systematic difference in clock output between ancestry
groups follows from genotype frequency alone, with no difference in aging.
Cruz-Gonzalez et al. (eLife 2026, 10.7554/eLife.105343) established that the
major clocks lose accuracy in admixed individuals. What has not been measured is
whether the size of that loss tracks each clock's meQTL exposure, and the first
half of that measurement is a property of the coefficients.

**Is this clock regulatory?** Patel et al. (bioRxiv 2025.10.07.680024) report
that most CpGs used by established clocks do **not** overlap known transcription
factor binding sites. Reproducing that against any annotation class -- enhancers,
bivalent promoters, PRC2 targets, partially methylated domains, solo-WCGWs -- is
the same computation with a different list.

Why mass and not count
----------------------
Counting annotated sites answers the wrong question. A clock's behaviour is
driven by its heavy features: a 353-CpG clock with two enormous weights and 351
small ones behaves like a two-CpG clock. So the statistic is the share of
:math:`\\sum_j |w_j|` sitting on annotated sites, and the plain count is
returned beside it so the two can be compared. When they diverge sharply, the
annotated sites are the heavy ones, and that is itself the finding.

What this needs
---------------
A list of annotated feature identifiers. FALCONAge bundles no such list --
meQTL catalogues and regulatory builds are large, versioned, and separately
licensed, and freezing one into a wheel would make it silently stale. GoDMC and
the EPIGEN meQTL database are the usual sources for the first question, and the
Ensembl Regulatory Build or a Roadmap chromatin-state track for the second.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from ..core.errors import RegistryError

__all__ = ["coefficient_mass"]


def coefficient_mass(annotations: Mapping[str, Iterable[str]] | Iterable[str], *,
                     registry=None, clocks: Iterable[str] | None = None,
                     name: str = "annotated") -> pd.DataFrame:
    """Share of each clock's absolute coefficient mass on annotated features.

    Parameters
    ----------
    annotations
        Either one iterable of feature ids, or a mapping of annotation name to
        iterable, in which case every class gets its own pair of columns. A
        mapping is the useful form: the comparison between classes is what makes
        a number interpretable, since "31% of mass on enhancers" means nothing
        until it sits beside the share on a class of equal size.
    clocks
        Restrict to these clock ids. Defaults to every clock that ships a
        coefficient vector, since a clock whose weights were never traced has no
        mass to apportion and is reported in ``skipped`` rather than as zero.

    Returns
    -------
    One row per clock: ``n_features``, ``total_abs_weight``, then per annotation
    class ``<class>_n``, ``<class>_frac_sites`` and ``<class>_frac_mass``.
    ``frac_mass`` is the statistic; ``frac_sites`` is there to be compared
    against it.

    ``.attrs["skipped"]`` maps clock id to why it was left out, and
    ``.attrs["unmatched"]`` gives, per class, how many annotated ids matched no
    feature in any clock -- the number that catches an identifier mismatch, such
    as an EPIC v2 list with ``_TC21`` suffixes meeting bare ``cg`` probe ids.

    Notes
    -----
    An intercept is not a feature and is excluded wherever the coefficient file
    names it, because including it puts mass on a term no annotation can cover
    and shrinks every fraction by the same arbitrary amount.
    """
    from . import load as _load

    reg = registry if registry is not None else _load()

    if isinstance(annotations, Mapping):
        classes = {str(k): set(map(str, v)) for k, v in annotations.items()}
    else:
        classes = {name: set(map(str, annotations))}
    if not classes:
        raise RegistryError("no annotation class was supplied")
    empty = [k for k, v in classes.items() if not v]
    if empty:
        raise RegistryError(f"annotation class(es) with no features: {empty}")

    ids = list(reg.list()) if clocks is None else [str(c) for c in clocks]

    rows: dict[str, dict[str, float]] = {}
    skipped: dict[str, str] = {}
    matched: dict[str, set[str]] = {k: set() for k in classes}

    for cid in ids:
        if not reg.has_coefficient_vector(cid):
            skipped[cid] = reg.unavailable_message(cid) if not reg.has_coefficients(cid) \
                else "no plain coefficient vector (composite or neural model)"
            continue
        try:
            features, weights = reg.coefficients(cid)
        except Exception as exc:  # pragma: no cover - defensive
            skipped[cid] = f"{type(exc).__name__}: {exc}"
            continue

        feats = np.asarray([str(f) for f in features])
        w = np.abs(np.asarray(weights, dtype=np.float64).ravel())
        if feats.size != w.size:
            skipped[cid] = (f"{feats.size} feature ids against {w.size} weights; "
                            "the coefficient file is not a plain vector")
            continue

        keep = ~np.isin(np.char.lower(feats), ("intercept", "(intercept)", "_intercept"))
        feats, w = feats[keep], w[keep]
        w = np.where(np.isfinite(w), w, 0.0)
        total = float(w.sum())
        if feats.size == 0 or total <= 0:
            skipped[cid] = "no non-zero weight outside the intercept"
            continue

        row = {"n_features": int(feats.size), "total_abs_weight": total}
        for label, members in classes.items():
            hit = np.fromiter((f in members for f in feats), dtype=bool, count=feats.size)
            matched[label].update(feats[hit].tolist())
            row[f"{label}_n"] = int(hit.sum())
            row[f"{label}_frac_sites"] = float(hit.mean())
            row[f"{label}_frac_mass"] = float(w[hit].sum() / total)
        rows[cid] = row

    if not rows:
        raise RegistryError(
            "no clock in this selection ships a plain coefficient vector.\n"
            "  fa.registry.load().filter(availability='bundled') lists what "
            "carries weights; the rest are catalogued metadata only.")

    out = pd.DataFrame.from_dict(rows, orient="index")
    lead = ["n_features", "total_abs_weight"]
    out = out[lead + [c for c in out.columns if c not in lead]]
    out.index.name = "clock"
    out.attrs["skipped"] = skipped
    out.attrs["unmatched"] = {k: len(v - matched[k]) for k, v in classes.items()}
    out.attrs["class_sizes"] = {k: len(v) for k, v in classes.items()}
    return out.sort_values(f"{list(classes)[0]}_frac_mass", ascending=False)
