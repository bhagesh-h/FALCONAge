"""Repertoire structure: the covariate blood clocks do not currently carry.

Deconvolution answers *how many cells of each type*. It says nothing about *how
many distinct clones those cells represent*, and the two are different
measurements. Two donors with an identical CD8 memory fraction can differ by
orders of magnitude in clone count.

Why that should move a clock at all
-----------------------------------
Take one cell type holding a fraction :math:`f` of the sample, split into clones
of relative size :math:`w_k` with :math:`\\sum_k w_k = 1`. Give clone :math:`k`
a somatic offset :math:`\\delta_k` from its lineage mean at some site, with
:math:`\\mathbb{E}[\\delta_k] = 0` and :math:`\\mathrm{Var}[\\delta_k]
= \\sigma^2`. The compartment's bulk beta is the size-weighted mean, so

.. math::

    \\mathrm{Var}\\Big[\\sum_k w_k \\delta_k\\Big]
      = \\sigma^2 \\sum_k w_k^2
      = \\frac{\\sigma^2}{N_{\\text{eff}}},
    \\qquad N_{\\text{eff}} = \\Big(\\sum_k w_k^2\\Big)^{-1}

and :math:`\\sum_k w_k^2` is exactly the Simpson index. So the variance a clonal
compartment contributes to bulk methylation is inversely proportional to its
**effective number of clones**, and clone *count* enters through that quantity
and no other. A pool of three equal clones carries the somatic state of three
cells at full weight; ten thousand clones average it to nothing.

This is why the effect is invisible to composition adjustment: :math:`f` can be
identical between two donors while :math:`N_{\\text{eff}}` differs
thousand-fold, and every published blood clock sees only :math:`f`.

The lineage-restricted version of this question is settled. Nachun et al.
(Aging Cell 20:e13366, 2021) measured it for myeloid clonal hematopoiesis: 1.31
years of acceleration on GrimAge to 3.08 on extrinsic EEAA. The lymphoid
analogue has not been asked.

What is here, and what it needs
-------------------------------
:func:`repertoire_diversity` takes a clone table -- the standard output of
TCRB or BCR sequencing -- and returns the structure metrics per sample, ready
to hand to ``fa.acceleration(result, adjust=[...])``, which already accepts
arbitrary ``obs`` columns.

:func:`simulate_clonality` is the mechanism arm, and it does **not** need paired
data. It needs a sorted-cell methylation reference, features x cell types. That
matrix is not bundled: every deconvolution model in the FALCONAge registry is
catalogued as ``untraced``, carrying metadata but no coefficients, so the
reference has to be supplied. ``FlowSorted.Blood.EPIC`` on Bioconductor is the
usual source.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ..core.container import FalconData
from ..core.errors import AnalysisError, DataError

__all__ = ["effective_clones", "repertoire_diversity", "simulate_clonality",
           "zipf_clone_sizes"]


def effective_clones(sizes) -> float:
    """Inverse Simpson index: the number of equal-sized clones that would
    produce the same averaging as this distribution.

    The quantity the derivation above singles out, and the one to use when a
    single number has to stand for "how clonal". Unlike observed richness it
    barely moves with sequencing depth, because a rare clone contributes
    :math:`w_k^2` and a rare clone's :math:`w_k^2` is negligible.
    """
    w = np.asarray(sizes, dtype=np.float64)
    w = w[np.isfinite(w) & (w > 0)]
    if w.size == 0:
        return np.nan
    w = w / w.sum()
    return float(1.0 / np.square(w).sum())


def zipf_clone_sizes(n_clones: int, alpha: float = 0.0) -> np.ndarray:
    """Clone sizes following :math:`w_k \\propto k^{-\\alpha}`, normalised.

    A convenience for :func:`simulate_clonality`, not a claim about real
    repertoires. ``alpha=0`` is uniform, which is the maximally diverse case;
    raising alpha concentrates the repertoire into its largest clones. Around
    ``alpha=1`` the distribution is Zipf, which is the shape most often fitted
    to observed TCR data, and the useful property here is that the effective
    clone count falls far faster than the nominal one.
    """
    if n_clones < 1:
        raise AnalysisError(f"n_clones must be at least 1, got {n_clones}")
    k = np.arange(1, int(n_clones) + 1, dtype=np.float64)
    w = k ** (-float(alpha))
    return w / w.sum()


def repertoire_diversity(clones: pd.DataFrame, *, sample_col: str = "sample_id",
                         count_col: str = "count", top_n: int = 100,
                         rarefy: int | str | None = None,
                         seed: int = 0) -> pd.DataFrame:
    """Structure metrics per sample, from a clone table.

    Parameters
    ----------
    clones
        One row per clone per sample. ``sample_col`` identifies the donor and
        ``count_col`` holds that clone's read or template count. Rows with a
        non-positive count are dropped; rows are summed if a clonotype appears
        more than once for one sample.
    top_n
        The share of the repertoire held by the largest ``top_n`` clones, which
        is the metric most often reported and the one least comparable between
        studies, since it depends on how many clones were sampled at all.
    rarefy
        Subsample every donor to a common depth before computing anything.
        ``"min"`` uses the smallest depth present. This matters more than any
        other choice here: richness and Shannon both rise with depth, so an
        unrarefied comparison between a deeply and a shallowly sequenced donor
        measures the library, not the donor. ``clonality``, ``simpson`` and
        ``effective_clones`` are far more stable and are the ones to trust if
        rarefaction is not possible.

    Returns
    -------
    Per sample: ``n_reads``, ``richness``, ``shannon``, ``evenness``,
    ``clonality``, ``simpson``, ``effective_clones``, ``top{n}_share``,
    ``chao1``. Indexed by sample id, so it drops straight into ``obs``::

        div = fa.immune.repertoire_diversity(clones)
        data.obs = data.obs.join(div)
        acc = fa.acceleration(res, adjust=["clonality", "cd8mem_fraction"])
    """
    for col in (sample_col, count_col):
        if col not in clones.columns:
            raise DataError(f"no {col!r} column in the clone table")

    counts = pd.to_numeric(clones[count_col], errors="coerce")
    tab = pd.DataFrame({"sample": clones[sample_col].astype(str), "n": counts})
    tab = tab[np.isfinite(tab["n"]) & (tab["n"] > 0)]
    if tab.empty:
        raise DataError(
            f"no clone in the table has a positive {count_col!r}")

    depth = tab.groupby("sample")["n"].sum()
    target: int | None = None
    if rarefy is not None:
        target = int(depth.min()) if rarefy == "min" else int(rarefy)
        if target < 1:
            raise AnalysisError(f"rarefy depth must be positive, got {target}")
        too_shallow = depth[depth < target]
        if len(too_shallow):
            raise AnalysisError(
                f"{len(too_shallow)} sample(s) are shallower than the "
                f"rarefaction depth {target:,} (smallest is {int(depth.min()):,}).\n"
                "  Rarefying up is not defined. Use rarefy='min', lower the "
                "target, or drop those samples.")

    rng = np.random.default_rng(seed)
    rows = {}
    for name, grp in tab.groupby("sample", sort=True):
        w = grp["n"].to_numpy(dtype=np.float64)

        if target is not None:
            # Multivariate hypergeometric: sampling `target` reads without
            # replacement, which is what a shallower run would actually have
            # produced. Sampling *with* replacement inflates richness at low
            # depth, in the same direction as the bias being corrected.
            w = _hypergeometric(w, target, rng).astype(np.float64)
            w = w[w > 0]

        n_reads = float(w.sum())
        p = w / n_reads
        richness = int(w.size)

        shannon = float(-(p * np.log(p)).sum())
        evenness = shannon / np.log(richness) if richness > 1 else np.nan
        simpson = float(np.square(p).sum())
        order = np.sort(p)[::-1]

        rows[name] = {
            "n_reads": n_reads,
            "richness": richness,
            "shannon": shannon,
            "evenness": evenness,
            # 1 - Pielou evenness. Zero for a perfectly even repertoire, one for
            # a repertoire that is a single clone. Undefined for richness 1,
            # where evenness has no denominator, rather than silently 0 or 1.
            "clonality": (1.0 - evenness) if np.isfinite(evenness) else np.nan,
            "simpson": simpson,
            "effective_clones": float(1.0 / simpson),
            f"top{top_n}_share": float(order[:top_n].sum()),
            "chao1": _chao1(w),
        }

    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = sample_col
    out.attrs["rarefied_to"] = target
    return out


def _hypergeometric(counts: np.ndarray, draws: int, rng) -> np.ndarray:
    """Multivariate hypergeometric sample, by sequential binomials.

    numpy has no direct multivariate hypergeometric for large populations, and
    the sequential-binomial construction is exact: draw clone 1's share from a
    hypergeometric against the rest, then recurse on what is left.
    """
    counts = counts.astype(np.int64)
    total = int(counts.sum())
    out = np.zeros_like(counts)
    remaining_draws = int(draws)
    for i in range(counts.size):
        if remaining_draws <= 0:
            break
        good = int(counts[i])
        bad = total - good
        taken = int(rng.hypergeometric(good, bad, remaining_draws)) if bad > 0 else remaining_draws
        out[i] = taken
        remaining_draws -= taken
        total -= good
    return out


def _chao1(counts: np.ndarray) -> float:
    """Chao1 richness estimate: observed richness plus a term for what was missed.

    ``S_obs + f1(f1-1) / (2(f2+1))``, the bias-corrected form, which is defined
    when ``f2 = 0`` where the classic ``f1^2 / 2 f2`` is not. Counts that are not
    integers (normalised or UMI-collapsed tables) make the singleton and doubleton
    counts meaningless, so the estimate is refused rather than computed on
    rounded values.
    """
    if not np.allclose(counts, np.round(counts)):
        return np.nan
    c = np.round(counts).astype(np.int64)
    f1 = int((c == 1).sum())
    f2 = int((c == 2).sum())
    return float(c.size + f1 * (f1 - 1) / (2.0 * (f2 + 1)))


def simulate_clonality(reference: pd.DataFrame,
                       fractions: Mapping[str, float] | pd.Series, *,
                       clonal_types: Sequence[str],
                       clone_sizes: Sequence[Sequence[float]],
                       sigma: float = 0.05,
                       n_replicates: int = 1,
                       seed: int = 0,
                       age: float | None = None,
                       platform: str | None = None) -> FalconData:
    """Synthetic bulk methylomes at **fixed** cell fractions and varying clonality.

    The mechanism arm of the repertoire question, and the reason it does not
    need a paired cohort: cell-type fractions are held identical across every
    simulated sample, so any movement in a clock score is attributable to
    clone structure alone. Score the result and regress:

        sim = fa.immune.simulate_clonality(ref, fracs,
                                           clonal_types=["CD8mem"],
                                           clone_sizes=[zipf_clone_sizes(n, 1.0)
                                                        for n in (3, 30, 300, 3000)],
                                           n_replicates=20)
        res = fa.score(sim, clocks="compatible")
        acc = fa.acceleration(res)          # against sim.obs["effective_clones"]

    A clock that is genuinely independent of clone structure returns a flat line
    against ``effective_clones``. The prediction from the derivation in this
    module's docstring is a spread that falls as
    :math:`1/\\sqrt{N_{\\text{eff}}}`, so plot it on a log axis.

    Parameters
    ----------
    reference
        Sorted-cell mean beta, features x cell types. Not bundled with
        FALCONAge; see this module's docstring.
    fractions
        Cell type to proportion. Must name columns of ``reference``, and is
        renormalised to sum to one with a warning-free adjustment only if it is
        already within 1e-6, since a fraction vector that does not sum to one is
        more likely a mistake than a rounding artefact.
    clonal_types
        Which cell types carry clone structure. The rest are treated as
        effectively infinite pools, which is the right model for neutrophils and
        the wrong one for memory T cells, and naming them explicitly is how that
        assumption stays visible.
    sigma
        Per-site standard deviation of one clone's somatic departure from its
        lineage mean, in beta units. 0.05 is a deliberately modest default. This
        is the parameter the whole result scales with and it is not well
        measured in the literature, so treat the output as a sensitivity
        analysis over sigma rather than as a calibrated prediction.

    Returns
    -------
    A :class:`~falconage.core.FalconData` of simulated samples. ``obs`` carries
    ``effective_clones``, ``n_clones``, ``simpson``, ``sigma`` and the fixed
    fractions, so the covariate the analysis needs is already attached.
    """
    if reference.empty:
        raise DataError("the sorted-cell reference is empty")
    frac = pd.Series(fractions, dtype=np.float64)
    missing = [c for c in frac.index if c not in reference.columns]
    if missing:
        raise DataError(
            f"fractions name cell types absent from the reference: {missing}\n"
            f"  reference has: {list(reference.columns)}")
    if not np.isfinite(frac.to_numpy()).all() or (frac < 0).any():
        raise DataError("every cell fraction must be finite and non-negative")
    total = float(frac.sum())
    if abs(total - 1.0) > 1e-6:
        raise DataError(
            f"cell fractions sum to {total:.6g}, not 1. Renormalising silently "
            "would hide a mislabelled column, so fix the vector instead.")
    frac = frac / total

    unknown = [c for c in clonal_types if c not in frac.index]
    if unknown:
        raise DataError(f"clonal_types not among the fractions: {unknown}")
    if not clone_sizes:
        raise AnalysisError("clone_sizes is empty; nothing to simulate")
    if sigma < 0:
        raise AnalysisError(f"sigma must be non-negative, got {sigma}")

    ref = reference.reindex(columns=frac.index).astype(np.float64)
    if ref.isna().all(axis=None):
        raise DataError("the reference has no usable values for these cell types")
    n_feat = ref.shape[0]
    rng = np.random.default_rng(seed)

    # The fixed part: the mixture every simulated sample shares. Computed once,
    # because it is precisely what must not vary between conditions.
    base = (ref.to_numpy() * frac.to_numpy()).sum(axis=1)

    rows, meta = [], []
    for c_idx, sizes in enumerate(clone_sizes):
        w = np.asarray(sizes, dtype=np.float64)
        w = w[np.isfinite(w) & (w > 0)]
        if w.size == 0:
            raise AnalysisError(f"clone_sizes[{c_idx}] has no positive entries")
        w = w / w.sum()
        n_eff = effective_clones(w)

        for rep in range(int(n_replicates)):
            beta = base.copy()
            for ct in clonal_types:
                f = float(frac[ct])
                if f <= 0 or sigma == 0:
                    continue
                # The compartment offset at site i is sum_k w_k d_ki with the
                # d_ki independent N(0, sigma^2). That sum is exactly
                # N(0, sigma^2 * sum_k w_k^2), and the sums at different sites
                # are independent because their d are. So one draw per site from
                # that variance is the identical distribution to materialising
                # the whole (clones x sites) matrix -- not an approximation of
                # it. The full matrix would be 2,000 x 800,000 doubles for a
                # realistic repertoire on an EPIC panel, which is 13 GB and the
                # reason this is written the short way.
                spread = sigma * np.sqrt(float(np.square(w).sum()))
                beta = beta + f * rng.normal(0.0, spread, size=n_feat)

            rows.append(np.clip(beta, 0.0, 1.0))
            meta.append({
                "n_clones": int(w.size),
                "effective_clones": n_eff,
                "simpson": float(np.square(w).sum()),
                "sigma": float(sigma),
                "replicate": int(rep),
                "clone_set": int(c_idx),
            })

    ids = [f"sim{i:04d}" for i in range(len(rows))]
    X = pd.DataFrame(np.vstack(rows), index=ids, columns=ref.index)
    obs = pd.DataFrame(meta, index=ids)
    for ct, f in frac.items():
        obs[f"frac_{ct}"] = float(f)
    obs["clonal_types"] = ",".join(clonal_types)
    if age is not None:
        obs["age"] = float(age)

    return FalconData(
        X=X, obs=obs, modality="dna_methylation", platform=platform,
        uns={"falconage_simulation": {
            "kind": "clonality",
            "sigma": float(sigma),
            "fractions": {k: float(v) for k, v in frac.items()},
            "clonal_types": list(clonal_types),
            "seed": int(seed),
            "note": ("Simulated. Cell-type fractions are identical across "
                     "every sample by construction, so any variation in a "
                     "clock score is clone structure and nothing else."),
        }})
