"""Bulk transcriptomes onto the scale transcriptomic clocks were fitted on.

THE CHAIN, IN THE ORDER IT MUST RUN

    counts
      -> keep genes the model names          (an orthologue map, across species)
      -> RLE size-factor normalisation       (library depth)
      -> log10(x + 1)
      -> per-sample z                        (between-sample scale)
      -> YuGene cumulative transform         (platform)
      -> align to the model's gene list, padding absent genes
      -> median-centre each gene ACROSS THE DATASET

WHY THE LAST STEP DECIDES EVERYTHING ABOUT THE API. Centring each gene on its
median across the samples you supplied makes every score a function of the whole
batch. Run one sample alone and its median is itself: every gene becomes zero
and the model returns its intercept -- the same confident number for anybody.
Nothing in the arithmetic can notice.

That is exactly the property ``Clock.requires_cohort`` was added for, and it is
why transcriptomic clocks carry it. :func:`falconage.score` refuses them below
``min_samples`` rather than warning, because a warning beside a plausible number
reads as a caveat on a result rather than notice that there is no result.

WHY RLE RATHER THAN TPM OR CPM. RLE -- the median-of-ratios size factor from
DESeq -- is what the published transcriptomic clocks were fitted under. TPM
divides by gene length and CPM by total counts, and both give different numbers
for the same library. A clock fitted on one and scored on another is being asked
about a different quantity at every gene.

Reference: the tAge pipeline (Meyer & Schumacher and successors), summarised in
docs/science.qmd.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.container import FalconData
from ..core.errors import DataError

__all__ = ["median_centre", "prepare_transcriptomic", "read_counts",
           "rle_normalise", "yugene"]


def read_counts(path, *, index_col: int | str = 0, genes_in_rows: bool = True,
                obs: pd.DataFrame | None = None) -> FalconData:
    """A bulk RNA-seq count matrix.

    ``genes_in_rows`` because that is how every quantifier writes one and how
    GEO distributes them, while everything downstream here is samples x
    features. Transposed on the way in rather than left to the caller: a matrix
    the wrong way round produces a clock score for each gene, which is a
    plausible-looking table of nonsense.
    """
    from pathlib import Path

    p = Path(path)
    sep = "\t" if p.suffix.lower() in (".tsv", ".txt", ".gz") else ","
    df = pd.read_csv(p, sep=sep, index_col=index_col)
    df = df.apply(pd.to_numeric, errors="coerce")
    if genes_in_rows:
        df = df.T
    if df.empty:
        raise DataError(f"{path}: no numeric values")
    if float(np.nanmin(df.to_numpy())) < 0:
        raise DataError(f"{path}: negative values; counts cannot be negative. "
                        "This looks like an already-transformed matrix.")
    return FalconData(X=df, obs=obs if obs is not None else pd.DataFrame(index=df.index),
                      modality="transcriptomics",
                      uns={"input": "counts", "genes_in_rows": genes_in_rows})


def rle_normalise(X: pd.DataFrame) -> pd.DataFrame:
    """Median-of-ratios size factors (DESeq's RLE).

    Each sample is divided by the median, over genes, of its ratio to the
    geometric mean across samples. Genes with a zero anywhere drop out of the
    size-factor calculation -- their geometric mean is zero and the ratio is
    undefined -- which is the standard treatment and the reason RLE is robust to
    a few very highly expressed genes that would dominate a total-count scaling.
    """
    v = X.to_numpy(dtype=np.float64)
    # A gene that is zero in every sample has no finite log anywhere, and numpy
    # says so once per such column. On a filtered matrix that is thousands of
    # identical lines; the `usable` mask below is what handles the case.
    import warnings as _w

    with np.errstate(divide="ignore", invalid="ignore"), _w.catch_warnings():
        _w.simplefilter("ignore", RuntimeWarning)
        logs = np.log(v)
        gmean = np.exp(np.nanmean(np.where(np.isfinite(logs), logs, np.nan), axis=0))
    usable = np.isfinite(gmean) & (gmean > 0)
    if usable.sum() < 10:
        raise DataError(
            "fewer than ten genes are non-zero in every sample, so there is no "
            "stable reference for RLE size factors. Filter the matrix first.")
    ratios = v[:, usable] / gmean[usable]
    size = np.nanmedian(ratios, axis=1)
    size = np.where(np.isfinite(size) & (size > 0), size, 1.0)
    return pd.DataFrame(v / size[:, None], index=X.index, columns=X.columns)


def yugene(X: pd.DataFrame) -> pd.DataFrame:
    """The YuGene cumulative-proportion transform, per sample.

    For each sample, a gene's value becomes one minus the cumulative share of
    that sample's total expression held by genes ranked above it. It is
    rank-based, so it is invariant to any monotone rescaling of a sample, which
    is what makes values comparable across platforms that quantify differently.
    """
    v = np.nan_to_num(X.to_numpy(dtype=np.float64), nan=0.0)
    v = np.clip(v, 0.0, None)
    out = np.empty_like(v)
    for i in range(v.shape[0]):
        row = v[i]
        total = row.sum()
        if total <= 0:
            out[i] = 0.0
            continue
        order = np.argsort(-row, kind="stable")
        cum = np.cumsum(row[order]) / total
        res = np.empty_like(row)
        res[order] = 1.0 - cum
        out[i] = res
    return pd.DataFrame(out, index=X.index, columns=X.columns)


def median_centre(X: pd.DataFrame) -> pd.DataFrame:
    """Subtract each gene's median across the samples given.

    The step that makes a transcriptomic score a property of the batch. See the
    module docstring; :attr:`falconage.registry.Clock.requires_cohort` is what
    stops it being applied to one sample.
    """
    if X.shape[0] < 2:
        raise DataError(
            "median-centring one sample centres it against itself: every gene "
            "becomes zero and the model returns its intercept.\n"
            "  Score the whole cohort in one call.")
    return X.sub(X.median(axis=0), axis=1)


def prepare_transcriptomic(data: FalconData, *,
                           orthologues: dict[str, str] | pd.Series | None = None,
                           genes: list[str] | None = None,
                           centre: bool = True) -> FalconData:
    """Run the whole chain, in order, recording each step.

    Parameters
    ----------
    orthologues
        Source gene id to the model's gene id, for scoring one species with a
        model fitted in another. Applied first, and one-to-one only: a
        many-to-one map would silently sum paralogues into one column, and which
        paralogue a clock's coefficient refers to is not recoverable afterwards.
    genes
        The model's gene list. Absent genes are padded with NaN rather than
        dropped, so coverage reporting downstream sees them missing instead of
        never knowing they were expected.
    centre
        The per-dataset median centring. ``False`` leaves it out, which is only
        right if you are going to do it yourself over the correct cohort.
    """
    if data.modality != "transcriptomics":
        raise DataError(f"prepare_transcriptomic on a {data.modality} dataset")

    X = data.X
    steps: list[str] = []

    if orthologues is not None:
        m = (pd.Series(orthologues) if not isinstance(orthologues, pd.Series)
             else orthologues)
        dup = m[m.duplicated(keep=False)]
        if len(dup):
            raise DataError(
                f"the orthologue map is not one-to-one: {len(dup)} source genes "
                f"share a target (e.g. {list(dup.index[:4])}).\n"
                "  Summing paralogues into one column loses which one a "
                "coefficient referred to, and nothing downstream can recover it.")
        keep = [c for c in X.columns if c in m.index]
        X = X[keep].rename(columns=m.to_dict())
        steps.append(f"orthologues:{len(keep)}")

    X = rle_normalise(X)
    steps.append("rle")

    X = np.log10(X + 1.0)
    steps.append("log10(x+1)")

    sd = X.std(axis=1, ddof=1).replace(0, np.nan)
    X = X.sub(X.mean(axis=1), axis=0).div(sd, axis=0)
    steps.append("z_per_sample")

    X = yugene(X - X.min(axis=1).min())      # YuGene needs non-negative input
    steps.append("yugene")

    if genes is not None:
        X = X.reindex(columns=list(genes))
        steps.append(f"align:{len(genes)}")

    if centre:
        X = median_centre(X)
        steps.append("median_centre")

    out = FalconData(X=X, obs=data.obs, modality="transcriptomics",
                     platform=data.platform, uns=dict(data.uns))
    out.uns["transcriptomic_pipeline"] = {
        "steps": steps, "n_samples": int(X.shape[0]), "n_genes": int(X.shape[1]),
        "centred_over": ("this call's samples" if centre else "not centred"),
    }
    return out
