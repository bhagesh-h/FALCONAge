"""The ComputAgeBench corpus, as a loader rather than a copied snippet.

WHY THIS IS A MODULE AND NOT AN EXAMPLE. Three places in this repository were
independently turning ComputAgeBench parquets into a :class:`FalconData` --
``test/run_all.py``, the integration tests, and two of the derivation scripts in
``python/tools/`` -- each with its own copy of the same column renames. A rename
that is right in three places and wrong in the fourth is exactly the kind of
divergence the run manifest cannot see, because every copy produces a
well-formed object.

WHAT THE CORPUS IS. Sixty-five harmonised case/control methylation studies,
10,404 samples, nineteen aging-accelerating conditions, across 27K, 450K and
EPIC. It is the reference benchmark the AA1/AA2 methodology in
:func:`falconage.run_benchmark` was defined against, so scoring against it is
how a number here becomes comparable with a published one.

    Kriukov D, Efimov E, Kuzmina EA, Khrameeva EE, Dylov DV (2024).
    ComputAgeBench: Epigenetic Aging Clocks Benchmark. bioRxiv 2024.06.06.597715.
    CC-BY-SA-4.0.

PINNED TO A COMMIT, NOT TO ``main``. The published benchmark is a fixed set of
studies; a moving reference would let the corpus change under a result that
cites it. The revision below is the one ``test/data/datasets.yaml`` fetches, so
the corpus on disk and the corpus this downloads are the same bytes.

TWO CAVEATS THAT TRAVEL WITH THE DATA. It is stored as float32, which is fine
for benchmarking and wrong for gold-standard vectors -- the 1e-6 assertions
elsewhere need full double precision. And the tissue column is ``Blood`` or
``Saliva`` at that granularity only, which is enough for the specimen check to
fire on saliva and not enough to tell buffy coat from PBMC.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Sequence

import pandas as pd

from ..core.container import FalconData
from ..core.errors import DataError

__all__ = ["computage_bench_meta", "list_computage_bench", "read_computage_bench"]

REVISION = "3cf52681537b303a5b7e21df03a3ae0a94ed497e"
BASE = f"https://huggingface.co/datasets/computage/computage_bench/resolve/{REVISION}"
META_URL = f"{BASE}/computage_bench_meta.tsv"

#: ComputAgeBench's column names against this package's conventions. The single
#: place this mapping lives.
COLUMNS = {
    "Age": "age", "Gender": "sex", "Condition": "condition",
    "DatasetID": "dataset", "PlatformID": "gpl", "Tissue": "tissue",
    "CellType": "cell_type", "Class": "class",
}

#: GEO platform accessions to the names ``detect_platform`` uses, so a loaded
#: dataset arrives with the platform already declared rather than guessed from
#: its probe count.
PLATFORM = {
    "GPL8490": "27K", "GPL13534": "450K", "GPL16304": "450K",
    "GPL21145": "EPICv1", "GPL23976": "EPICv1", "GPL29753": "EPICv2",
}


def _meta_path(root: Path | None) -> Path:
    if root is not None:
        p = Path(root) / "computage_bench_meta.tsv"
        if not p.exists():
            raise DataError(
                f"no computage_bench_meta.tsv under {root}.\n"
                "  Fetch the corpus with test/data/fetch_test_data.py, or call "
                "with root=None to download into the cache.")
        return p
    from ..download import fetch

    return fetch(META_URL)


@functools.lru_cache(maxsize=4)
def computage_bench_meta(root: str | None = None) -> pd.DataFrame:
    """The harmonised sample table for the whole benchmark.

    One row per sample, already renamed to this package's column conventions.
    Cached, because it is 10,404 rows read from the same file by every call in a
    benchmarking loop.
    """
    df = pd.read_csv(_meta_path(Path(root) if root else None), sep="\t", index_col=0)
    return df.rename(columns=COLUMNS)


def list_computage_bench(root: str | None = None) -> pd.DataFrame:
    """What is in the benchmark: one row per study.

    Use this to choose before downloading. With ``root`` set it also reports
    which studies are present locally, so a run can be scoped to what is on
    disk instead of failing partway through.
    """
    meta = computage_bench_meta(root)
    g = meta.groupby("dataset")
    out = pd.DataFrame({
        "n": g.size(),
        "platform": g["gpl"].first().map(lambda x: PLATFORM.get(x, x)),
        "tissue": g["tissue"].agg(lambda s: ", ".join(sorted(set(s.astype(str))))),
        "conditions": g["condition"].agg(
            lambda s: ", ".join(f"{k}={v}" for k, v in
                                sorted(s.astype(str).value_counts().items()))),
        "age_min": g["age"].min().round(1),
        "age_max": g["age"].max().round(1),
    })
    if root:
        here = Path(root)
        out["local"] = [(here / f"{d}.parquet").exists()
                        or (here / f"computage_bench_data_{d}.parquet").exists()
                        for d in out.index]
    return out.sort_values("n", ascending=False)


def _data_path(dataset: str, root: Path | None) -> Path:
    if root is not None:
        for name in (f"{dataset}.parquet", f"computage_bench_data_{dataset}.parquet"):
            p = Path(root) / name
            if p.exists():
                return p
        raise DataError(
            f"{dataset} is not under {root}.\n"
            "  list_computage_bench(root) shows which studies are present "
            "locally; call with root=None to download this one into the cache.")
    from ..download import fetch

    return fetch(f"{BASE}/data/benchmark/computage_bench_data_{dataset}.parquet")


def read_computage_bench(dataset: str, *, root: str | None = None,
                         conditions: Sequence[str] | None = None,
                         prepare: bool = True) -> FalconData:
    """One ComputAgeBench study, ready to score.

    Parameters
    ----------
    dataset
        A GEO series accession, e.g. ``"GSE182991"``. See
        :func:`list_computage_bench`.
    root
        A local directory holding the parquets and the metadata TSV. ``None``
        downloads into the ordinary FALCONAge cache.
    conditions
        Keep only these values of ``condition``. The control label in this
        corpus is ``"HC"``.
    prepare
        Run :func:`falconage.prepare`, which collapses EPIC v2 replicate
        suffixes and declares the platform. Left on by default because a v2
        matrix without that step overlaps almost nothing.

    Notes
    -----
    Stored float32 is widened to float64 here. Not because it recovers the lost
    bits -- it does not -- but because everything downstream computes in double
    and a silent upcast partway through a chain is harder to reason about than
    one at the door.
    """
    root_p = Path(root) if root else None
    meta = computage_bench_meta(root)
    if dataset not in set(meta["dataset"]):
        near = sorted(d for d in set(meta["dataset"]) if dataset.upper() in str(d).upper())
        raise DataError(
            f"{dataset!r} is not in ComputAgeBench."
            + (f" Did you mean {near[:4]}?" if near else
               " list_computage_bench() shows all 65."))

    X = pd.read_parquet(_data_path(dataset, root_p)).T.astype("float64")
    obs = meta[meta["dataset"] == dataset].reindex(X.index)
    if conditions is not None:
        keep = obs["condition"].astype(str).isin([str(c) for c in conditions])
        if not keep.any():
            have = sorted(set(obs["condition"].astype(str)))
            raise DataError(
                f"{dataset} has no sample with condition in {list(conditions)}; "
                f"it carries {have}")
        X, obs = X[keep.to_numpy()], obs[keep.to_numpy()]

    gpl = str(obs["gpl"].iloc[0]) if "gpl" in obs.columns and len(obs) else ""
    d = FalconData(X=X, obs=obs, modality="dna_methylation",
                   platform=PLATFORM.get(gpl),
                   uns={"source": "ComputAgeBench", "revision": REVISION,
                        "dataset": dataset, "gpl": gpl,
                        "licence": "CC-BY-SA-4.0",
                        "stored_precision": "float32"})
    if prepare:
        from ..preprocess import prepare as _prepare

        d = _prepare(d)
    return d
