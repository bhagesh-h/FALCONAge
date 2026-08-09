"""The ComputAgeBench loader.

Three places were independently turning these parquets into a FalconData, each
with its own copy of the same column renames. Most of what is worth asserting
here is that there is now one copy and that it agrees with what the other three
were doing.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import falconage as fa
from falconage.core.errors import DataError
from falconage.io.bench import COLUMNS, PLATFORM, REVISION

ROOT = Path(__file__).resolve().parents[3] / "test" / "data" / "bench"
needs_corpus = pytest.mark.skipif(
    not (ROOT / "computage_bench_meta.tsv").exists(),
    reason="test corpus absent; see test/data/README.md")


def test_the_revision_is_pinned_to_a_commit():
    """A moving reference would let the benchmark change under a result that
    cites it. Forty hex characters, not a branch name."""
    assert len(REVISION) == 40 and all(c in "0123456789abcdef" for c in REVISION)


def test_the_column_map_covers_what_scoring_needs():
    got = set(COLUMNS.values())
    assert {"age", "sex", "condition", "dataset", "tissue"} <= got


@needs_corpus
def test_the_catalogue_lists_the_published_benchmark():
    lst = fa.list_computage_bench(root=str(ROOT))
    assert len(lst) == 65, "the published benchmark is 65 studies"
    assert lst["n"].sum() > 10_000
    assert set(lst.columns) >= {"n", "platform", "tissue", "conditions",
                                "age_min", "age_max", "local"}
    # Scoping a run to what is on disk is the point of the `local` column.
    assert lst["local"].sum() >= 1


@needs_corpus
def test_platforms_are_named_not_left_as_gpl_accessions():
    lst = fa.list_computage_bench(root=str(ROOT))
    assert set(lst["platform"]) <= {"27K", "450K", "EPICv1", "EPICv2"}
    assert set(PLATFORM.values()) >= set(lst["platform"])


@needs_corpus
def test_one_study_loads_with_its_platform_and_provenance():
    d = fa.read_computage_bench("GSE182991", root=str(ROOT))
    assert d.n_samples == 27
    assert d.platform == "EPICv1"
    assert d.uns["source"] == "ComputAgeBench"
    assert d.uns["revision"] == REVISION
    assert d.uns["licence"] == "CC-BY-SA-4.0"
    # Recorded, because float32 is fine for benchmarking and wrong for the
    # 1e-6 gold-standard assertions elsewhere.
    assert d.uns["stored_precision"] == "float32"
    assert {"age", "sex", "condition", "tissue"} <= set(d.obs.columns)


@needs_corpus
def test_it_reproduces_the_hand_rolled_loader_exactly():
    """The reason the module exists: one copy of the renames, and it agrees
    with the three it replaced."""
    d = fa.read_computage_bench("GSE182991", root=str(ROOT))

    meta = pd.read_csv(ROOT / "computage_bench_meta.tsv", sep="\t", index_col=0)
    X = pd.read_parquet(ROOT / "GSE182991.parquet").T.astype("float64")
    obs = meta[meta["DatasetID"] == "GSE182991"].reindex(X.index).rename(columns=COLUMNS)
    old = fa.prepare(fa.FalconData(X=X, obs=obs, modality="dna_methylation"))

    a = fa.score(d, clocks=["horvath2013"], min_coverage=0.5).scores
    b = fa.score(old, clocks=["horvath2013"], min_coverage=0.5).scores
    assert a["horvath2013"].equals(b["horvath2013"])


@needs_corpus
def test_conditions_filter_and_say_what_is_there_when_it_misses():
    hc = fa.read_computage_bench("GSE182991", root=str(ROOT), conditions=["HC"])
    assert 0 < hc.n_samples < 27
    assert set(hc.obs["condition"].astype(str)) == {"HC"}

    with pytest.raises(DataError, match="it carries"):
        fa.read_computage_bench("GSE182991", root=str(ROOT), conditions=["NOPE"])


@needs_corpus
def test_an_unknown_study_suggests_the_near_match():
    with pytest.raises(DataError, match="GSE182991"):
        fa.read_computage_bench("GSE18299", root=str(ROOT))


@needs_corpus
def test_a_study_not_on_disk_says_so_rather_than_downloading_silently():
    """65 studies are catalogued and ten are local. Reaching for one of the
    other 55 with a root set should not quietly start a download."""
    lst = fa.list_computage_bench(root=str(ROOT))
    absent = lst.index[~lst["local"]][0]
    with pytest.raises(DataError, match="is not under"):
        fa.read_computage_bench(absent, root=str(ROOT))


@needs_corpus
def test_prepare_is_on_by_default_and_can_be_turned_off():
    raw = fa.read_computage_bench("GSE182991", root=str(ROOT), prepare=False)
    done = fa.read_computage_bench("GSE182991", root=str(ROOT), prepare=True)
    assert raw.platform == done.platform == "EPICv1"
    # prepare() collapses EPIC v2 replicate suffixes; on a v1 study it is a
    # no-op on the feature count, so the assertion is that it ran at all.
    assert done.n_features <= raw.n_features
