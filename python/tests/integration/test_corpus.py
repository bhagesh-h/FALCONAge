"""Every feature exercised against the real public corpus in ``test/data``.

One test per group in ``datasets.yaml``, so a failure names the data path that
broke rather than "integration". Skipped wholesale when the corpus is absent.
"""

from __future__ import annotations


import numpy as np
import pandas as pd
import pytest

import falconage as fa

pytestmark = pytest.mark.corpus


def _bench(corpus, gse):
    meta = pd.read_csv(corpus / "bench" / "computage_bench_meta.tsv", sep="\t", index_col=0)
    X = pd.read_parquet(corpus / "bench" / f"{gse}.parquet").T.astype("float64")
    obs = meta[meta["DatasetID"] == gse].reindex(X.index).rename(columns={
        "Age": "age", "Gender": "sex", "Condition": "condition",
        "DatasetID": "dataset", "PlatformID": "gpl", "Tissue": "tissue"})
    return fa.prepare(fa.FalconData(X=X, obs=obs, modality="dna_methylation"))


# ---------------------------------------------------------------------------
# bench: EPIC, 450K, 27K
# ---------------------------------------------------------------------------
def test_epic_progeria_scores_every_tier_a_clock(corpus):
    d = _bench(corpus, "GSE182991")
    assert d.n_samples == 27
    assert d.platform == "EPICv1"

    res = fa.score(d, clocks="compatible")
    assert res.scores.shape[1] >= 18
    assert res.scores.notna().all().all()

    # Horvath on blood spanning 0-41 should land in a human range, not a
    # transform-gone-wrong range.
    h = res.scores["horvath2013"]
    assert 0 < h.min() and h.max() < 120
    assert h.corr(pd.to_numeric(d.obs["age"])) > 0.5


def test_progeria_is_the_known_negative_result(corpus):
    """Every clock fails to separate HGPS from age-matched healthy. That is the
    published finding, not a defect, and it is why the corpus carries it."""
    d = _bench(corpus, "GSE182991")
    res = fa.score(d, clocks=["horvath2013", "dnamphenoage", "skinandblood", "hannum"])
    b = fa.run_benchmark(res, condition_col="condition", control="HC",
                         dataset_col="dataset")
    assert b.summary()["AA2"].sum() == 0


def test_450k_ankylosing_spondylitis_does_accelerate(corpus):
    """The other end: a condition where the clocks agree, which is what keeps
    the progeria result above from looking like a broken pipeline."""
    d = _bench(corpus, "GSE107143")
    res = fa.score(d, clocks=["horvath2013", "dnamphenoage", "hannum"])
    b = fa.run_benchmark(res, condition_col="condition", control="HC",
                         dataset_col="dataset")
    assert b.summary()["AA2"].sum() >= 2
    assert (b.per_dataset["delta"] > 0).all()


def test_27k_loses_most_features_and_says_so(corpus):
    """27K carries 27,578 probes against 450K's 485,512, so most modern clocks
    should be reported as under-covered rather than scored on imputed values."""
    d = _bench(corpus, "GSE56606")
    assert d.n_samples == 90
    res = fa.score(d, clocks="compatible")
    assert res.scores.shape[1] <= 8, "a 27K array cannot support the whole catalogue"
    assert any("coverage" in v or "floor" in v for v in res.skipped.values())


def test_multi_dataset_benchmark(corpus):
    """The real workflow: score each study separately, combine, benchmark."""
    results = []
    for gse in ("GSE107143", "GSE71841", "GSE130030", "GSE182991"):
        d = _bench(corpus, gse)
        results.append(fa.score(d, clocks=["horvath2013", "dnamphenoage", "hannum",
                                           "skinandblood"]))
        del d
    res = fa.combine(results)
    assert res.scores.shape[0] == 95

    b = fa.run_benchmark(res, condition_col="condition", control="HC",
                         dataset_col="dataset")
    assert len(b.per_dataset) == 16
    assert b.summary()["total"].max() > 0
    assert (b.per_dataset["q"] >= b.per_dataset["p"] - 1e-12).all()


# ---------------------------------------------------------------------------
# epicv2: suffix aggregation
# ---------------------------------------------------------------------------
def test_epicv2_suffixes_are_aggregated(corpus):
    raw = fa.read_series_matrix(corpus / "epicv2" / "GSE330325_series_matrix.txt.gz")
    assert raw.platform == "EPICv2"
    assert any("_" in str(c) for c in raw.features[:5000]), "v2 probes carry suffixes"

    # Before aggregation a clock sees nothing; after it, most of its features.
    before = raw.coverage(list(fa.registry.load().feature_ids("horvath2013")))
    prepared = fa.prepare(raw)
    after = prepared.coverage(list(fa.registry.load().feature_ids("horvath2013")))
    assert before < 0.05, "unaggregated EPIC v2 overlaps almost nothing"
    assert after > 0.85, "aggregation is what makes the clocks work on v2"
    assert prepared.n_features < raw.n_features

    res = fa.score(prepared, clocks=["horvath2013", "hannum"])
    assert res.scores.notna().all().all()


# ---------------------------------------------------------------------------
# gestational
# ---------------------------------------------------------------------------
def test_cord_blood_gestational_clocks(corpus):
    d = fa.prepare(fa.read_series_matrix(
        corpus / "gestational" / "GSE66459_series_matrix.txt.gz"))
    assert d.n_samples == 22
    assert d.platform == "450K"
    assert "gestational_age_days" in d.obs.columns

    res = fa.score(d, clocks=["knight", "leecontrol", "leerobust", "leerefinedrobust"])
    # Predictions are in weeks and should sit in a term-ish range.
    for cid in res.scores.columns:
        assert 25 < res.scores[cid].median() < 50, cid

    # And they should track the recorded gestational age once it is in weeks --
    # the conversion the units module refuses to do silently.
    weeks = pd.to_numeric(d.obs["gestational_age_days"], errors="coerce") / 7.0
    assert res.scores["knight"].corr(weeks) > 0.3


def test_gestational_days_are_not_silently_read_as_weeks(corpus):
    """The metadata keeps its published name and number; nothing converts it."""
    d = fa.read_series_matrix(corpus / "gestational" / "GSE66459_series_matrix.txt.gz")
    days = pd.to_numeric(d.obs["gestational_age_days"], errors="coerce")
    assert days.median() > 150, "still days, as GEO published them"


# ---------------------------------------------------------------------------
# mammalian
# ---------------------------------------------------------------------------
def test_mammalian_array_covers_human_clocks_but_that_is_not_validity(corpus):
    """The finding that motivates the species check.

    The mammalian array was designed on CpGs conserved across mammals, and 96%
    of Horvath2013's 353 probes are among them. So a zebra scores at higher
    coverage than many human 450K datasets and returns a confident number from
    a clock fitted on people. Nothing in the arithmetic can notice; only the
    declared species can, and the warning is the only thing standing between
    this and a plausible-looking table of zebra Horvath ages.
    """
    d = fa.read_betas(corpus / "mammalian" / "GSE184222_datBetaNormalized.csv.gz")
    assert 30_000 < d.n_features < 45_000
    assert d.n_samples == 12
    assert d.platform == "MammalMethylChip40"

    reg = fa.registry.load()
    assert d.coverage(list(reg.feature_ids("horvath2013"))) > 0.9

    d.species = "Equus grevyi"
    res = fa.score(d, clocks=["horvath2013"])
    species_warnings = [w for w in res.manifest.warnings if w["category"] == "species"]
    assert species_warnings, "a human clock on a zebra must say so"
    assert "Homo sapiens" in species_warnings[0]["message"]

    # Left as human it says nothing, which is why the field has to be set
    # rather than inferred -- there is no signal in a beta matrix that says
    # which animal it came from.
    d.species = "Homo sapiens"
    quiet = fa.score(d, clocks=["horvath2013"])
    assert not [w for w in quiet.manifest.warnings if w["category"] == "species"]


def test_mammalian_metadata_is_a_separate_file(corpus):
    """These series publish betas as a supplementary CSV and leave the series
    matrix as a header -- the pair is the fixture for that case."""
    import gzip

    text = gzip.open(corpus / "mammalian" / "GSE184222_series_matrix.txt.gz",
                     "rt", errors="replace").read()
    assert "!Sample_characteristics_ch1" in text
    assert "age:" in text
    assert "!series_matrix_table_begin" not in text or text.count("\ncg") < 10


# ---------------------------------------------------------------------------
# mouse RRBS
# ---------------------------------------------------------------------------
def test_rrbs_reads_coordinate_keyed_sites(corpus):
    paths = sorted((corpus / "mouse").glob("GSM*.overlap.txt.gz"))
    assert len(paths) == 4
    d = fa.read_rrbs_dir(paths, min_coverage=5)
    assert d.modality == "rrbs"
    assert d.n_samples == 4
    assert d.n_features > 10_000
    assert all(":" in str(f) for f in list(d.features)[:20]), "chr:pos keys"
    v = d.X.to_numpy()
    assert np.nanmin(v) >= 0.0 and np.nanmax(v) <= 1.0


def test_rrbs_coverage_filter_actually_filters(corpus):
    p = sorted((corpus / "mouse").glob("GSM*.overlap.txt.gz"))[0]
    lo = fa.io.read_rrbs(p, min_coverage=1)
    hi = fa.io.read_rrbs(p, min_coverage=20)
    assert len(hi) < len(lo), "a site read once is not the same measurement as one read 400 times"


def test_mouse_ages_are_months_labelled_years(corpus):
    """The corpus's best argument for declaring units. A mouse does not live 35
    years, and GEO says 'age (years): 35'."""
    d = fa.read_series_matrix(corpus / "mouse" / "GSE80672_series_matrix.txt.gz")
    ages = pd.to_numeric(d.obs["age_years"], errors="coerce")
    assert ages.max() > 30, "the published field really does say 35"
    from falconage.core.units import convert

    assert convert(35.0, "months", "years") == pytest.approx(35 / 12)


# ---------------------------------------------------------------------------
# idat
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sub,n", [("epicv1", 1_051_815), ("epicv2", 1_105_209)])
def test_idat_pairs_parse(corpus, sub, n):
    grn = sorted((corpus / "idat" / sub).glob("*_Grn.idat.gz"))[0]
    red = grn.with_name(grn.name.replace("_Grn.", "_Red."))
    pair = fa.io.read_idat_pair(grn, red)
    assert pair["grn"].shape == pair["red"].shape == pair["illumina_ids"].shape
    assert pair["grn"].size > 500_000
    assert (pair["grn"] >= 0).all()
    # Two channels of the same array must not be identical.
    assert not np.array_equal(pair["grn"], pair["red"])


# ---------------------------------------------------------------------------
# clinical
# ---------------------------------------------------------------------------
def test_nhanes_files_are_present_for_the_r_side(corpus):
    """The NHANES extracts are .rda, which R reads natively and Python does not.
    The Python clinical path is tested on a synthetic cohort; this asserts the
    fixture the R conformance test needs is actually here."""
    for name in ("NHANES3.rda", "NHANES4.rda", "NHANES3_HDTrain.rda"):
        p = corpus / "clinical" / name
        assert p.exists() and p.stat().st_size > 100_000


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def test_report_is_one_self_contained_file(corpus, tmp_path):
    pytest.importorskip("matplotlib")
    from falconage.report import write_report

    d = _bench(corpus, "GSE107143")
    res = fa.score(d, clocks=["horvath2013", "hannum", "dnamphenoage"])
    p = write_report(res, tmp_path / "report.html")
    html = p.read_text(encoding="utf-8")
    assert "data:image/png;base64," in html, "figures are inlined, not linked"
    assert "src=\"figures/" not in html
    assert "horvath2013" in html
    assert p.stat().st_size > 20_000
