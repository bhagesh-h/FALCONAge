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
    """Through the package's own loader, not a local copy of its renames."""
    return fa.read_computage_bench(gse, root=str(corpus / "bench"))


# ---------------------------------------------------------------------------
# bench: EPIC, 450K, 27K
# ---------------------------------------------------------------------------
def test_epic_progeria_scores_every_tier_a_clock_the_specimen_allows(corpus):
    """Every tier A clock that has a calibration on blood, and no others.

    This asserted 18 until the specimen check existed, which meant it was
    scoring three placenta clocks, a cord-blood clock and a buccal clock on
    adult blood and calling the numbers results. They are refused now, and the
    five names are pinned -- a silent drop back to 18 would mean the check
    stopped running.

    24 rather than 13 since the mitotic clocks, AltumAge and Weidner started
    shipping: epiTOC1, HypoClock, stemTOC, stemTOCvitro, epiCMIT-hyper,
    epiCMIT-hypo, RepliTali, epiTOC2, epiTOC3, AltumAge and Weidner. The
    mitotic ones and AltumAge are multi-tissue; Weidner is a blood clock and
    this is blood.

    34 since the ten McCartney EpiScores were traced to their own supplement.
    All ten were fitted in whole blood, so all ten are compatible here. The
    cortical clock arrived in the same batch and is NOT among them: it refuses
    blood, which is the check doing its job on a clock added the same day.
    """
    d = _bench(corpus, "GSE182991")
    assert d.n_samples == 27
    assert d.platform == "EPICv1"

    res = fa.score(d, clocks="compatible")
    assert res.scores.shape[1] == 34
    assert res.scores.notna().all().all()

    off_tissue = {cid for cid, why in res.skipped.items() if "tissue_policy=refuse" in why}
    # Three placenta clocks, a cord-blood clock, a buccal clock, and the cortical
    # clock, which is post-mortem brain and has no peripheral counterpart.
    assert off_tissue == {"knight", "leecontrol", "leerobust", "leerefinedrobust",
                          "pedbe", "corticalclock"}

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

    # GSE66459 is umbilical cord blood. Knight was trained on it; the three Lee
    # clocks were trained on placenta, and this test used to score all four and
    # check only that the answers looked like gestational weeks -- which they
    # do, because gestational age is gestational age whatever tissue you read
    # it from. That is precisely the failure the specimen check exists to catch.
    assert d.obs["tissue"].str.lower().str.contains("cord blood").all()

    res = fa.score(d, clocks="compatible")
    off_tissue = {cid for cid, why in res.skipped.items() if "tissue_policy=refuse" in why}
    assert {"leecontrol", "leerobust", "leerefinedrobust"} <= off_tissue

    # Predictions are in weeks and should sit in a term-ish range.
    assert 25 < res.scores["knight"].median() < 50

    # And they should track the recorded gestational age once it is in weeks --
    # the conversion the units module refuses to do silently.
    weeks = pd.to_numeric(d.obs["gestational_age_days"], errors="coerce") / 7.0
    assert res.scores["knight"].corr(weeks) > 0.3

    # Asking for a placenta clock on cord blood by name is a refusal, not a
    # skip: an explicit request is never silently dropped.
    with pytest.raises(fa.core.errors.ScoringError, match="category error"):
        fa.score(d, clocks=["leecontrol"])


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


@pytest.mark.network
def test_raw_idats_reproduce_the_published_betas_for_the_same_samples(corpus):
    """The end-to-end check on the IDAT chain, and the only one that matters.

    ``idat/epicv1`` holds the raw IDATs for GSM5548192 and GSM5548193.
    ``bench/GSE182991.parquet`` holds the betas the authors published for those
    same two physical samples. The two paths share nothing -- different files,
    different pipelines, different decade -- so agreement is a real check on
    the address decoding, the type I/II split and the channel assignment, which
    is where a raw-array reader goes wrong.

    Asserted on the *uncorrected* betas. GSE182991's published matrix agrees
    with our decode at r = 0.999, which says its own processing was minimal;
    comparing our background-corrected output against it would be asserting
    that a correction does nothing.

    Marked network: it needs the Illumina manifest, which is fetched once and
    cached and is the only step in FALCONAge that is not offline.
    """
    grn = sorted((corpus / "idat" / "epicv1").glob("GSM5548193*_Grn.idat.gz"))[0]
    red = grn.with_name(grn.name.replace("_Grn.", "_Red."))

    sig = fa.preprocess.idat_to_betas(grn, red, correct=False, raw=True)
    assert sig.platform == "EPICv1"
    assert 0.005 < sig.summary()["frac_undetected"] < 0.10, "a normal failure rate"

    ours = sig.betas(detection_p=None)
    ref = fa.read_computage_bench("GSE182991", root=str(corpus / "bench"),
                                  prepare=False).X.loc["GSM5548193"].dropna()
    shared = ours.dropna().index.intersection(ref.index)
    assert len(shared) > 750_000

    x, y = ours[shared].to_numpy(), ref[shared].to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    assert np.corrcoef(x, y)[0, 1] > 0.99
    assert float(np.median(np.abs(x - y))) < 0.03
    assert float(np.mean(np.abs(x - y) < 0.05)) > 0.95


@pytest.mark.network
def test_the_manifest_that_decoded_a_matrix_is_recorded(corpus):
    """A beta matrix is a function of which manifest turned addresses into
    probes, in the way a score is a function of which coefficient file was
    used. Recording one and not the other would be an odd place to stop."""
    d = fa.read_idat_dir(corpus / "idat" / "epicv1")
    rec = d.uns["idat_manifest"]
    assert rec["platform"] == "EPICv1"
    assert len(rec["sha256"]) == 64
    assert rec["redistributed"].startswith("no")
    assert int(rec["n_type_i"]) > 100_000 and int(rec["n_type_ii"]) > 500_000
    assert "poobah" in d.uns["pipeline"] and "noob" in d.uns["pipeline"]


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
