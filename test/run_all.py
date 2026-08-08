#!/usr/bin/env python3
"""Run every feature against every corpus dataset and write the results.

Outputs land in ``test/output/<group>/<dataset>/`` -- one directory per input,
overwritten on each run so the tree always reflects the current code rather than
accumulating a history nobody prunes. Nothing here is committed; ``.gitignore``
excludes the whole tree except its README.

The last thing this does is rewrite the generated tables in ``test/README.md``
between their marker comments. That is the point of running it as one script
rather than by hand: the numbers in the documentation cannot drift from the
numbers the code produces, because they are the same numbers.

Usage
-----
    python test/run_all.py                 # everything
    python test/run_all.py --groups bench  # one group
    python test/run_all.py --no-readme     # skip the README rewrite
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import falconage as fa

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "output"
FIGS = HERE / "output_figures"
README = HERE / "README.md"

BEGIN = "<!-- BEGIN GENERATED: {name} -->"
END = "<!-- END GENERATED: {name} -->"


def log(msg: str) -> None:
    print(msg, flush=True)


def outdir(group: str, dataset: str) -> Path:
    d = OUT / group / dataset
    d.mkdir(parents=True, exist_ok=True)
    return d


def figdir(group: str, dataset: str) -> Path:
    """Figures go somewhere else from the tables, and are tracked in git.

    The tables are regenerated noise; the figures are what the documentation
    points at, so a reader who clones the repository sees them without first
    downloading the corpus and running the pipeline.
    """
    d = FIGS / group / dataset
    d.mkdir(parents=True, exist_ok=True)
    return d


def make_figures(group, dataset, result, *, data=None, bench=None, acc=None,
                 group_col=None, platform_col=None, dataset_col=None):
    from falconage import plot as fplot

    try:
        w = fplot.save_all(result, figdir(group, dataset), data=data, bench=bench,
                           acc=acc, group=group_col, platform_col=platform_col,
                           dataset_col=dataset_col)
        log(f"    {len(w)} figure(s)")
        return list(w)
    except Exception as exc:
        log(f"    figures skipped: {exc}")
        return []


def write_table(d: Path, name: str, df: pd.DataFrame, index: bool = True) -> None:
    df.to_csv(d / f"{name}.csv", index=index)


# ---------------------------------------------------------------------------
# bench: the ComputAgeBench studies
# ---------------------------------------------------------------------------
def load_bench(gse: str, meta: pd.DataFrame) -> fa.FalconData:
    X = pd.read_parquet(DATA / "bench" / f"{gse}.parquet").T.astype("float64")
    obs = meta[meta["DatasetID"] == gse].reindex(X.index).rename(columns={
        "Age": "age", "Gender": "sex", "Condition": "condition",
        "DatasetID": "dataset", "PlatformID": "gpl", "Tissue": "tissue"})
    return fa.prepare(fa.FalconData(X=X, obs=obs, modality="dna_methylation"))


def run_bench(records: dict) -> None:
    meta_path = DATA / "bench" / "computage_bench_meta.tsv"
    if not meta_path.exists():
        log("  bench: absent, skipped")
        return
    meta = pd.read_csv(meta_path, sep="\t", index_col=0)

    results, rows = [], []
    for p in sorted(glob.glob(str(DATA / "bench" / "GSE*.parquet"))):
        gse = Path(p).stem
        t0 = time.time()
        d = load_bench(gse, meta)
        res = fa.score(d, clocks="compatible", min_coverage=0.5)
        secs = time.time() - t0

        dd = outdir("bench", gse)
        write_table(dd, "scores_wide", res.wide())
        write_table(dd, "scores_long", res.long(), index=False)
        write_table(dd, "coverage", res.qc(), index=False)
        res.manifest.write(dd / "run_manifest.json")

        acc = fa.acceleration(res, method="residual")
        write_table(dd, "acceleration_residual", acc)
        write_table(dd, "agreement_spearman", fa.agreement(res))

        try:
            from falconage.report import write_report
            write_report(res, dd / "report.html", group="condition")
        except Exception as exc:
            log(f"    report skipped: {exc}")

        figs = make_figures("bench", gse, res, data=d, acc=acc, group_col="condition",
                            platform_col="gpl", dataset_col="dataset")

        conds = d.obs["condition"].value_counts().to_dict()
        rows.append({
            "dataset": gse, "n": d.n_samples, "platform": d.platform or "?",
            "probes": d.n_features, "clocks_scored": res.scores.shape[1],
            "skipped": len(res.skipped),
            "conditions": ", ".join(f"{k}={v}" for k, v in sorted(conds.items())),
            "age_range": f"{d.obs['age'].min():.0f}-{d.obs['age'].max():.0f}",
            "figures": len(figs),
            "seconds": round(secs, 1),
        })
        log(f"  bench/{gse}: {d.n_samples}n {d.platform} -> "
            f"{res.scores.shape[1]} clocks in {secs:.1f}s")
        results.append(res)
        del d

    if not results:
        return

    combined = fa.combine(results)
    dd = outdir("bench", "_combined")
    write_table(dd, "scores_wide", combined.wide())
    combined.manifest.write(dd / "run_manifest.json")

    bench = fa.run_benchmark(combined, condition_col="condition", control="HC",
                             dataset_col="dataset")
    write_table(dd, "benchmark_summary", bench.summary())
    write_table(dd, "benchmark_per_dataset", bench.per_dataset, index=False)

    acc_all = fa.acceleration(combined, method="within_group", group="dataset")
    write_table(dd, "acceleration_within_dataset", acc_all)
    make_figures("bench", "_combined", combined, bench=bench, acc=acc_all,
                 group_col="condition", dataset_col="dataset")

    records["bench_inputs"] = pd.DataFrame(rows)
    records["benchmark"] = bench.summary().reset_index()
    sig = bench.per_dataset[bench.per_dataset["significant"]]
    records["benchmark_significant"] = sig[
        ["clock", "dataset", "condition", "test", "n_case", "n_control",
         "delta", "q"]].round(4)
    records["benchmark_counts"] = {
        "comparisons": int(len(bench.per_dataset)),
        "significant": int(len(sig)),
        "datasets": int(bench.per_dataset["dataset"].nunique()),
        "clocks": int(bench.summary().shape[0]),
    }
    log(f"  bench/_combined: {len(sig)}/{len(bench.per_dataset)} significant")


# ---------------------------------------------------------------------------
# the other groups
# ---------------------------------------------------------------------------
def run_epicv2(records: dict) -> None:
    p = DATA / "epicv2" / "GSE330325_series_matrix.txt.gz"
    if not p.exists():
        return
    raw = fa.read_series_matrix(p)
    reg = fa.registry.load()
    before = raw.coverage(list(reg.feature_ids("horvath2013")))
    prepared = fa.prepare(raw)
    after = prepared.coverage(list(reg.feature_ids("horvath2013")))

    res = fa.score(prepared, clocks="compatible", min_coverage=0.5)
    dd = outdir("epicv2", "GSE330325")
    write_table(dd, "scores_wide", res.wide())
    write_table(dd, "coverage", res.qc(), index=False)
    res.manifest.write(dd / "run_manifest.json")

    make_figures("epicv2", "GSE330325", res, data=prepared)

    records["epicv2"] = pd.DataFrame([{
        "probes_raw": raw.n_features,
        "probes_after_aggregation": prepared.n_features,
        "collapsed": raw.n_features - prepared.n_features,
        "horvath_coverage_raw": round(before, 4),
        "horvath_coverage_aggregated": round(after, 4),
        "clocks_scored": res.scores.shape[1],
    }])
    log(f"  epicv2/GSE330325: coverage {before:.1%} -> {after:.1%}")


def run_gestational(records: dict) -> None:
    p = DATA / "gestational" / "GSE66459_series_matrix.txt.gz"
    if not p.exists():
        return
    d = fa.prepare(fa.read_series_matrix(p))
    clocks = ["knight", "leecontrol", "leerobust", "leerefinedrobust"]
    res = fa.score(d, clocks=clocks)

    dd = outdir("gestational", "GSE66459")
    write_table(dd, "scores_wide", res.wide())
    write_table(dd, "coverage", res.qc(), index=False)
    res.manifest.write(dd / "run_manifest.json")

    weeks = pd.to_numeric(d.obs["gestational_age_days"], errors="coerce") / 7.0
    cmp = pd.DataFrame({
        "clock": clocks,
        "median_predicted_weeks": [round(float(res.scores[c].median()), 2) for c in clocks],
        "median_recorded_weeks": round(float(weeks.median()), 2),
        "cor_with_recorded": [round(float(res.scores[c].corr(weeks)), 3) for c in clocks],
        "medae_weeks": [round(float((res.scores[c] - weeks).abs().median()), 2) for c in clocks],
    })
    write_table(dd, "vs_recorded_gestational_age", cmp, index=False)
    make_figures("gestational", "GSE66459", res, data=d)
    records["gestational"] = cmp
    log(f"  gestational/GSE66459: {d.n_samples}n, 4 clocks")


def run_mammalian(records: dict) -> None:
    rows = []
    for gse, species in (("GSE184222", "Equus grevyi"), ("GSE184224", "Homo sapiens")):
        p = DATA / "mammalian" / f"{gse}_datBetaNormalized.csv.gz"
        if not p.exists():
            continue
        d = fa.read_betas(p)
        d.species = species
        reg = fa.registry.load()
        cov = d.coverage(list(reg.feature_ids("horvath2013")))
        res = fa.score(d, clocks=["horvath2013", "hannum"], min_coverage=0.5)

        dd = outdir("mammalian", gse)
        write_table(dd, "scores_wide", res.wide())
        res.manifest.write(dd / "run_manifest.json")

        make_figures("mammalian", gse, res, data=d)

        warns = [w for w in res.manifest.warnings if w["category"] == "species"]
        rows.append({
            "dataset": gse, "species": species, "n": d.n_samples,
            "probes": d.n_features,
            "horvath_coverage": round(cov, 4),
            "median_horvath": round(float(res.scores["horvath2013"].median()), 2),
            "species_warning": bool(warns),
        })
        log(f"  mammalian/{gse}: {species}, coverage {cov:.1%}, "
            f"warning={bool(warns)}")
    if rows:
        records["mammalian"] = pd.DataFrame(rows)


def run_mouse(records: dict) -> None:
    paths = sorted((DATA / "mouse").glob("GSM*.overlap.txt.gz"))
    if not paths:
        return
    d = fa.read_rrbs_dir(paths, min_coverage=5)
    dd = outdir("mouse", "GSE80672")
    d.X.iloc[:, :2000].to_csv(dd / "betas_head.csv")

    meta = fa.read_series_matrix(DATA / "mouse" / "GSE80672_series_matrix.txt.gz")
    ages = pd.to_numeric(meta.obs["age_years"], errors="coerce")
    rows = []
    for cov in (1, 5, 20):
        one = fa.io.read_rrbs(paths[0], min_coverage=cov)
        rows.append({"min_coverage": cov, "sites_kept": len(one)})
    tbl = pd.DataFrame(rows)
    write_table(dd, "coverage_filter", tbl, index=False)

    records["mouse"] = pd.DataFrame([{
        "samples": d.n_samples,
        "shared_sites": d.n_features,
        "sites_at_cov1": int(tbl.loc[0, "sites_kept"]),
        "sites_at_cov5": int(tbl.loc[1, "sites_kept"]),
        "sites_at_cov20": int(tbl.loc[2, "sites_kept"]),
        "series_age_field": "age (years)",
        "series_age_min": float(ages.min()),
        "series_age_max": float(ages.max()),
    }])
    log(f"  mouse/GSE80672: {d.n_samples} samples, {d.n_features} shared sites")


def run_idat(records: dict) -> None:
    rows = []
    for sub in ("epicv1", "epicv2"):
        grns = sorted((DATA / "idat" / sub).glob("*_Grn.idat.gz"))
        for grn in grns:
            red = grn.with_name(grn.name.replace("_Grn.", "_Red."))
            if not red.exists():
                continue
            pair = fa.io.read_idat_pair(grn, red)
            rows.append({
                "platform": sub, "sample": grn.name.split("_")[0],
                "addresses": int(pair["grn"].size),
                "grn_median": float(np.median(pair["grn"])),
                "red_median": float(np.median(pair["red"])),
                "channels_identical": bool(np.array_equal(pair["grn"], pair["red"])),
            })
    if rows:
        df = pd.DataFrame(rows)
        write_table(outdir("idat", "raw"), "intensities", df, index=False)
        records["idat"] = df
        log(f"  idat: {len(rows)} pairs parsed")


def run_clinical(records: dict) -> None:
    """Synthetic, because NHANES ships as .rda and Python does not read it.

    The R suite covers the real NHANES path; this covers the arithmetic on a
    cohort whose answers can be checked, which is the part that has to be right.
    """
    rng = np.random.default_rng(20260807)
    n = 500
    age = rng.uniform(25, 85, n)
    ids = [f"C{i:04d}" for i in range(n)]
    df = pd.DataFrame({
        "albumin": rng.normal(43 - 0.03 * age, 2.5),
        "creatinine": rng.normal(70 + 0.25 * age, 12),
        "glucose": rng.normal(5.0 + 0.012 * age, 0.7),
        "crp": np.exp(rng.normal(-1.0 + 0.012 * age, 0.6)),
        "lymphocyte_percent": rng.normal(32 - 0.06 * age, 5),
        "mean_cell_volume": rng.normal(89 + 0.03 * age, 4),
        "red_cell_distribution_width": rng.normal(12.8 + 0.012 * age, 0.7),
        "alkaline_phosphatase": rng.normal(70 + 0.20 * age, 15),
        "white_blood_cell_count": rng.normal(6.5, 1.4),
        "age": age,
    }, index=ids)
    d = fa.FalconData(X=df, obs=pd.DataFrame({"age": age}, index=ids),
                      modality="clinical_chemistry")

    from falconage.models.clinical import fit_hd, fit_kdm
    markers = [c for c in df.columns if c != "age"]
    kref = fit_kdm(df, markers)
    href = fit_hd(df[df["age"] < 35], markers)

    res_p = fa.score(d, clocks=["phenoage"])
    res_k = fa.score(d, clocks=["kdm"], reference=kref)
    res_h = fa.score(d, clocks=["hd"], reference=href)

    dd = outdir("clinical", "synthetic")
    out = pd.concat([res_p.scores, res_k.scores, res_h.scores], axis=1)
    out.insert(0, "age", age)
    write_table(dd, "scores_wide", out)
    res_p.manifest.write(dd / "run_manifest.json")

    make_figures("clinical", "synthetic", res_p)

    records["clinical"] = pd.DataFrame([{
        "clock": c,
        "median": round(float(out[c].median()), 2),
        "cor_with_age": round(float(out[c].corr(out["age"])), 3),
        "medae_vs_age": (round(float((out[c] - out["age"]).abs().median()), 2)
                         if c != "hd" else np.nan),
    } for c in ("phenoage", "kdm", "hd")])
    log("  clinical/synthetic: 3 clocks on 500 samples")


def run_registry(records: dict) -> None:
    reg = fa.registry.load()
    s = reg.summary()
    dd = outdir("registry", "catalogue")
    write_table(dd, "clocks", s)
    write_table(dd, "tier_a", s[s["availability"] == "A"])
    write_table(dd, "tier_c", s[s["availability"] == "C"])

    records["registry"] = pd.DataFrame([{
        "clocks": len(reg),
        "tier_A": len(reg.filter(availability="A")),
        "tier_B": len(reg.filter(availability="B")),
        "tier_C": len(reg.filter(availability="C")),
        "untraced": len(reg.untraced()),
        "coefficients_bundled": sum(1 for c in reg if c.ships_coefficients),
        "registry_version": reg.version,
    }])



# ---------------------------------------------------------------------------
# the gallery
# ---------------------------------------------------------------------------
# Every figure the run produces lands under test/output_figures/<group>/<dataset>/,
# which comes to roughly five hundred PNGs and forty megabytes of near-duplicates
# -- the same twenty figure types, once per input. That is the right thing to
# have on disk after a run and the wrong thing to put in a repository.
#
# So one representative of each figure TYPE is copied into gallery/, chosen from
# the dataset that shows it best rather than from whichever happened to run
# first. The gallery is what the README and both documentation sites link to,
# it is what a reader sees without downloading 586 MB of corpus, and it is small
# enough that a diff saying "this figure changed" is a useful review signal when
# somebody edits colorscheme.yaml.
#
# The choice of source dataset per figure is deliberate and is stated here so it
# can be argued with.
GALLERY_SOURCES = [
    # figure type              group        dataset      why this one
    ("clock_atlas",           "bench", "_combined", "the one figure that answers which algorithms work"),
    ("ba_vs_ca",              "bench", "GSE182991", "widest age range in the corpus, 0-41"),
    ("bland_altman",          "bench", "GSE182991", "age range wide enough for a trend to show"),
    ("calibration",           "bench", "GSE182991", "same"),
    ("acceleration_group",    "bench", "GSE107143", "the one condition every clock detects"),
    ("acceleration_density",  "bench", "GSE130030", "balanced 14 vs 14"),
    ("acceleration_heatmap",  "bench", "_combined", "all ten studies at once"),
    ("clock_corr",            "bench", "_combined", "twenty clocks, enough to cluster"),
    ("clock_chord",           "bench", "_combined", "feature sharing needs the full clock set"),
    ("clock_radar",           "bench", "_combined", "profile across conditions"),
    ("clock_pca",             "bench", "_combined", "structure only appears across studies"),
    ("coverage_bar",          "bench", "GSE56606",  "27K: the platform where coverage bites"),
    ("beta_density",          "bench", "GSE182991", "EPIC, clean bimodal"),
    ("missingness",           "bench", "GSE130030", "typical 450K series"),
    ("platform",              "bench", "_combined", "three array generations side by side"),
    ("study",                 "bench", "_combined", "between-study spread"),
    ("benchmark_bars",        "bench", "_combined", "the headline benchmark figure"),
    ("benchmark_error_bias",  "bench", "_combined", "the MedAE/MedE tradeoff"),
    ("benchmark_heatmap",     "bench", "_combined", "clock x dataset"),
    ("forest",                "bench", "_combined", "effect sizes with intervals"),
]


def build_gallery(records: dict) -> None:
    """Copy one representative of each figure type into gallery/."""
    import shutil

    gal = FIGS / "gallery"
    if gal.exists():
        shutil.rmtree(gal)
    gal.mkdir(parents=True, exist_ok=True)

    rows = []
    for kind, group, dataset in [(k, g, d) for k, g, d, _ in GALLERY_SOURCES]:
        why = next(w for k, g, d, w in GALLERY_SOURCES if (k, g, d) == (kind, group, dataset))
        src_dir = FIGS / group / dataset
        if not src_dir.exists():
            continue
        # An exact name first, then the first per-clock variant (ba_vs_ca_horvath2013).
        cands = sorted(src_dir.glob(f"{kind}.png")) or sorted(src_dir.glob(f"{kind}_*.png"))
        if not cands:
            continue
        src = cands[0]
        dst = gal / f"{kind}.png"
        shutil.copyfile(src, dst)
        rows.append({"figure": kind, "source": f"{group}/{dataset}/{src.name}",
                     "chosen_because": why,
                     "kb": round(dst.stat().st_size / 1024)})

    records["gallery"] = pd.DataFrame(rows)
    header = [
        "# Gallery sources",
        "",
        "One figure per type, copied from the dataset that shows it best.",
        "Regenerated by `test/run_all.py`; edit the choices in `GALLERY_SOURCES` there.",
        "",
        "| figure | source | chosen because |",
        "|---|---|---|",
    ]
    body = [f"| `{r['figure']}` | `{r['source']}` | {r['chosen_because']} |" for r in rows]
    header = [
        "# Gallery sources",
        "",
        "One figure per type, copied from the dataset that shows it best.",
        "Regenerated by `test/run_all.py`; edit the choices in `GALLERY_SOURCES` there.",
        "",
        "| figure | source | chosen because |",
        "|---|---|---|",
    ]
    body = [f"| `{r['figure']}` | `{r['source']}` | {r['chosen_because']} |"
            for r in rows]
    (gal / "SOURCES.md").write_text("\n".join(header + body) + "\n",
                                    encoding="utf-8", newline="\n")
    log(f"  gallery: {len(rows)} figure(s) in {gal}")


# ---------------------------------------------------------------------------
# README regeneration
# ---------------------------------------------------------------------------
def md_table(df: pd.DataFrame) -> str:
    if df is None or len(df) == 0:
        return "_no rows_"
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for v in r:
            if isinstance(v, float):
                cells.append("" if pd.isna(v) else f"{v:g}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def splice(text: str, name: str, body: str) -> str:
    b, e = BEGIN.format(name=name), END.format(name=name)
    if b not in text or e not in text:
        return text
    head = text.split(b)[0]
    tail = text.split(e)[1]
    return f"{head}{b}\n{body}\n{e}{tail}"


def update_readme(records: dict) -> None:
    if not README.exists():
        log("  test/README.md missing; skipping the rewrite")
        return
    text = README.read_text(encoding="utf-8")

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    cfg = fa.config()
    text = splice(text, "stamp",
                  f"_Generated by `test/run_all.py` on {stamp} · FALCONAge "
                  f"{cfg['falconage']} · registry {cfg['registry_version']} · "
                  f"{cfg['backend'] if 'backend' in cfg else 'numpy'} on "
                  f"{', '.join(cfg['devices'])}._")

    for key in ("registry", "bench_inputs", "benchmark", "benchmark_significant",
                "epicv2", "gestational", "mammalian", "mouse", "idat", "clinical",
                "gallery"):
        if key in records:
            v = records[key]
            text = splice(text, key, md_table(v) if isinstance(v, pd.DataFrame)
                          else "```json\n" + json.dumps(v, indent=2) + "\n```")

    if "benchmark_counts" in records:
        c = records["benchmark_counts"]
        text = splice(text, "benchmark_counts",
                      f"**{c['significant']} of {c['comparisons']} comparisons "
                      f"significant** at BH q < 0.05, across {c['datasets']} "
                      f"datasets and {c['clocks']} clocks.")

    README.write_text(text, encoding="utf-8", newline="\n")
    log(f"  rewrote the generated blocks in {README}")


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--groups", default="all",
                    help="comma-separated: bench, epicv2, gestational, mammalian, "
                         "mouse, idat, clinical, registry")
    ap.add_argument("--no-readme", action="store_true")
    args = ap.parse_args(argv)

    if not (DATA / "checksums.sha256").exists():
        sys.exit("test corpus absent. Fetch it first:\n"
                 "  docker run --rm -v \"$PWD/test/data:/data\" "
                 "falconage-testdata:1.0.0 python")

    runners = {
        "registry": run_registry, "bench": run_bench, "epicv2": run_epicv2,
        "gestational": run_gestational, "mammalian": run_mammalian,
        "mouse": run_mouse, "idat": run_idat, "clinical": run_clinical,
    }
    wanted = list(runners) if args.groups == "all" else \
        [g.strip() for g in args.groups.split(",")]

    OUT.mkdir(parents=True, exist_ok=True)
    records: dict = {}
    failed = []

    log(f"FALCONAge {fa.__version__} -- writing to {OUT}")
    for g in wanted:
        if g not in runners:
            log(f"  unknown group {g!r}; known: {', '.join(runners)}")
            continue
        log(f"[{g}]")
        try:
            runners[g](records)
        except Exception:
            failed.append(g)
            log(f"  FAILED:\n{traceback.format_exc()}")

    log("[gallery]")
    build_gallery(records)

    figs = sorted(p.relative_to(FIGS).as_posix() for p in FIGS.rglob("*.png")
                  if not p.parent.name == "gallery")
    records["figure_inventory"] = pd.DataFrame(
        [{"group": f.split("/")[0], "dataset": f.split("/")[1],
          "figure": f.split("/")[-1].replace(".png", "")} for f in figs])
    records["figure_counts"] = {
        "figures": len(figs),
        "megabytes": round(sum((FIGS / f).stat().st_size for f in figs) / 1e6, 1),
    }

    if not args.no_readme:
        log("[readme]")
        update_readme(records)

    figs = sorted(p for p in FIGS.rglob("*.png"))
    if figs:
        log(f"[figures] {len(figs)} in {FIGS}")
    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    log(f"\n{sum(1 for _ in OUT.rglob('*') if _.is_file())} file(s), "
        f"{total / 1e6:.1f} MB in {OUT}")
    if failed:
        log("failed groups: " + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
