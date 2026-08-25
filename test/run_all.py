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
                 group_col=None, platform_col=None, dataset_col=None,
                 se=None, conformal=None, consensus=None):
    from falconage import plot as fplot

    try:
        w = fplot.save_all(result, figdir(group, dataset), data=data, bench=bench,
                           acc=acc, group=group_col, platform_col=platform_col,
                           dataset_col=dataset_col, se=se, conformal=conformal,
                           consensus=consensus)
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
    """One benchmark study, through the package's own loader.

    This used to be four lines of column renames here, four more in the
    integration tests, and two more in each derivation script. They agreed
    until they did not; `fa.read_computage_bench` is now the single copy.
    `meta` is unused and kept so the call sites read unchanged.
    """
    return fa.read_computage_bench(gse, root=str(DATA / "bench"))


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

        # How much of the table above is the assay. Written beside the scores
        # rather than offered separately: a score table without its measurement
        # error is the thing this check exists to stop shipping.
        se = conf = cons = None
        try:
            se = fa.technical_se(res, d)
            write_table(dd, "technical_se", se.se)
            write_table(dd, "reliability_diagnostics", se.diagnostics)
        except Exception as exc:
            log(f"    technical_se skipped: {exc}")
        try:
            conf = fa.conformal_interval(res, level=0.90)
            write_table(dd, "conformal_interval", conf, index=False)
        except Exception as exc:
            log(f"    conformal skipped: {exc}")
        try:
            cons = fa.consensus(res, "condition", reference="HC")
            write_table(dd, "consensus", cons.table)
            (dd / "consensus_verdict.txt").write_text(
                f"{cons.verdict}\n{cons.why}\n", encoding="utf-8")
        except Exception as exc:
            log(f"    consensus skipped: {exc}")

        try:
            from falconage.report import write_report
            write_report(res, dd / "report.html", group="condition")
        except Exception as exc:
            log(f"    report skipped: {exc}")

        figs = make_figures("bench", gse, res, data=d, acc=acc, group_col="condition",
                            platform_col="gpl", dataset_col="dataset",
                            se=se, conformal=conf, consensus=cons)

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
    # GSE66459 is umbilical cord blood. Knight was trained on it; the three Lee
    # clocks were trained on placenta. At first this scored all four and
    # checked only that the answers looked like gestational weeks -- which they
    # did, because gestational age is gestational age whatever tissue you read
    # it from. `compatible` now refuses the placenta clocks by name, and the
    # refusals are recorded beside the scores rather than worked around.
    res = fa.score(d, clocks="compatible")
    clocks = list(res.scores.columns)

    dd = outdir("gestational", "GSE66459")
    write_table(dd, "scores_wide", res.wide())
    write_table(dd, "coverage", res.qc(), index=False)
    res.manifest.write(dd / "run_manifest.json")
    off_tissue = {k: v for k, v in res.skipped.items() if "tissue_policy=refuse" in v}
    if off_tissue:
        write_table(dd, "refused_off_tissue",
                    pd.DataFrame({"clock": list(off_tissue), "why": list(off_tissue.values())}),
                    index=False)

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
    log(f"  gestational/GSE66459: {d.n_samples}n, {len(clocks)} clocks, "
        f"{len(off_tissue)} refused off-tissue")


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
        # size=n is load-bearing: the other markers get an array `loc` and so
        # come out length n, but this one has a scalar mean and without size
        # numpy returns a single float that broadcasts to a constant column.
        # A constant marker makes KDM a division by zero, which fit_kdm now
        # refuses rather than returning a plausible number from.
        "white_blood_cell_count": rng.normal(6.5, 1.4, size=n),
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

    # ---- the two outcome figures -------------------------------------------
    # Survival and association need columns the methylation corpus does not
    # carry: a follow-up time, an event indicator, and a continuous outcome to
    # regress on. NHANES has real mortality linkage but ships as .rda, which
    # Python does not read -- the R suite covers that path. So these two are
    # drawn on the same synthetic cohort as everything else above, and the
    # gallery says so rather than implying a real survival analysis.
    #
    # The hazard is driven by PhenoAge acceleration on purpose. A survival
    # figure fitted to noise would render, look plausible, and demonstrate
    # nothing about whether the estimator works.
    aa = fa.acceleration(res_p, method="residual")["phenoage"]
    z = (aa - aa.mean()) / aa.std()
    time = rng.exponential(30.0 / np.exp(0.45 * z))
    event = (time < 25).astype(int)

    res_p.obs = res_p.obs.copy()
    res_p.obs["time"] = time.to_numpy() if hasattr(time, "to_numpy") else time
    res_p.obs["event"] = event.to_numpy() if hasattr(event, "to_numpy") else event
    res_p.obs["crp"] = df["crp"].to_numpy()

    fd = figdir("clinical", "synthetic")
    dpi = fa.plot.theme_value("dpi")

    fig, km = fa.plot.kaplan_meier(res_p, "phenoage",
                                   time_col="time", event_col="event")
    fig.savefig(fd / "kaplan_meier.png", bbox_inches="tight", dpi=dpi)
    write_table(dd, "kaplan_meier", km, index=False)

    # Three clocks rather than one, so the volcano has points to separate.
    # They are scored above in three calls because `reference=` takes one
    # reference object and KDM and HD need different ones; the columns are
    # merged here rather than re-scored.
    res_v = res_p
    res_v.scores = pd.concat([res_p.scores, res_k.scores, res_h.scores], axis=1)
    assoc = fa.associate(res_v, "crp", covariates=("age",))
    fig, vol = fa.plot.volcano(assoc)
    fig.savefig(fd / "volcano.png", bbox_inches="tight", dpi=dpi)
    write_table(dd, "volcano", vol)

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
    # These filters name the availability groups, not the old A/B/C letters.
    # Matching on the letters silently produced three empty tables, because the
    # summary column carries the names the registry now uses.
    for group in ("bundled", "untraced", "licensed"):
        write_table(dd, group, s[s["availability"] == group])

    records["registry"] = pd.DataFrame([{
        "clocks": len(reg),
        "bundled": len(reg.filter(availability="bundled")),
        "untraced": len(reg.filter(availability="untraced")),
        "licensed": len(reg.filter(availability="licensed")),
        "coefficients_bundled": sum(1 for c in reg if c.ships_coefficients),
        "registry_version": reg.version,
    }])


def run_disorder(records: dict) -> None:
    """The readouts that are not clocks: entropy, drift, repertoire, mass.

    Run on GSE182991 because it is the widest age range in the corpus (0 to 41)
    on EPIC, and every statistic here is either about age-related variance or is
    reported per sample, so a cohort that is all one age would show nothing.
    """
    gse = "GSE182991"
    path = DATA / "bench" / f"{gse}.parquet"
    if not path.exists():
        log(f"  disorder: {gse} absent, skipped")
        return
    meta = pd.read_csv(DATA / "bench" / "computage_bench_meta.tsv", sep="\t", index_col=0)
    d = load_bench(gse, meta)
    dd = outdir("disorder", gse)

    # -- entropy and drift, per sample --------------------------------------
    ent = fa.entropy(d)
    dri = fa.drift(d)
    per_sample = ent.join(dri, rsuffix="_d")[["entropy", "drift", "n_sites"]]
    if "age" in d.obs.columns:
        per_sample["age"] = d.obs["age"]
    write_table(dd, "entropy_drift", per_sample)

    # -- which sites actually widen with age --------------------------------
    row = {"dataset": gse, "n_samples": d.n_samples, "n_sites": d.n_features,
           "entropy_mean": float(ent["entropy"].mean()),
           "entropy_sd": float(ent["entropy"].std()),
           "drift_mean": float(dri["drift"].mean())}
    try:
        var = fa.variable_sites(d)
        tested = int(var["p"].notna().sum())
        rising = int(var["rising"].sum())
        row["sites_tested"] = tested
        row["sites_rising_fdr"] = rising

        # Zero is the expected answer here and is reported as the headline
        # rather than worked around: 48 samples split into three bins of 16,
        # against ~820,000 Brown-Forsythe tests, has almost no power once
        # Benjamini-Hochberg is applied. A barometer needs a cohort in the
        # hundreds, and saying so is more use than a number that survived only
        # because the correction was skipped.
        nominal = var.index[(var["p"] < 0.01) & (var["direction"] > 0)]
        row["sites_rising_nominal_p01"] = int(len(nominal))
        write_table(dd, "variable_sites_nominal", var.loc[nominal].head(2000))

        chosen = list(var.index[var["rising"]]) or list(nominal)
        if chosen:
            bar = fa.noise_barometer(d, sites=chosen)
            bar["selection"] = "fdr" if rising else "nominal p<0.01, uncorrected"
            write_table(dd, "noise_barometer", bar)
            row["barometer"] = float(bar["barometer"].iloc[0])
            row["mean_sd"] = float(bar["mean_sd"].iloc[0])
            row["barometer_selection"] = bar["selection"].iloc[0]
    except Exception as exc:
        log(f"    variable_sites skipped: {exc}")

    records["disorder"] = pd.DataFrame([row])

    # -- where each clock's weight sits, against the other clocks ------------
    # No external annotation is needed for a real question: how much of one
    # clock's coefficient mass sits on CpGs another clock also uses. Shared
    # features are why two clocks agree, and mass is the honest way to measure
    # the sharing, because a handful of heavy probes in common matters more
    # than a long tail of light ones.
    reg = fa.registry.load()
    anchors = [c for c in ("horvath2013", "hannum2013", "dnamphenoage")
               if c in reg and reg.has_coefficient_vector(c)]
    if anchors:
        classes = {a: set(reg.coefficients(a)[0]) for a in anchors}
        mass = fa.coefficient_mass(classes, registry=reg)
        write_table(outdir("disorder", "coefficient_mass"), "shared_mass", mass)
        records["coefficient_mass"] = mass.head(12).reset_index()

    # -- the clonality mechanism, simulated ---------------------------------
    # Cell fractions are identical across every simulated sample by
    # construction, so anything the clocks do here is clone structure alone.
    # The two lineage profiles are built from this dataset's own per-site means
    # with a fixed perturbation, which keeps the simulation on a real CpG panel
    # and a real beta distribution rather than on uniform noise.
    base = d.X.mean(axis=0).dropna()
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({
        "lineageA": base.to_numpy(),
        "lineageB": np.clip(base.to_numpy() + rng.normal(0, 0.05, base.size), 0, 1),
    }, index=base.index)

    sizes = [fa.immune.zipf_clone_sizes(n, alpha=1.0)
             for n in (2, 5, 20, 100, 500, 2000)]
    sim = fa.simulate_clonality(ref, {"lineageA": 0.4, "lineageB": 0.6},
                                clonal_types=["lineageA"], clone_sizes=sizes,
                                n_replicates=8, sigma=0.05, seed=0,
                                age=float(d.obs["age"].median())
                                if "age" in d.obs.columns else 50.0)
    sim.platform = d.platform
    sim_res = fa.score(sim, clocks="compatible", min_coverage=0.5)
    sd = outdir("disorder", "clonality_simulation")
    write_table(sd, "scores", sim_res.wide().join(
        sim.obs[["n_clones", "effective_clones", "simpson"]]))

    # The prediction from the derivation: spread falls as 1/sqrt(N_eff), so the
    # slope of log(spread) on log(N_eff) should be about -0.5 for a clock that
    # reads the clonal compartment at all.
    slopes = []
    for cid in sim_res.scores.columns:
        g = sim_res.scores[cid].groupby(sim.obs["effective_clones"]).std()
        g = g[np.isfinite(g) & (g > 0)]
        if len(g) >= 4:
            b = np.polyfit(np.log(g.index.to_numpy(dtype=float)),
                           np.log(g.to_numpy()), 1)[0]
            slopes.append({"clock": cid, "log_log_slope": round(float(b), 3),
                           "spread_at_2_clones": round(float(g.iloc[0]), 4),
                           "spread_at_most_clones": round(float(g.iloc[-1]), 4)})
    if slopes:
        tab = pd.DataFrame(slopes).sort_values("log_log_slope")
        write_table(sd, "clonality_slopes", tab, index=False)
        records["clonality"] = tab.head(12)
        records["disorder"]["clonality_clocks"] = len(tab)
        records["disorder"]["clonality_median_slope"] = round(
            float(tab["log_log_slope"].median()), 3)
        # Only the clocks whose output is in years get a spread quoted in years.
        # epiTOC2 tops the raw table at 66, and that number is cell divisions:
        # reporting it as a worst case in years would be a units error of the
        # exact kind scale_type exists to prevent.
        in_years = [r for r in slopes
                    if reg.get(r["clock"]).scale_type.startswith("age_years")]
        if in_years:
            worst = max(in_years, key=lambda r: r["spread_at_2_clones"])
            records["disorder"]["clonality_worst_clock_in_years"] = worst["clock"]
            records["disorder"]["clonality_worst_spread_years"] = worst["spread_at_2_clones"]



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
    ("kaplan_meier",     "clinical", "synthetic", "the only cohort with a follow-up time; synthetic, and the gallery notes say so"),
    ("volcano",          "clinical", "synthetic", "same cohort, the only one with a continuous outcome to regress on"),
    # ---- how much of the numbers above is the assay ----------------
    ("reliability_forest", "bench", "GSE182991", "EPIC, widest age range, so the spread the noise is measured against is real"),
    ("score_interval",     "bench", "GSE182991", "same cohort; both intervals drawn on one clock"),
    ("platform_bias",      "bench", "_combined", "read off the shipped measurement table, not this dataset"),
    ("consensus_plot",     "bench", "GSE107143", "the one condition every clock detects, so the verdict is 'supported'"),
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
                "disorder", "coefficient_mass", "clonality", "gallery"):
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
                         "mouse, idat, clinical, registry, disorder")
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
        "disorder": run_disorder,
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
