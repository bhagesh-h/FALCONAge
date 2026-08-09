"""Measure, in years, what each clock loses when probes go missing.

THE QUESTION ``probe_loss()`` CANNOT ANSWER. It reports how much of a clock is
absent -- probe count and coefficient mass. It cannot say how many years that
costs, or in which direction, and those are the numbers a reader needs. The
published finding is that EPICv2's probe loss "markedly disrupts traditional
DNAmAge clock outputs", with Hannum and GrimAge v2 over- or under-estimating
while the correlation with age stays high (Mech Ageing Dev 2025,
S0047637425001186). A correlation-based check passes; the intercept has moved.

HOW THIS MEASURES IT. Every platform's probe set is read off real arrays in the
test corpus rather than from a manifest, so what is measured is what a user with
that array actually has, filtering and all. Then, for each 450K dataset with
ages:

    score every clock on the full 450K matrix          -> reference
    mask to (450K INTERSECT target platform), re-score -> what the target sees
    shift = target - reference, per sample

and the per-clock summary is the median shift with a bootstrap interval. A clock
whose rotation spreads weight thinly -- the PC clocks -- should come out near
zero; a sparse elastic-net clock that lost a heavy probe should not.

The result is *reported*, never applied. An automatic offset would be a second
number nobody can trace, which is the failure the coefficient digests exist to
prevent.

Usage
-----
    python python/tools/build_platform_bias.py            # derive and write
    python python/tools/build_platform_bias.py --check    # CI: fail if stale
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))

CORPUS = ROOT / "test" / "data"
TARGET = (ROOT / "python" / "src" / "falconage" / "registry" / "data"
          / "platform_bias.csv")

#: Bootstrap resamples for the interval on the median shift. 2,000 is enough for
#: two decimal places on a median and costs under a second per clock.
N_BOOT = 2000
SEED = 20260809


def platform_probe_sets() -> dict[str, set[str]]:
    """Observed probe sets per platform, read off real arrays in the corpus."""
    import falconage as fa

    sets: dict[str, set[str]] = {}

    meta = pd.read_csv(CORPUS / "bench" / "computage_bench_meta.tsv", sep="\t",
                       index_col=0)
    # GPL13534 is the 450K; GPL21145 is EPIC v1. Read one dataset of each rather
    # than the union: a union across studies is the union of their filtering
    # decisions, not a platform.
    for gpl, name in (("GPL13534", "450K"), ("GPL21145", "EPICv1")):
        ids = meta.loc[meta["PlatformID"] == gpl, "DatasetID"].unique()
        best, n = None, -1
        for gse in ids:
            p = CORPUS / "bench" / f"{gse}.parquet"
            if not p.exists():
                continue
            cols = pd.read_parquet(p).shape[0]
            if cols > n:
                best, n = p, cols
        if best is not None:
            sets[name] = set(map(str, pd.read_parquet(best).index))

    v2 = CORPUS / "epicv2" / "GSE330325_series_matrix.txt.gz"
    if v2.exists():
        d = fa.prepare(fa.read_series_matrix(v2))
        sets["EPICv2"] = set(map(str, d.X.columns))
    return sets


def reference_datasets():
    """450K datasets with usable ages, as (name, FalconData) pairs."""
    import falconage as fa

    meta = pd.read_csv(CORPUS / "bench" / "computage_bench_meta.tsv", sep="\t",
                       index_col=0)
    out = []
    for gse in meta.loc[meta["PlatformID"] == "GPL13534", "DatasetID"].unique():
        p = CORPUS / "bench" / f"{gse}.parquet"
        if not p.exists():
            continue
        X = pd.read_parquet(p).T.astype("float64")
        obs = meta[meta["DatasetID"] == gse].reindex(X.index).rename(columns={
            "Age": "age", "Gender": "sex", "Condition": "condition",
            "DatasetID": "dataset", "Tissue": "tissue"})
        if pd.to_numeric(obs["age"], errors="coerce").notna().sum() < 10:
            continue
        out.append((gse, fa.prepare(fa.FalconData(X=X, obs=obs,
                                                  modality="dna_methylation"))))
    return out


def _boot_median(x: np.ndarray, rng) -> tuple[float, float]:
    if len(x) < 3:
        return float("nan"), float("nan")
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    meds = np.median(x[idx], axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def render() -> tuple[str, dict]:
    import falconage as fa

    rng = np.random.default_rng(SEED)
    probes = platform_probe_sets()
    if "450K" not in probes:
        raise SystemExit("no 450K dataset in the corpus; cannot build the table")
    # 450K is the reference: every measured shift is "what a 450K matrix becomes
    # when only the target platform's probes survive", so it is never a target.
    targets = {k: v for k, v in probes.items() if k != "450K"}

    per_clock: dict[tuple[str, str], list[float]] = {}
    n_datasets = 0
    for name, d in reference_datasets():
        n_datasets += 1
        full = fa.score(d, clocks="compatible", min_coverage=0.0)
        for plat, keep in targets.items():
            shared = [c for c in d.X.columns if str(c) in keep]
            if len(shared) < 1000:
                continue
            masked = fa.FalconData(X=d.X[shared], obs=d.obs,
                                   modality=d.modality, platform=plat)
            got = fa.score(masked, clocks=list(full.scores.columns),
                           min_coverage=0.0)
            for cid in got.scores.columns:
                if cid not in full.scores.columns:
                    continue
                delta = (got.scores[cid] - full.scores[cid]).dropna()
                per_clock.setdefault((cid, plat), []).extend(delta.tolist())

    reg = fa.registry.load()
    rows = []
    for (cid, plat), vals in sorted(per_clock.items()):
        v = np.asarray(vals, dtype=float)
        lo, hi = _boot_median(v, rng)
        c = reg.get(cid)
        kept = len(set(reg.feature_ids(cid)) & probes[plat])
        rows.append({
            "clock": cid, "platform": plat, "unit": ", ".join(c.unit) or "",
            "n_samples": len(v),
            "probes_retained": kept, "probes_total": len(reg.feature_ids(cid)),
            "median_shift": round(float(np.median(v)), 4),
            "iqr": round(float(np.subtract(*np.percentile(v, [75, 25]))), 4),
            "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
            "max_abs_shift": round(float(np.max(np.abs(v))), 4),
        })

    tab = pd.DataFrame(rows).sort_values(["platform", "clock"])
    buf = io.StringIO()
    buf.write("# Platform probe-loss bias, measured not assumed.\n")
    buf.write("# Each 450K dataset in the FALCONAge test corpus is scored twice: "
              "once whole,\n# once masked to the probes it shares with the target "
              "platform. The shift is\n# the difference, per sample, in the "
              "clock's own unit.\n")
    buf.write("# Probe sets are read off real arrays in the corpus, so they "
              "reflect what a user\n# with that platform has after ordinary "
              "filtering -- not the manifest's ideal.\n")
    buf.write(f"# datasets: {n_datasets}; bootstrap resamples: {N_BOOT}; "
              f"seed: {SEED}\n")
    buf.write("# Reported, never applied. See docs/science.qmd.\n")
    tab.to_csv(buf, index=False, lineterminator="\n")

    worst = tab.reindex(tab["median_shift"].abs().sort_values(ascending=False).index)
    return buf.getvalue(), {
        "rows": len(tab), "datasets": n_datasets,
        "platforms": sorted(targets),
        "worst": worst.head(5)[["clock", "platform", "median_shift",
                                "probes_retained", "probes_total"]],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)

    if not (CORPUS / "checksums.sha256").exists():
        print("test corpus absent; see test/data/README.md")
        return 1

    text, stats = render()
    current = TARGET.read_text(encoding="utf-8") if TARGET.exists() else None
    if args.check:
        if current != text:
            print(f"{TARGET.relative_to(ROOT)} is stale; run "
                  "python/tools/build_platform_bias.py")
            return 1
        print(f"{TARGET.relative_to(ROOT)} is current ({stats['rows']} rows)")
        return 0

    TARGET.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {TARGET.relative_to(ROOT)}: {stats['rows']} clock x platform "
          f"rows from {stats['datasets']} dataset(s), platforms "
          f"{stats['platforms']}")
    print("\nlargest shifts:")
    print(stats["worst"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
