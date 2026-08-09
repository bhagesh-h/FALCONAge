"""Calibrate distribution-free prediction intervals for the age clocks.

THE OTHER UNCERTAINTY. :mod:`falconage.uncertainty` answers "how much of this
score is the assay". This answers "how far from the truth is it likely to be" --
a different question with a different, larger answer, and the one a reader
assumes a number carries.

SPLIT CONFORMAL, IN THREE LINES. Take a calibration set of healthy samples with
known chronological age. Compute the absolute residuals. The
``ceil((n+1)(1-alpha))/n`` quantile of those residuals is a half-width that, on
any future sample exchangeable with the calibration set, contains the truth with
probability at least ``1-alpha``. No distributional assumption, finite-sample
guarantee (Vovk et al.; Romano, Patterson & Candes 2019).

Conformalised quantile regression is the better version -- it widens the
interval where the training data were sparse instead of using one width
everywhere (Genet Epidemiol 2025, 10.1002/gepi.70008). It needs fitted quantile
models per clock, which is a training step this package does not otherwise have.
Split conformal has the same coverage guarantee, so it is what ships, and the
width is reported per age band so a reader can see where the single width is
lying to them.

THE LIMIT, WHICH IS NOT SMALL. Coverage holds for data exchangeable with the
calibration cohort. These are public blood datasets, overwhelmingly of European
ancestry, mostly adult. On a paediatric cohort or a different population the
guarantee does not transfer, and saying so is more use than widening the
interval by an arbitrary factor.

Usage
-----
    python python/tools/build_conformal.py
    python python/tools/build_conformal.py --check
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
          / "conformal.csv")

LEVELS = (0.80, 0.90, 0.95)
#: Below this many calibration points the quantile is not an estimate. At n=20
#: the 95% level needs the largest residual in the set, which is one number.
MIN_CALIBRATION = 40
BANDS = ((0, 30), (30, 50), (50, 70), (70, 120))


def calibration_set():
    """Healthy blood samples with a chronological age, from the corpus.

    Healthy only. A conformal interval calibrated on a cohort half of whom have
    an aging-accelerating condition would quote the spread of the disease as if
    it were the clock's error.
    """
    import falconage as fa

    meta = pd.read_csv(CORPUS / "bench" / "computage_bench_meta.tsv", sep="\t",
                       index_col=0)
    frames = []
    for gse in meta["DatasetID"].unique():
        p = CORPUS / "bench" / f"{gse}.parquet"
        if not p.exists():
            continue
        X = pd.read_parquet(p).T.astype("float64")
        obs = meta[meta["DatasetID"] == gse].reindex(X.index).rename(columns={
            "Age": "age", "Gender": "sex", "Condition": "condition",
            "DatasetID": "dataset", "Tissue": "tissue"})
        keep = (obs["condition"].astype(str) == "HC")
        keep &= pd.to_numeric(obs["age"], errors="coerce").notna()
        if keep.sum() < 5:
            continue
        d = fa.prepare(fa.FalconData(X=X[keep.to_numpy()], obs=obs[keep.to_numpy()],
                                     modality="dna_methylation"))
        # The ordinary coverage floor, deliberately. Calibrating on a clock
        # whose probes are 40% imputed measures the imputation, and the width
        # would then be quoted at users whose arrays are fine.
        res = fa.score(d, clocks="compatible")
        long = res.scores.copy()
        long["age"] = pd.to_numeric(d.obs["age"], errors="coerce").to_numpy()
        long["tissue"] = d.obs["tissue"].astype(str).str.lower().to_numpy()
        long["dataset"] = gse
        frames.append(long)
    if not frames:
        raise SystemExit("no calibration samples found in the corpus")
    return pd.concat(frames, axis=0)


def render() -> tuple[str, dict]:
    import falconage as fa

    cal = calibration_set()
    reg = fa.registry.load()
    rows = []
    for cid in [c for c in cal.columns if c in {x.id for x in reg}]:
        c = reg.get(cid)
        if c.scale_type != "age_years":
            continue                      # a conformal band on a log-hazard is
            # not an age interval, and reporting one in years would invent units
        sub = cal[[cid, "age", "tissue"]].dropna()
        if len(sub) < MIN_CALIBRATION:
            continue
        resid = (sub[cid] - sub["age"]).to_numpy(dtype=float)
        n = len(resid)
        absr = np.abs(resid)
        for level in LEVELS:
            k = int(np.ceil((n + 1) * level))
            # k > n means the guarantee needs a sample larger than we have; the
            # honest half-width is then the maximum residual, flagged as such.
            q = float(np.sort(absr)[min(k, n) - 1])
            bias = float(np.median(resid))
            rows.append({
                "clock": cid, "level": level, "half_width": round(q, 4),
                "n_calibration": n,
                "exact": k <= n,
                "median_bias": round(bias, 4),
                "mae": round(float(np.mean(absr)), 4),
                # A clock offset from chronological age by more than its own
                # spread is not predicting age on this cohort. The interval is
                # still valid -- it covers the truth at the stated rate -- and
                # it is useless, because the bias is the message. Ying's DamAge
                # and AdaptAge are the clear cases: they are causality-
                # partitioned components reported on an age-like scale, not
                # age predictors, and this column is where that becomes visible.
                "usable": bool(abs(bias) <= q),
            })
        # Per band, so a reader can see where one width is too wide or too narrow.
        for lo, hi in BANDS:
            m = (sub["age"] >= lo) & (sub["age"] < hi)
            if m.sum() < 20:
                continue
            w = float(np.quantile(np.abs(resid[m.to_numpy()]), 0.90))
            bb = float(np.median(resid[m.to_numpy()]))
            rows.append({
                "clock": cid, "level": 0.90, "half_width": round(w, 4),
                "n_calibration": int(m.sum()), "exact": True,
                "median_bias": round(bb, 4),
                "mae": round(float(np.mean(np.abs(resid[m.to_numpy()]))), 4),
                "usable": bool(abs(bb) <= w),
                "age_band": f"{lo}-{hi}",
            })

    tab = pd.DataFrame(rows)
    tab["age_band"] = tab.get("age_band", pd.Series(dtype=object)).fillna("all")
    tab = tab[["clock", "age_band", "level", "half_width", "median_bias", "mae",
               "usable", "n_calibration", "exact"]].sort_values(
                   ["clock", "age_band", "level"])

    buf = io.StringIO()
    buf.write("# Split-conformal prediction intervals against chronological age.\n")
    buf.write("# half_width is the ceil((n+1)*level)/n quantile of the absolute\n"
              "# residual on the calibration set: healthy-control blood samples\n"
              "# with a recorded age, from the FALCONAge test corpus.\n")
    buf.write("# Coverage holds for data EXCHANGEABLE with that cohort. It is\n"
              "# public blood data, adult, and overwhelmingly of European ancestry;\n"
              "# on a paediatric or non-European cohort the guarantee does not\n"
              "# transfer, and no widening factor here would make it.\n")
    buf.write("# age_band 'all' is the shipped interval; the banded rows are\n"
              "# diagnostics showing where one width is too wide or too narrow.\n")
    tab.to_csv(buf, index=False, lineterminator="\n")

    allrows = tab[tab["age_band"] == "all"]
    return buf.getvalue(), {
        "clocks": allrows["clock"].nunique(), "rows": len(tab),
        "n": int(allrows["n_calibration"].max()) if len(allrows) else 0,
        "table": allrows[allrows["level"] == 0.90][
            ["clock", "half_width", "mae", "median_bias", "usable",
             "n_calibration"]],
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
                  "python/tools/build_conformal.py")
            return 1
        print(f"{TARGET.relative_to(ROOT)} is current ({stats['rows']} rows)")
        return 0

    TARGET.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {TARGET.relative_to(ROOT)}: {stats['clocks']} clock(s), "
          f"{stats['rows']} rows, up to n={stats['n']} calibration samples")
    print("\n90% half-widths (years):")
    print(stats["table"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
