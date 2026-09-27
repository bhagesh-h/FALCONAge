"""Stage 1: one complete input matrix that every package scores.

WHY COMPLETE. Item 19 of the audit found that every difference between
FALCONAge, methylclock and dnaMethyAge on the same betas came from how each
fills a CpG the array does not carry: methylclock drops it (a fill of 0),
dnaMethyAge takes a mean from its own reference, FALCONAge a per-CpG
reference. Those are policies, not errors, and a comparison that includes them
measures the policies. So the probes GSE182991 lacks are filled here, once,
with FALCONAge's reference value where the clock has one and the cohort mean of
the present probes otherwise, and every package receives the same numbers. What
is left to differ is the clocks themselves.

Writes, under test/conformance/out/: betas.csv (CpGs x samples), pheno.csv
(sample, age), and filled.csv (how many values were filled, per CpG).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import falconage as fa

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
ROOT = HERE.parents[1]


def main() -> int:
    pairs = yaml.safe_load((HERE / "pairs.yaml").read_text())["clocks"]
    d = fa.read_computage_bench(pairs_dataset(), root=str(ROOT / "test" / "data" / "bench"))
    reg = fa.registry.load()
    X = d.X
    cols: dict[str, pd.Series] = {}
    filled = {}
    for cid in pairs:
        feats = list(reg.feature_ids(cid))
        ref = reg.reference_values(cid) or {}
        for f in feats:
            if f in cols:
                continue
            if f in X.columns and X[f].notna().any():
                v = X[f].astype(float)
                if v.isna().any():
                    filled[f] = int(v.isna().sum())
                    v = v.fillna(v.mean())
            else:
                v = pd.Series(ref.get(f, np.nan), index=X.index, dtype=float)
                filled[f] = len(X)
            cols[f] = v
    M = pd.DataFrame(cols)
    pooled = M.stack().mean()
    M = M.fillna(pooled)                       # CpGs with no reference either
    OUT.mkdir(exist_ok=True)
    M.T.to_csv(OUT / "betas.csv", float_format="%.10g")
    pd.DataFrame({"sample": d.X.index, "age": pd.to_numeric(d.obs["age"], errors="coerce")}).to_csv(
        OUT / "pheno.csv", index=False)
    pd.Series(filled, name="n_filled").rename_axis("cpg").to_csv(OUT / "filled.csv")
    print(f"{M.shape[1]} CpGs x {M.shape[0]} samples; {len(filled)} CpGs filled for "
          f"at least one sample")
    return 0


def pairs_dataset() -> str:
    return yaml.safe_load((HERE / "pairs.yaml").read_text())["dataset"]


if __name__ == "__main__":
    sys.exit(main())
