"""Stage 3: FALCONAge against every other package, clock by clock.

Scores the matrix prepare.py wrote with FALCONAge (no tissue declared, so that
nothing is refused: this checks arithmetic, not applicability), joins the
other packages' scores, and fails when a difference is larger than the
tolerance and not listed under ``explained`` in pairs.yaml. Writes
out/report.csv, one row per clock and package.
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


def main() -> int:
    cfg = yaml.safe_load((HERE / "pairs.yaml").read_text())
    pairs, tol = cfg["clocks"], float(cfg["tolerance"])
    explained = {(e["clock"], e["package"]): e for e in cfg.get("explained") or []}
    betas = pd.read_csv(OUT / "betas.csv", index_col=0)
    pheno = pd.read_csv(OUT / "pheno.csv", index_col=0)
    d = fa.FalconData(X=betas.T, obs=pheno, modality="dna_methylation", platform="EPICv1")
    scores, skipped = {}, {}
    for cid in pairs:
        try:
            scores[cid] = fa.score(d, clocks=[cid], min_coverage=0.0).scores[cid]
        except Exception as exc:  # noqa: BLE001
            skipped[cid] = f"{type(exc).__name__}: {exc}".splitlines()[0][:200]
    others = []
    for f in ("r_packages.csv", "biolearn.csv"):
        if (OUT / f).exists() and (OUT / f).stat().st_size > 1:
            others.append(pd.read_csv(OUT / f))
    other = pd.concat(others) if others else pd.DataFrame(
        columns=["package", "clock", "name", "sample", "value"])
    rows, bad = [], 0
    for (cid, pkg), g in other.groupby(["clock", "package"]):
        name = g["name"].iloc[0]
        o = g.set_index("sample")["value"].astype(float)
        if cid not in scores:
            status, note = "FALCONAge did not score", skipped.get(cid, "")
            diff = pd.Series(dtype=float)
        else:
            f = scores[cid].reindex(o.index)
            diff = (f - o).dropna()
            ex = explained.get((cid, pkg))
            if diff.empty:
                status, note = "no overlap", ""
            elif diff.abs().max() <= tol:
                status, note = "agrees", ""
            elif ex:
                status, note = "explained", ex["reason"]
            else:
                status, note = "UNEXPLAINED", ""
                bad += 1
        rows.append({"clock": cid, "package": pkg, "their_name": name, "n": len(diff),
                     "max_abs_diff": float(diff.abs().max()) if len(diff) else np.nan,
                     "mean_diff": float(diff.mean()) if len(diff) else np.nan,
                     "r": float(np.corrcoef(scores[cid].reindex(o.index).loc[diff.index], o.loc[diff.index])[0, 1])
                     if len(diff) > 2 else np.nan,
                     "status": status, "note": note})
    rep = pd.DataFrame(rows).sort_values(["status", "clock", "package"])
    rep.to_csv(OUT / "report.csv", index=False, float_format="%.6g")
    pd.set_option("display.width", 200, "display.max_colwidth", 60, "display.max_rows", 500)
    print(rep.drop(columns=["note"]).to_string(index=False))
    for f in ("r_errors.csv", "biolearn_errors.csv"):
        if (OUT / f).exists():
            e = pd.read_csv(OUT / f)
            if len(e):
                print(f"\n{f}:\n" + e.to_string(index=False))
    print(f"\n{(rep.status == 'agrees').sum()} agree, {(rep.status == 'explained').sum()} explained, "
          f"{bad} unexplained, of {len(rep)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
