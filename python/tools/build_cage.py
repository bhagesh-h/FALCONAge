#!/usr/bin/env python3
"""Extract cAge's two models from the paper's supplement.

Bernabeu et al. 2023, "Refining epigenetic prediction of chronological and
biological age", Genome Medicine 15:12 (CC BY 4.0). Additional file 4 carries
the models as tables:

    S8   cAge, an elastic net on age: intercept and weights
    S9   the same on log(age)

A term with the suffix ``_2`` is the square of that CpG's beta. Written to
``registry/data/coefficients/cage.csv`` with the columns ``feature_id,
coefficient, log_coefficient`` (the S8 and S9 weights; blank where a term is
in one model only), the intercept as the first row, and every value as the
workbook stores it.

Usage
-----
    python python/tools/build_cage.py            # needs openpyxl
    python python/tools/build_cage.py --check
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "python" / "src" / "falconage" / "registry" / "data" / "coefficients" / "cage.csv"
URL = ("https://static-content.springer.com/esm/art%3A10.1186%2Fs13073-023-01161-y/"
       "MediaObjects/13073_2023_1161_MOESM4_ESM.xlsx")
SHA256 = "537352f34f98423c9eea7119592fc30d4b0a32ec33cbc956ad748ae2ae5af76f"


def fetch(cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    p = cache / URL.rsplit("/", 1)[1]
    if not p.exists():
        req = urllib.request.Request(URL, headers={"User-Agent": "falconage-build"})
        with urllib.request.urlopen(req, timeout=300) as r:
            p.write_bytes(r.read())
    got = hashlib.sha256(p.read_bytes()).hexdigest()
    if got != SHA256:
        raise SystemExit(f"{p.name}: SHA-256 {got}, expected {SHA256}")
    return p


def table(wb, sheet: str) -> dict[str, float]:
    rows = list(wb[sheet].iter_rows(values_only=True))
    start = next(i for i, r in enumerate(rows) if r[0] == "Variable" and r[1] == "Coefficient")
    out: dict[str, float] = {}
    for r in rows[start + 1:]:
        if r[0] is None:
            continue
        name = "(Intercept)" if str(r[0]).strip() == "Intercept" else str(r[0]).strip()
        if name in out:
            raise SystemExit(f"{sheet}: {name} appears twice")
        out[name] = r[1]
    return out


def build(cache: Path) -> bytes:
    import openpyxl

    wb = openpyxl.load_workbook(fetch(cache), read_only=True)
    lin, log = table(wb, "S8"), table(wb, "S9")
    terms = ["(Intercept)"] + [t for t in dict.fromkeys([*lin, *log]) if t != "(Intercept)"]
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["feature_id", "coefficient", "log_coefficient"])
    for t in terms:
        w.writerow([t, *(repr(float(m[t])) if t in m else "" for m in (lin, log))])
    return buf.getvalue().encode("utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--cache", default=str(Path.home() / ".cache" / "falconage-build"))
    args = ap.parse_args(argv)
    data = build(Path(args.cache))
    if args.check:
        if not OUT.exists() or OUT.read_bytes() != data:
            print(f"{OUT.name} is stale; run python/tools/build_cage.py")
            return 1
        print(f"{OUT.name} is current")
        return 0
    OUT.write_bytes(data)
    print(f"wrote {OUT.name}: sha256 {hashlib.sha256(data).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
