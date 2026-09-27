#!/usr/bin/env python3
"""Extract LinAge2's fitted model by running the authors' code.

LinAge2 (Fong et al. 2025, "LinAge2: providing actionable insights and
benchmarking with epigenetic clocks", npj Aging 11:29; CC BY 4.0) is published
as code rather than as a table: ``linAge2.R`` in Supplementary Information
(a zip) fits the clock on the NHANES 1999-2002 data it ships with every time
it runs. The fit is deterministic, so it is run once here and what it fitted
is written to ``registry/data/linage2/linage2.json``:

    features        the 59 inputs, in the order the projection expects
    lambda          the log / Box-Cox transform per feature (logNoLog.csv)
    normalisation   median and MAD per sex in the 1999-2000 wave aged 40 to 50,
                    or none for the five features the authors leave raw
    z_max           the fold-in limit for z-scores (6)
    male, female    the Cox model's terms, coefficients and centring means, the
                    null model's age coefficient and mean, and the loadings of
                    each PC the model uses (vMatDat99_*_pre.csv)

It also writes the test fixture: the example user data and sanity samples from
the archive, and the script's own LinAge2 for each.

Usage
-----
    python python/tools/build_linage2.py            # needs Rscript and survival
    python python/tools/build_linage2.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "python" / "src" / "falconage" / "registry" / "data" / "linage2"
FIXTURES = ROOT / "python" / "tests" / "data"
URL = ("https://static-content.springer.com/esm/art%3A10.1038%2Fs41514-025-00221-4/"
       "MediaObjects/41514_2025_221_MOESM1_ESM.zip")
SHA256 = "e986cdfc583f78753a2a7c759e3452ce6e87c54d1844bc7f2be3021b3e3b642e"


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


def build(cache: Path, out: Path, fixtures: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(fetch(cache)) as z:
            z.extractall(tmp)
        src = Path(tmp) / "linAge2_code"
        out.mkdir(parents=True, exist_ok=True)
        subprocess.run(["Rscript", str(Path(__file__).with_suffix(".R")), str(src), str(out),
                        str(fixtures)], check=True, stdout=subprocess.DEVNULL)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--cache", default=str(Path.home() / ".cache" / "falconage-build"))
    args = ap.parse_args(argv)
    if args.check:
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            (t / "fx").mkdir()
            build(Path(args.cache), t / "out", t / "fx")
            stale = [p.name for p in (t / "out").glob("linage2.json")
                     if (OUT / p.name).read_bytes() != p.read_bytes()]
            stale += [p.name for p in (t / "fx").iterdir()
                      if (FIXTURES / p.name).read_bytes() != p.read_bytes()]
        if stale:
            print(f"stale: {', '.join(stale)}; run python/tools/build_linage2.py")
            return 1
        print("linage2.json and its fixtures are current")
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        build(Path(args.cache), Path(tmp), FIXTURES)
        OUT.mkdir(parents=True, exist_ok=True)
        data = (Path(tmp) / "linage2.json").read_bytes()
        (OUT / "linage2.json").write_bytes(data)
        print(f"wrote linage2.json: sha256 {hashlib.sha256(data).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
