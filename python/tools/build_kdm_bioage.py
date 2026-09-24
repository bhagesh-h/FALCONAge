"""Fit the Klemera-Doubal reference BioAge ships as ``kdm0``, and keep it.

WHAT THIS IS. When a paper reports "KDM biological age" on NHANES, it almost
always means BioAge's ``kdm0``: nine biomarkers, fitted separately by sex on
NHANES III adults aged 30 to 75 who were not pregnant, and projected into other
data with the training ``s_BA^2`` (Kwon and Belsky, GeroScience 2021;
``data-raw/nhanes_all.R``). Scoring a new cohort on that scale needs the
fitted parameters, not the NHANES rows, so this writes them to
``registry/data/kdm_bioage_nhanes3.json`` for :func:`kdm_bioage` to load.

The input is the NHANES III fixture the tests use
(``python/tests/data/nhanes3_kdm0_fixture.csv.gz``, provenance in its
``SOURCE.md``). Refitting it with FALCONAge's ``fit_kdm`` reproduces BioAge's
own ``kdm0`` to 0.0007 years, which is recorded in the output so the file
carries its own check.

Usage
-----
    python python/tools/build_kdm_bioage.py            # fit and write
    python python/tools/build_kdm_bioage.py --check    # fail if the file is stale
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))
FIXTURE = ROOT / "python" / "tests" / "data" / "nhanes3_kdm0_fixture.csv.gz"
TARGET = ROOT / "python" / "src" / "falconage" / "registry" / "data" / "kdm_bioage_nhanes3.json"

#: BioAge's kdm0 biomarkers, with the NHANES III units the fit is in.
MARKERS = {
    "fev": "mL (FEV1)",
    "sbp": "mmHg",
    "totchol": "mg/dL",
    "hba1c": "% (NGSP)",
    "albumin": "g/dL",
    "creat": "mg/dL",
    "lncrp": "log(1 + CRP), CRP in mg/dL",
    "alp": "U/L",
    "bun": "mg/dL",
}
SEXES = {1: "male", 2: "female"}          # NHANES coding


def build() -> dict:
    from falconage.models.clinical import fit_kdm, kdm

    d = pd.read_csv(FIXTURE)
    out = {
        "description": ("BioAge kdm0: Klemera-Doubal reference fitted by sex on NHANES III "
                        "aged 30 to 75, non-pregnant (Kwon and Belsky, GeroScience "
                        "2021;43:2795-2808; BioAge data-raw/nhanes_all.R)."),
        "source": {
            "fixture": "python/tests/data/nhanes3_kdm0_fixture.csv.gz",
            "fixture_sha256": hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
            "bioage_nhanes3_rda": "https://raw.githubusercontent.com/dayoonkwon/BioAge/master/data/NHANES3.rda",
        },
        "units": MARKERS,
        "sex_coding": "NHANES: 1 = male, 2 = female",
        "fits": {},
    }
    for code, sex in SEXES.items():
        rows = d[d["gender"] == code]
        ref = fit_kdm(rows, list(MARKERS))
        both = pd.DataFrame({"k": kdm(rows, ref, max_missing=0), "k0": rows["kdm0"]}).dropna()
        diff = (both["k"] - both["k0"]).abs()
        out["fits"][sex] = {
            "markers": ref.markers,
            "k": ref.k.tolist(), "q": ref.q.tolist(), "s": ref.s.tolist(), "r": ref.r.tolist(),
            "r_char": ref.r_char, "s_r": ref.s_r, "s_ba2": ref.s_ba2,
            "n_reference": ref.n_reference, "n_per_marker": ref.n_per_marker.tolist(),
            "age_range": list(ref.age_range),
            "ranges": ref.ranges.to_dict(orient="index"),
            "reproduces_kdm0": {"n": int(len(both)),
                                "mean_abs_diff_years": round(float(diff.mean()), 5),
                                "max_abs_diff_years": round(float(diff.max()), 5)},
        }
    return out


def render(obj: dict) -> str:
    """Stable text: sorted keys, fixed float precision, so --check compares bytes."""
    def rnd(x):
        if isinstance(x, float):
            return float(f"{x:.10g}")
        if isinstance(x, dict):
            return {k: rnd(v) for k, v in x.items()}
        if isinstance(x, list):
            return [rnd(v) for v in x]
        return x
    return json.dumps(rnd(obj), indent=1, sort_keys=True) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    text = render(build())
    if args.check:
        if not TARGET.exists() or TARGET.read_text() != text:
            print(f"stale: {TARGET.relative_to(ROOT)}; run python/tools/build_kdm_bioage.py")
            return 1
        print(f"current: {TARGET.relative_to(ROOT)}")
        return 0
    TARGET.write_text(text)
    fits = json.loads(text)["fits"]
    for sex, f in fits.items():
        rep = f["reproduces_kdm0"]
        print(f"{sex}: n={rep['n']} mean |kdm - kdm0| {rep['mean_abs_diff_years']} y, "
              f"max {rep['max_abs_diff_years']} y")
    print(f"wrote {TARGET.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
