#!/usr/bin/env python3
"""Build the per-CpG values an absent clock feature is filled with.

THE PROBLEM. A CpG the data does not carry still has to become a number before
a linear clock can weight it. Until these files existed, every clock filled it
with the mean of its *other* CpGs in the data, about 0.5 for most clocks. Many
clock CpGs sit near 0 or 1, and half a unit of beta times a large weight moves
an age by years; the pooled mean is also different in every dataset, which is
one reason two implementations of the same clock disagree by a constant.

WHAT THE AUTHORS PUBLISH. Four bundled clocks come with values for this, and
each is used as published:

    dunedinpoam38   the Dunedin training-cohort mean per probe, which the
                    authors' PoAmProjector() puts in place of a probe the data
                    lacks (R/sysdata.rda, mPOA_Models$model_means; GPL-2)
    corticalclock   the mean across 700 control cortical samples, which the
                    authors' CorticalClock.r adds for probes the data lacks
                    (PredCorticalAge/Ref_DNAm_brain_values.rdat)
    altumage        the centre of the RobustScaler published with the model,
                    the training median; the scaled value of a feature filled
                    with it is 0 (example_dependencies/scaler.pkl; MIT)
    horvath2013     goldstandard2 (Horvath 2013, Additional file 22; CC BY 2.0),
                    the per-probe beta the tutorial's normalisation calibrates
                    every sample towards. The tutorial stops on an absent probe
                    rather than filling it, so this is the published training
                    reference and not a published fill.

THE GOLD STANDARD ITSELF. ``horvath2013_goldstandard2.csv.gz`` is the whole
goldstandard2 vector, all 21,368 probes, which
``falconage.preprocess.horvath_normalise`` calibrates each sample to.

EVERY OTHER BLOOD CLOCK. The mean beta over healthy adult whole blood in the
test corpus: ComputAgeBench (CC BY-SA 4.0) samples labelled HC, specimen whole
blood, buffy coat or peripheral blood leukocytes, aged 18 or over, the same
selection as the conformal calibration (``build_conformal.WHOLE_BLOOD`` and
``MIN_AGE``). The mean and not the median, because a linear clock's expected
contribution from a feature is its weight times the feature's mean. A CpG
measured on fewer than MIN_N samples gets no value and falls back, with a
warning, like a clock that has no reference at all.

NOT EVERY CLOCK GETS ONE. A clock fitted on placenta, buccal epithelium or cord
blood has no counterpart in adult blood, and filling from it would replace one
wrong constant with another. Those keep the pooled fallback, and every run that
uses it says so.

Usage
-----
    python python/tools/build_references.py            # needs Rscript and the corpus
    python python/tools/build_references.py --check
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT = ROOT / "python" / "src" / "falconage" / "registry" / "data" / "references"
CORPUS = ROOT / "test" / "data"

#: The published sources, pinned to a commit or a publisher URL, with the
#: SHA-256 of the file as fetched. A mismatch stops the build.
SOURCES = {
    "horvath2013": {
        "url": ("https://static-content.springer.com/esm/art%3A10.1186%2Fgb-2013-14-10-r115/"
                "MediaObjects/13059_2013_3156_MOESM22_ESM.csv"),
        "sha256": "e4ea35396c429df1ea3ee64a3c0a012be16eba365ca5b8ae17328bbee13d2dd5",
    },
    "dunedinpoam38": {
        "url": ("https://raw.githubusercontent.com/danbelsky/DunedinPoAm38/"
                "42499971f4b9ff34e2eff7788843c2d337666370/R/sysdata.rda"),
        "sha256": "5a23fe91f371f74c60ae16c112c600bb9bc1824293d40c4c5ae1c2d688ccdab0",
    },
    "corticalclock": {
        "url": ("https://raw.githubusercontent.com/gemmashireby/CorticalClock/"
                "80c3df19c01d9aac25c9fd5aecd5bbc42aba0939/PredCorticalAge/"
                "Ref_DNAm_brain_values.rdat"),
        "sha256": "9c69f7ebc0b05db1a0960c56efd1ba7e4d127b1d13a8cbe38bac7d1cfc2aa8ab",
    },
    "altumage_scaler": {
        "url": ("https://raw.githubusercontent.com/rsinghlab/AltumAge/"
                "e34fd1bf5b9d6be087db6edc2588d7ff0b29aaf5/example_dependencies/scaler.pkl"),
        "sha256": "68331c0b8974c192460e2ebf31e4a974da3292775f82c52d54a11dea629ff53d",
    },
    "altumage_cpgs": {
        "url": ("https://raw.githubusercontent.com/rsinghlab/AltumAge/"
                "e34fd1bf5b9d6be087db6edc2588d7ff0b29aaf5/example_dependencies/"
                "multi_platform_cpgs.pkl"),
        "sha256": "068ed91ba02ec575262c7e9d0857eb9589224e478b8f2a9181c0fbcb0e26359d",
    },
}

#: Bundled clocks fitted on blood with no published per-CpG values. Each one's
#: registry entry points at blood_adult.csv.gz.
BLOOD_CLOCKS = (
    "cage", "dnamphenoage", "dnamtl", "hannum", "hrsinchphenoage", "lin",
    "mccartneyalcohol", "mccartneybmi", "mccartneybodyfat", "mccartneyeducation",
    "mccartneyhdlcholesterol", "mccartneyldlcholesterol", "mccartneysmoking",
    "mccartneytotalcholesterol", "mccartneytotalhdlratio", "mccartneywhr",
    "skinandblood", "vidalbralo", "weidner",
    "yingadaptage", "yingcausage", "yingdamage", "zhangmortality",
)
#: At 20 samples the standard error of a mean beta whose SD is 0.1 is 0.022, a
#: small fraction of what the pooled fill gets wrong on a CpG near 0 or 1.
MIN_N = 20


def fetch(name: str, cache: Path) -> Path:
    src = SOURCES[name]
    cache.mkdir(parents=True, exist_ok=True)
    p = cache / src["url"].rsplit("/", 1)[1]
    if not p.exists():
        print(f"  fetching {p.name}")
        with urllib.request.urlopen(src["url"], timeout=120) as r:
            p.write_bytes(r.read())
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    if digest != src["sha256"]:
        raise SystemExit(f"{p.name}: SHA-256 {digest}, expected {src['sha256']}")
    return p


def features(registry, cid: str) -> list[str]:
    return list(registry.feature_ids(cid))


def rdata_vector(path: Path, expr: str) -> pd.Series:
    """A named numeric vector out of an .rda file, through base R.

    Base R only, so any image with Rscript can run it; the values are printed
    with 15 significant digits, which is what a double carries.
    """
    code = (f'e <- new.env(); load("{path.as_posix()}", envir = e); x <- {expr}; '
            'cat(paste(names(x), sprintf("%.15g", x), sep = ","), sep = "\\n")')
    out = subprocess.run(["Rscript", "-e", code], check=True, capture_output=True,
                         text=True).stdout
    rows = [line.split(",") for line in out.splitlines() if line.strip()]
    return pd.Series({k: float(v) for k, v in rows})


def rdata_matrix(path: Path, expr: str) -> pd.DataFrame:
    """A numeric matrix with row and column names out of an .rda file, the same
    way as :func:`rdata_vector`: base R, 15 significant digits."""
    code = (f'e <- new.env(); load("{path.as_posix()}", envir = e); x <- {expr}; '
            'cat(paste(c("feature_id", colnames(x)), collapse = ","), "\\n", sep = ""); '
            'for (i in seq_len(nrow(x))) cat(paste(c(rownames(x)[i], '
            'sprintf("%.15g", x[i, ])), collapse = ","), "\\n", sep = "")')
    out = subprocess.run(["Rscript", "-e", code], check=True, capture_output=True,
                         text=True).stdout
    return pd.read_csv(io.StringIO(out), index_col="feature_id")


def horvath(registry, cache: Path) -> pd.Series:
    gs = pd.read_csv(fetch("horvath2013", cache), index_col="Name", dtype=str)["goldstandard2"]
    return gs.reindex(features(registry, "horvath2013"))


def goldstandard(cache: Path) -> pd.Series:
    gs = pd.read_csv(fetch("horvath2013", cache), index_col="Name", dtype=str)["goldstandard2"]
    gs.index.name = None
    return gs


def dunedin(registry, cache: Path) -> pd.Series:
    v = rdata_vector(fetch("dunedinpoam38", cache),
                     'e$mPOA_Models$model_means[["DunedinPoAm_38"]]')
    return v.reindex(features(registry, "dunedinpoam38"))


def cortical(registry, cache: Path) -> pd.Series:
    v = rdata_vector(fetch("corticalclock", cache), "e$ref")
    return v.reindex(features(registry, "corticalclock"))


def altumage(registry, cache: Path) -> pd.Series:
    from build_altumage_weights import Restricted

    class ScalerState:
        """Stands in for sklearn's RobustScaler: unpickling it only sets the
        fitted attributes, and the centre is all that is read."""

        def __setstate__(self, state):
            self.__dict__.update(state)

    class Reader(Restricted):
        def find_class(self, module, name):
            if (module, name) == ("sklearn.preprocessing._data", "RobustScaler"):
                return ScalerState
            return super().find_class(module, name)

    with open(fetch("altumage_scaler", cache), "rb") as fh:
        centre = np.asarray(Reader(fh).load().center_, dtype=np.float64)
    with open(fetch("altumage_cpgs", cache), "rb") as fh:
        cpgs = [str(c) for c in np.asarray(Reader(fh).load()).ravel()]
    if cpgs != features(registry, "altumage"):
        raise SystemExit("the published CpG order differs from the bundled model's")
    return pd.Series(centre, index=cpgs)


def blood_parts(registry) -> dict[str, tuple[pd.Series, pd.Series, int]]:
    """Per dataset: the sum and the count of healthy adult whole-blood betas per
    CpG, and the number of samples. Kept apart so a caller can leave one
    dataset out (``build_platform_bias.py`` does, per dataset it scores)."""
    from build_conformal import MIN_AGE, WHOLE_BLOOD

    want = sorted({f for cid in BLOOD_CLOCKS for f in features(registry, cid)})
    meta = pd.read_csv(CORPUS / "bench" / "computage_bench_meta.tsv", sep="\t")
    meta = meta[(meta["Condition"] == "HC") & meta["CellType"].isin(WHOLE_BLOOD)
                & (pd.to_numeric(meta["Age"], errors="coerce") >= MIN_AGE)]
    parts = {}
    for gse in sorted(meta["DatasetID"].unique()):
        p = CORPUS / "bench" / f"{gse}.parquet"
        if not p.exists():
            continue
        X = pd.read_parquet(p)
        ids = [s for s in meta.loc[meta["DatasetID"] == gse, "SampleID"] if s in X.columns]
        if not ids:
            continue
        sub = X.reindex(index=want)[ids].astype("float64")
        parts[gse] = (sub.sum(axis=1, skipna=True), sub.notna().sum(axis=1), len(ids))
    return parts


def blood(registry, parts=None, exclude: str | None = None) -> pd.DataFrame:
    """The mean per CpG over :func:`blood_parts`, optionally without one dataset."""
    parts = parts if parts is not None else blood_parts(registry)
    keep = {g: v for g, v in parts.items() if g != exclude}
    total = sum(s for s, _, _ in keep.values())
    n = sum(c for _, c, _ in keep.values())
    if exclude is None:
        print("  blood reference from "
              + ", ".join(f"{g} ({k})" for g, (_, _, k) in keep.items()))
    out = pd.DataFrame({"value": (total / n.where(n > 0)).round(6), "n": n.astype(int)})
    return out[out["n"] >= MIN_N]


def left_out(registry, parts, dataset: str):
    """The registry, with the corpus blood reference rebuilt without ``dataset``.

    For the builders that score this corpus (platform bias, conformal): the
    blood reference is a mean over some of the very samples they score, and a
    sample filled from its own value reports less error than a user's will. A
    shallow copy in which only ``reference_values`` differs, handed to
    ``fa.score(registry=...)``.
    """
    import copy

    ref = blood(registry, parts, exclude=dataset)["value"].to_dict()
    view = copy.copy(registry)
    view.reference_values = (lambda cid: ref if cid in BLOOD_CLOCKS
                             else registry.reference_values(cid))
    return view


def render(frame: pd.DataFrame | pd.Series, name: str) -> bytes:
    if isinstance(frame, pd.Series):
        if frame.isna().any():
            raise SystemExit(f"{name}: {int(frame.isna().sum())} feature(s) have no value")
        frame = frame.to_frame("value")
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["feature_id", *frame.columns])
    for fid, row in frame.iterrows():
        vals = []
        for col, v in row.items():
            if col == "n":
                vals.append(str(int(v)))
            elif isinstance(v, str):
                vals.append(v)
            else:
                vals.append(repr(float(v)))
        w.writerow([fid, *vals])
    data = buf.getvalue().encode("utf-8")
    if name.endswith(".gz"):
        # mtime 0 and no stored name, so the same table is the same bytes and
        # the SHA-256 in the registry does not move on a rebuild.
        out = io.BytesIO()
        with gzip.GzipFile(filename="", mode="wb", fileobj=out, mtime=0) as gz:
            gz.write(data)
        return out.getvalue()
    return data


def build(cache: Path) -> dict[str, bytes]:
    import falconage as fa

    reg = fa.registry.load()
    return {
        "horvath2013.csv": render(horvath(reg, cache), "horvath2013.csv"),
        "horvath2013_goldstandard2.csv.gz": render(goldstandard(cache),
                                                   "horvath2013_goldstandard2.csv.gz"),
        "dunedinpoam38.csv": render(dunedin(reg, cache), "dunedinpoam38.csv"),
        "corticalclock.csv": render(cortical(reg, cache), "corticalclock.csv"),
        "altumage.csv.gz": render(altumage(reg, cache), "altumage.csv.gz"),
        "blood_adult.csv.gz": render(blood(reg), "blood_adult.csv.gz"),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--cache", default=None, help="where to keep the downloads")
    args = ap.parse_args(argv)
    if not (CORPUS / "checksums.sha256").exists():
        print("test corpus absent; see test/data/README.md")
        return 1
    cache = Path(args.cache) if args.cache else Path.home() / ".cache" / "falconage-references"
    files = build(cache)
    stale = []
    for name, data in files.items():
        p = OUT / name
        digest = hashlib.sha256(data).hexdigest()
        if args.check:
            if not p.exists() or p.read_bytes() != data:
                stale.append(name)
            continue
        OUT.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        print(f"  wrote {name}: sha256 {digest}")
    if args.check:
        if stale:
            print(f"stale: {', '.join(stale)}; run python/tools/build_references.py")
            return 1
        print(f"{len(files)} reference files are current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
