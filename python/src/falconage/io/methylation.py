"""Readers for the formats methylation data actually arrives in.

Four paths, in descending order of how often you meet them:

1. a GEO **series matrix** -- metadata and betas in one gzipped text file, which
   is what roughly 60% of GEO methylation series publish and nothing else;
2. a plain **beta matrix** as CSV, TSV or parquet, orientation unknown;
3. **IDATs**, the only path where the normalisation is ours rather than
   somebody else's;
4. **RRBS** coverage files, where a beta is a ratio of counts and the counts
   matter.

Every one of them returns the same :class:`~falconage.core.container.FalconData`,
because the scoring loop must not be able to tell which it came from.
"""

from __future__ import annotations

import gzip
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.container import FalconData
from ..core.errors import DataError

#: Probe counts per Illumina platform, used to identify one from a matrix whose
#: header says nothing. The windows are generous because real matrices are
#: filtered: a 450K series that dropped its cross-reactive probes still has
#: 450,000-ish, and nothing else is anywhere near that.
PLATFORM_SIZES: list[tuple[str, int, int]] = [
    ("27K", 20_000, 30_000),
    ("450K", 300_000, 500_000),
    ("EPICv1", 700_000, 900_000),
    ("EPICv2", 900_000, 1_000_000),
    ("MammalMethylChip40", 30_000, 40_000),
]


def _open_text(path: Path):
    if str(path).endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def detect_platform(features) -> str | None:
    """Guess the array from the probe identifiers and how many there are.

    Identifier shape first, count second. EPIC v2 is the only platform whose
    probes carry a suffix (``cg00000029_TC21``), and the mammalian array is the
    only one with ~37k probes, so both are identifiable outright. 27K, 450K and
    EPIC v1 share a namespace and can only be told apart by count -- which is a
    guess, and is why this returns ``None`` rather than a wrong answer when the
    count falls between the windows.
    """
    feats = [str(f) for f in features]
    if not feats:
        return None
    if sum(bool(re.match(r"^cg\d+_[A-Z]{2}\d+$", f)) for f in feats[:2000]) > 50:
        return "EPICv2"
    if sum(f.startswith(("cg", "ch.", "rs")) for f in feats[:2000]) < 100:
        if sum(":" in f for f in feats[:200]) > 100:
            return "RRBS"
        return None
    n = len(feats)
    for name, lo, hi in PLATFORM_SIZES:
        if lo <= n <= hi:
            return name
    return None


def read_betas(path: str | Path, *, samples_are: str = "auto",
               obs: pd.DataFrame | None = None) -> FalconData:
    """Read a beta matrix from CSV, TSV or parquet.

    Parameters
    ----------
    samples_are
        ``"rows"``, ``"columns"``, or ``"auto"``. Auto looks at which axis
        carries probe-shaped identifiers, which is reliable precisely because
        ``cg`` ids are unmistakable -- guessing from the shape alone would get a
        1000-sample 450K matrix right and a 500,000-probe cohort wrong.
    """
    p = Path(path)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    else:
        sep = "\t" if p.name.endswith((".tsv", ".tsv.gz", ".txt", ".txt.gz")) else ","
        df = pd.read_csv(p, sep=sep, index_col=0, low_memory=False)

    if samples_are == "auto":
        rows_are_probes = sum(str(i).startswith("cg") for i in df.index[:200]) > 100
        cols_are_probes = sum(str(c).startswith("cg") for c in df.columns[:200]) > 100
        if rows_are_probes and not cols_are_probes:
            samples_are = "columns"
        elif cols_are_probes and not rows_are_probes:
            samples_are = "rows"
        else:
            # Fall back on the shape, and say so: arrays have far more probes
            # than any cohort has samples.
            samples_are = "columns" if df.shape[0] > df.shape[1] else "rows"

    if samples_are == "columns":
        df = df.T

    df = df.apply(pd.to_numeric, errors="coerce")
    _check_beta_range(df)
    plat = detect_platform(df.columns)
    return FalconData(X=df, obs=obs if obs is not None else pd.DataFrame(index=df.index),
                      modality="dna_methylation", platform=plat)


def _check_beta_range(df: pd.DataFrame) -> None:
    v = df.to_numpy(dtype=np.float64)
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        raise DataError("the matrix contains no finite values")
    lo, hi = float(finite.min()), float(finite.max())
    if lo < -0.01 or hi > 1.01:
        raise DataError(
            f"values span [{lo:.3g}, {hi:.3g}], which is not the beta range.\n"
            "  If these are M-values, convert with "
            "falconage.models.ops.m_to_beta first -- every clock in the registry "
            "was fitted on beta, and scoring M-values silently returns a number "
            "roughly twice as far from the mean as it should be."
        )


def read_series_matrix(path: str | Path, *, characteristics: bool = True) -> FalconData:
    """Read a GEO ``*_series_matrix.txt.gz``: metadata and betas in one file.

    The characteristics block is parsed into ``obs``. GEO stores it as one row
    per characteristic with a ``"key: value"`` string per sample, and different
    submitters use different keys for the same thing, so the keys are kept
    verbatim rather than mapped onto a schema -- an ``age`` column that silently
    turned out to be ``age at diagnosis`` is worse than no column.
    """
    p = Path(path)
    meta_rows: dict[str, list[str]] = {}
    sample_ids: list[str] = []
    titles: list[str] = []
    platform: str | None = None
    table_lines: list[str] = []
    in_table = False

    with _open_text(p) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            if line.startswith("!series_matrix_table_end"):
                break
            if in_table:
                table_lines.append(line)
                continue
            if line.startswith("!Sample_geo_accession"):
                sample_ids = [c.strip('"') for c in line.split("\t")[1:]]
            elif line.startswith("!Sample_title"):
                titles = [c.strip('"') for c in line.split("\t")[1:]]
            elif line.startswith("!Series_platform_id"):
                platform = line.split("\t")[1].strip('"')
            elif characteristics and line.startswith("!Sample_characteristics_ch1"):
                vals = [c.strip('"') for c in line.split("\t")[1:]]
                key = None
                for v in vals:
                    if ": " in v:
                        key = v.split(": ", 1)[0].strip()
                        break
                if key is None:
                    continue
                key = re.sub(r"[^0-9a-zA-Z]+", "_", key).strip("_").lower()
                # A series can repeat a characteristic key; suffix rather than
                # overwrite, because the second one is usually a different thing.
                base, i = key, 2
                while key in meta_rows:
                    key = f"{base}_{i}"
                    i += 1
                meta_rows[key] = [v.split(": ", 1)[1].strip() if ": " in v else ""
                                  for v in vals]

    # A metadata-only series matrix is normal, not broken. Sequencing series and
    # the Horvath mammalian series publish their values as a supplementary file
    # and leave the matrix block empty; the metadata is then the whole payload,
    # and refusing to read it would mean no way to get the ages.
    body = [ln for ln in table_lines if ln.strip()]
    metadata_only = len(body) < 2

    if metadata_only:
        n = len(sample_ids) or len(titles)
        idx = pd.Index(sample_ids or [f"s{i}" for i in range(n)], name="sample_id")
        X = pd.DataFrame(index=idx)
    else:
        tbl = pd.read_csv(io.StringIO("\n".join(body)), sep="\t", index_col=0,
                          low_memory=False)
        tbl.columns = [str(c).strip('"') for c in tbl.columns]
        tbl.index = [str(i).strip('"') for i in tbl.index]
        X = tbl.apply(pd.to_numeric, errors="coerce").T  # matrices are probes x samples

    obs = pd.DataFrame(index=X.index)
    if sample_ids and len(sample_ids) == len(X.index):
        obs.index = pd.Index(sample_ids, name="sample_id")
        X.index = obs.index
    if titles and len(titles) == len(obs):
        obs["title"] = titles
    for k, vals in meta_rows.items():
        if len(vals) == len(obs):
            obs[k] = vals

    obs = _coerce_obs(obs)
    uns = {"gpl": platform or "", "source_file": p.name, "metadata_only": metadata_only}
    if metadata_only:
        uns["note"] = ("no value matrix in this series matrix; the betas are in "
                       "the series' suppl/ directory")
    else:
        _check_beta_range(X)
    return FalconData(X=X, obs=obs, modality="dna_methylation",
                      platform=_platform_from_gpl(platform)
                      or (detect_platform(X.columns) if not metadata_only else None),
                      uns=uns)


_GPL = {
    "GPL8490": "27K", "GPL13534": "450K", "GPL16304": "450K",
    "GPL21145": "EPICv1", "GPL23976": "EPICv1", "GPL29753": "EPICv1",
    "GPL33022": "EPICv2", "GPL34394": "MSA",
    "GPL28271": "MammalMethylChip40",
}


def _platform_from_gpl(gpl: str | None) -> str | None:
    return _GPL.get((gpl or "").strip()) if gpl else None


def _coerce_obs(obs: pd.DataFrame) -> pd.DataFrame:
    """Numeric-looking metadata becomes numeric; sex is normalised; nothing else.

    Deliberately conservative. Turning ``"gestational_age_days"`` into weeks, or
    ``"age (years)"`` that is really months into years, is exactly the inference
    the units module exists to refuse -- so the column keeps its published name
    and its published number and the user converts on purpose.
    """
    out = obs.copy()
    for c in out.columns:
        if out[c].dtype == object:
            conv = pd.to_numeric(out[c], errors="coerce")
            if conv.notna().mean() > 0.8:
                out[c] = conv
    for cand in ("sex", "gender", "Sex", "Gender"):
        if cand in out.columns:
            s = out[cand].astype(str).str.strip().str.lower()
            out["sex"] = s.map(lambda v: "F" if v.startswith(("f", "wom")) else
                               ("M" if v.startswith(("m", "man")) else "U"))
            break
    return out


def read_idat_pair(grn: str | Path, red: str | Path) -> dict[str, np.ndarray]:
    """Parse one Grn/Red IDAT pair into raw intensities keyed by address.

    Returns the address ids and the two channels. Turning addresses into probe
    identifiers needs the platform manifest, which Illumina distributes
    separately and which FALCONAge does not vendor -- see
    :func:`falconage.preprocess.methylation.idat_to_betas` for that step and why
    it is a separate one.
    """
    return {"illumina_ids": _read_idat(grn)["IlluminaID"],
            "grn": _read_idat(grn)["Mean"],
            "red": _read_idat(red)["Mean"]}


_IDAT_FIELDS = {1000: "nSNPsRead", 102: "IlluminaID", 103: "SD", 104: "Mean", 107: "NBeads"}


def _read_idat(path: str | Path) -> dict[str, np.ndarray]:
    """Minimal IDAT reader: the four fields a beta needs, nothing else.

    Illumina's format is undocumented but stable and widely reimplemented. Only
    the field table, the bead ids, the means and the bead counts are read --
    the run metadata blocks are skipped, which is what keeps this to eighty
    lines instead of the eight hundred a full parser takes.
    """
    p = Path(path)
    raw = gzip.open(p, "rb") if p.suffix == ".gz" else open(p, "rb")
    with raw as fh:
        buf = fh.read()

    if buf[:4] != b"IDAT":
        raise DataError(f"{p.name} is not an IDAT (magic is {buf[:4]!r})")
    version = int.from_bytes(buf[4:12], "little")
    if version != 3:
        raise DataError(f"{p.name}: IDAT version {version}, only version 3 is supported")

    n_fields = int.from_bytes(buf[12:16], "little")
    offsets: dict[int, int] = {}
    pos = 16
    for _ in range(n_fields):
        code = int.from_bytes(buf[pos:pos + 2], "little")
        off = int.from_bytes(buf[pos + 2:pos + 10], "little")
        offsets[code] = off
        pos += 10

    n = int.from_bytes(buf[offsets[1000]:offsets[1000] + 4], "little")
    out: dict[str, np.ndarray] = {"nSNPsRead": np.array([n])}
    for code, name in ((102, "IlluminaID"), (104, "Mean"), (103, "SD"), (107, "NBeads")):
        if code not in offsets:
            continue
        o = offsets[code]
        if name == "IlluminaID":
            out[name] = np.frombuffer(buf, dtype="<i4", count=n, offset=o)
        elif name == "NBeads":
            out[name] = np.frombuffer(buf, dtype="<u1", count=n, offset=o)
        else:
            out[name] = np.frombuffer(buf, dtype="<u2", count=n, offset=o).astype(np.float64)
    return out


def read_rrbs(path: str | Path, *, min_coverage: int = 5,
              sample_id: str | None = None) -> pd.Series:
    """Read one RRBS site file into coverage-filtered betas.

    Handles the two layouts these files actually come in:

    * a header naming a ``...Percentage`` and a ``...Coverage`` column, keyed by
      a site identifier in the first column -- the Petkovich/GSE80672 layout;
    * four headerless columns of chromosome, position, methylated count and
      total count, which is what most bisulfite pipelines emit.

    Sites below ``min_coverage`` are dropped, not kept as noisy zeros and ones.
    A ratio from four reads and a ratio from four hundred are not the same
    measurement, and a clock handed both without distinction reports the
    sequencing depth as if it were biology. This is the one preprocessing
    decision an array pipeline never has to make.

    Site identifiers are kept exactly as published. GSE80672 keys on NCBI ``gi``
    accessions rather than ``chr:pos``; remapping them would need an assembly
    and a decision about which one, and inventing that silently is how a
    coordinate-keyed clock ends up matching nothing while looking like it
    matched something.
    """
    p = Path(path)
    comp = "gzip" if p.suffix == ".gz" else None
    head = pd.read_csv(p, sep="\t", nrows=2, compression=comp)
    cols = [str(c) for c in head.columns]
    named = any(c.lower().endswith(("percentage", "coverage", "beta", "count"))
                for c in cols)

    if named:
        df = pd.read_csv(p, sep="\t", index_col=0, low_memory=False, compression=comp)
        pct = next((c for c in df.columns
                    if str(c).lower().endswith(("percentage", "beta", "ratio"))), None)
        cov = next((c for c in df.columns
                    if str(c).lower().endswith(("coverage", "depth", "total"))), None)
        if pct is None or cov is None:
            raise DataError(
                f"{p.name}: header is {cols}; expected a percentage/beta column and "
                "a coverage/depth column")
        beta = pd.to_numeric(df[pct], errors="coerce")
        depth = pd.to_numeric(df[cov], errors="coerce")
        # Percentages, not fractions -- 0-100 in this layout.
        if float(np.nanmax(beta.to_numpy())) > 1.5:
            beta = beta / 100.0
        idx = df.index.astype(str)
    else:
        df = pd.read_csv(p, sep="\t", header=None, low_memory=False, compression=comp)
        if df.shape[1] < 4:
            raise DataError(
                f"{p.name}: {df.shape[1]} headerless columns; expected chromosome, "
                "position, methylated count, total count")
        meth = pd.to_numeric(df.iloc[:, 2], errors="coerce")
        depth = pd.to_numeric(df.iloc[:, 3], errors="coerce")
        if (meth > depth).mean() > 0.5:      # columns the other way round
            meth, depth = depth, meth
        beta = meth / depth.replace(0, np.nan)
        idx = (df.iloc[:, 0].astype(str).str.replace("^chr", "", regex=True) + ":"
               + pd.to_numeric(df.iloc[:, 1], errors="coerce").astype("Int64").astype(str))

    keep = depth.to_numpy() >= min_coverage
    s = pd.Series(np.clip(beta.to_numpy()[keep], 0.0, 1.0),
                  index=pd.Index(np.asarray(idx)[keep], name="site"),
                  name=sample_id or p.stem.split("_")[0])
    return s[~s.index.duplicated()]


def read_rrbs_dir(paths, *, min_coverage: int = 5,
                  obs: pd.DataFrame | None = None) -> FalconData:
    """Assemble several RRBS files into one matrix on their shared sites."""
    series = []
    for p in paths:
        sid = Path(p).name.split("_")[0]
        series.append(read_rrbs(p, min_coverage=min_coverage, sample_id=sid))
    X = pd.concat(series, axis=1).T
    return FalconData(X=X, obs=obs if obs is not None else pd.DataFrame(index=X.index),
                      modality="rrbs", platform="RRBS",
                      uns={"min_coverage": min_coverage, "n_files": len(series)})
