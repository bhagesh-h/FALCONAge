"""Readers and writers. Every reader returns a FalconData."""

from pathlib import Path

import pandas as pd

from ..core.container import FalconData
from ..core.errors import DataError
from .bench import computage_bench_meta, list_computage_bench, read_computage_bench
from .methylation import (
    detect_platform,
    read_bedmethyl,
    read_bedmethyl_dir,
    read_betas,
    read_idat_pair,
    read_panel,
    read_rrbs,
    read_rrbs_dir,
    read_series_matrix,
)

__all__ = [
    "computage_bench_meta", "detect_platform", "list_computage_bench",
    "read", "read_bedmethyl", "read_bedmethyl_dir",
    "read_betas", "read_clinical", "read_computage_bench", "read_idat_pair",
    "read_nightingale", "read_panel",
    "read_rrbs", "read_rrbs_dir", "read_series_matrix", "write_results",
]


def _units(units, path) -> dict:
    """Validate the units argument before pandas turns it into a puzzle.

    `units="SI"` is the obvious thing to try, and it used to fail as
    "dictionary update sequence element #0 has length 1; 2 is required" from
    inside dict(), which names neither the argument nor the file. The units are
    per column because a table mixes them -- albumin in g/L beside creatinine
    in umol/L -- so there is no single name to accept, and saying so is the
    whole job of this function.
    """
    if units is None:
        return {}
    if isinstance(units, str):
        raise DataError(
            f"read_clinical({str(path)!r}, units={units!r}): units are declared "
            "per column, not per file.\n"
            "  A clinical table mixes them: albumin in g/L beside creatinine in "
            "umol/L beside glucose in mmol/L.\n"
            "  Pass a mapping, for example:\n"
            "    units={'albumin': 'g/L', 'creatinine': 'umol/L', "
            "'glucose': 'mmol/L'}\n"
            "  falconage.core.units.require_units() lists the exact keys each "
            "clinical clock needs.")
    try:
        return dict(units)
    except (TypeError, ValueError) as exc:
        raise DataError(
            f"read_clinical({str(path)!r}): units must be a mapping of column "
            f"name to unit; got {type(units).__name__}.") from exc


def read_clinical(path: str | Path, *, units: dict[str, str] | None = None,
                  index_col: int | str = 0) -> FalconData:
    """Read a clinical chemistry table.

    ``units`` is not optional in effect: the clinical models call
    :func:`falconage.core.units.require_units`, which raises with the exact dict
    to supply. It is accepted as ``None`` here so that reading a file to look at
    it does not require knowing the units first.
    """
    p = Path(path)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    else:
        sep = "\t" if p.name.endswith((".tsv", ".txt")) else ","
        df = pd.read_csv(p, sep=sep, index_col=index_col)
    obs_cols = [c for c in ("age", "sex", "gender", "tissue", "condition", "mortstat",
                            "permth_exm", "permth_int") if c in df.columns]
    return FalconData(X=df, obs=df[obs_cols].copy() if obs_cols else pd.DataFrame(index=df.index),
                      modality="clinical_chemistry", units=_units(units, path))



def read_nightingale(path: str | Path, *, index_col: int | str = 0,
                     sheet: str | int | None = None) -> FalconData:
    """Read a Nightingale Health NMR export (CSV, TSV or Excel) for MetaboAge and
    MetaboHealth.

    Column names are translated to the BBMRI-NL names the scores were fitted
    with, by MiMIR's table (``metabo_names_translator``, GPL-3), the way MiMIR's
    ``find_BBMRI_names`` does it: lower-cased, and translated only when exactly
    one entry lists the name, so ``UnSat`` in a 2016 export becomes ``unsatdeg``.
    A name the table does not know is kept as it is and listed in
    ``obs.attrs["untranslated"]`` rather than guessed. Non-numeric columns and
    ``age``/``sex`` go to ``obs``.
    """
    from ..registry.registry import DATA_DIR

    p = Path(path)
    if p.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(p, sheet_name=sheet or 0, index_col=index_col)
    else:
        sep = "\t" if p.name.lower().endswith((".tsv", ".txt", ".tsv.gz")) else ","
        df = pd.read_csv(p, sep=sep, index_col=index_col)

    table = pd.read_csv(DATA_DIR / "metabolomics" / "nmr_names.csv")
    lookup: dict[str, list[str]] = {}
    for bbmri, alts in zip(table["bbmri"], table["alternatives"].fillna("")):
        for a in str(alts).split("|"):
            lookup.setdefault(a.strip().lower(), []).append(bbmri)
    known = set(table["bbmri"])

    obs_cols = [c for c in df.columns if str(c).lower() in ("age", "sex", "gender")
                or not pd.api.types.is_numeric_dtype(df[c])]
    rename, untranslated = {}, []
    for c in df.columns:
        if c in obs_cols:
            continue
        key = str(c).strip().lower()
        hits = lookup.get(key, [])
        if len(hits) == 1:
            rename[c] = hits[0]
        elif key in known:
            rename[c] = key
        else:
            untranslated.append(str(c))
    X = df.drop(columns=obs_cols).rename(columns=rename)
    obs = df[obs_cols].copy() if obs_cols else pd.DataFrame(index=df.index)
    obs.columns = [str(c).lower() if str(c).lower() in ("age", "sex", "gender") else c
                   for c in obs.columns]
    obs.attrs["untranslated"] = untranslated
    return FalconData(X=X.astype(float), obs=obs, modality="metabolomics_nmr")

def read(path: str | Path, **kw) -> FalconData:
    """Dispatch on the filename. Convenience, not magic -- it says what it chose."""
    p = Path(path)
    name = p.name.lower()
    if name.endswith(".h5ad"):
        return FalconData.read_h5ad(p)
    if "series_matrix" in name:
        return read_series_matrix(p, **kw)
    if name.endswith((".parquet", ".csv", ".csv.gz", ".tsv", ".tsv.gz", ".txt.gz")):
        return read_betas(p, **kw)
    raise DataError(
        f"cannot tell what {p.name} is.\n"
        "  Call the specific reader: read_betas, read_series_matrix, "
        "read_clinical, read_rrbs_dir, or FalconData.read_h5ad."
    )


def write_results(result, outdir: str | Path) -> dict[str, Path]:
    """Write the standard results layout. Returns what it wrote."""
    d = Path(outdir)
    d.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    written["scores"] = d / "scores.csv"
    result.long().to_csv(written["scores"], index=False)

    written["scores_wide"] = d / "scores_wide.csv"
    result.wide().to_csv(written["scores_wide"])

    written["qc"] = d / "qc.csv"
    result.qc().to_csv(written["qc"], index=False)

    written["manifest"] = result.manifest.write(d / "run_manifest.json")
    return written
