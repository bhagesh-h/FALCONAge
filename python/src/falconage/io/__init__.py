"""Readers and writers. Every reader returns a FalconData."""

from pathlib import Path

import pandas as pd

from ..core.container import FalconData
from ..core.errors import DataError
from .methylation import (
    detect_platform,
    read_betas,
    read_idat_pair,
    read_rrbs,
    read_rrbs_dir,
    read_series_matrix,
)

__all__ = [
    "detect_platform", "read", "read_betas", "read_clinical", "read_idat_pair",
    "read_rrbs", "read_rrbs_dir", "read_series_matrix", "write_results",
]


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
                      modality="clinical_chemistry", units=dict(units or {}))


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
