"""Stage 2b: biolearn's scores for the mapped clocks (run with /opt/biolearn/bin/python)."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"


def main() -> int:
    from biolearn.data_library import GeoData
    from biolearn.model_gallery import ModelGallery

    pairs = yaml.safe_load((HERE / "pairs.yaml").read_text())["clocks"]
    betas = pd.read_csv(OUT / "betas.csv", index_col=0)
    pheno = pd.read_csv(OUT / "pheno.csv", index_col=0)
    data = GeoData(pheno, betas)
    gallery = ModelGallery()
    rows, errors = [], []
    for cid, names in pairs.items():
        name = names.get("biolearn")
        if not name:
            continue
        try:
            pred = gallery.get(name).predict(data)
            col = "Predicted" if "Predicted" in pred.columns else pred.columns[0]
            for s, v in pred[col].items():
                rows.append({"package": "biolearn", "clock": cid, "name": name, "sample": s, "value": v})
        except Exception as exc:  # noqa: BLE001 - recorded, not hidden
            errors.append({"package": "biolearn", "clock": cid, "name": name,
                           "error": f"{type(exc).__name__}: {exc}"[:300]})
            traceback.print_exc(limit=1)
    pd.DataFrame(rows).to_csv(OUT / "biolearn.csv", index=False, float_format="%.12g")
    pd.DataFrame(errors, columns=["package", "clock", "name", "error"]).to_csv(
        OUT / "biolearn_errors.csv", index=False)
    import biolearn
    print(f"biolearn {getattr(biolearn, '__version__', '?')}: {len(rows)} scores, {len(errors)} errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
