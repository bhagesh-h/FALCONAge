"""Everything that happened to the sample before the array read it.

No algorithm here. A recognised set of ``obs`` columns, the thresholds at which
each becomes worth mentioning, and a block for the run manifest -- because the
published pre-analytical effects are mostly small, with one exception large
enough that a run which cannot rule it out is not interpretable.

What the literature actually says, so the thresholds are not arbitrary:

* **Anticoagulant** (EDTA / heparin / ACD): no reported effect on methylation.
  Recorded, never flagged.
* **Short-term room-temperature storage**: negligible (Epigenomes 2017;
  PMC5813389). Archived DNA at 4 C for twenty years still looks like recently
  collected DNA.
* **Long storage of whole blood**: the exception. After ten months, DNA yield
  fell by up to 97.45% and methylation rose by up to 42.0% (PLOS One 2018,
  PMC5802893). Six months is the conservative point to start saying so.
* **Delay between collection and processing**: 24 h at 4 C already shifts buffy
  coat composition -- lymphocytes down, granulocytes up (PMC4723336). That
  becomes a cell-composition artefact, which becomes an age-acceleration
  artefact, and deconvolution cannot tell it from biology because it *is* a real
  composition difference.

The point of recording the fields is not to correct for them. It is that a run
whose manifest says "storage time unknown" is honestly less interpretable than
one that says "14 days", and the difference should be visible without asking
the person who ran it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class Field:
    name: str
    what: str
    why: str
    #: Value above which the field is worth a warning. ``None`` means the field
    #: is recorded for provenance and never flagged.
    flag_above: float | None = None


#: Recognised columns. Nothing is required -- an absent field produces a note
#: saying the run is not reconstructable in that respect, which is a true
#: statement and the correct incentive.
FIELDS: tuple[Field, ...] = (
    Field("specimen_type", "whole blood, buffy coat, PBMC, saliva, buccal, ...",
          "drives the clock specimen check; saliva against buffy coat differs "
          "by 3.83-16.46 years in the same people"),
    Field("collection_to_processing_h", "hours between draw and freeze or extraction",
          "24 h at 4 C shifts buffy coat composition, which reads downstream as "
          "age acceleration", flag_above=24.0),
    Field("storage_months", "months the specimen was stored before extraction",
          "ten months of whole-blood storage moved methylation by up to 42% and "
          "cost up to 97% of the DNA yield", flag_above=6.0),
    Field("storage_temp_c", "storage temperature in Celsius",
          "interpreted together with storage_months; -80 C is not 4 C"),
    Field("anticoagulant", "EDTA, heparin, ACD",
          "no measurable effect on methylation; recorded so that stays checkable"),
    Field("extraction_method", "kit or protocol",
          "part of reconstructing the run, and a plausible batch axis"),
    Field("bisulfite_kit", "conversion kit and lot",
          "conversion efficiency is the step with no post-hoc check on an array"),
    Field("dna_input_ng", "nanograms into bisulfite conversion",
          "low input is the most common cause of a failed or noisy array"),
    Field("array_version", "450K, EPICv1, EPICv2, MSA",
          "clock estimates differ between array versions on the same samples"),
    Field("scan_date", "date the chip was read", "a batch axis, and a drift axis"),
    Field("plate", "plate or chip barcode", "the batch axis for correction"),
    Field("sentrix_position", "position on the chip",
          "position effects are real and are not removed by plate-level correction"),
)

BY_NAME = {f.name: f for f in FIELDS}


def audit(obs: pd.DataFrame) -> dict[str, Any]:
    """Which pre-analytical fields are present, and what they say.

    Returns a manifest-shaped dict: ``present`` maps a recognised field to a
    short description of its values, ``absent`` lists the rest, and ``flags``
    carries the ones past a published threshold.
    """
    present: dict[str, str] = {}
    flags: list[str] = []
    for f in FIELDS:
        if f.name not in obs.columns:
            continue
        col = obs[f.name].dropna()
        if col.empty:
            continue
        if f.flag_above is not None:
            num = pd.to_numeric(col, errors="coerce").dropna()
            if num.empty:
                present[f.name] = f"{len(col)} non-numeric value(s)"
                continue
            present[f.name] = (f"median {num.median():g}, range "
                               f"{num.min():g}-{num.max():g}")
            over = int((num > f.flag_above).sum())
            if over:
                flags.append(
                    f"{over} sample(s) with {f.name} above {f.flag_above:g}: {f.why}")
        else:
            vals = sorted({str(v) for v in col})
            present[f.name] = (", ".join(vals[:4])
                               + (f" (+{len(vals) - 4} more)" if len(vals) > 4 else ""))
    absent = [f.name for f in FIELDS if f.name not in present]
    return {"present": present, "absent": absent, "flags": flags,
            "n_recognised": len(present), "n_fields": len(FIELDS)}


def notes(obs: pd.DataFrame) -> list[str]:
    """Warning lines for :func:`falconage.qc`."""
    a = audit(obs)
    out = list(a["flags"])
    if not a["present"]:
        out.append(
            "no pre-analytical metadata recorded. Specimen type, storage time and "
            "collection-to-processing delay all move methylation, and none of "
            f"them can be recovered later. Recognised columns: "
            f"{', '.join(f.name for f in FIELDS)}")
    return out
