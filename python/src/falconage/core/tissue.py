"""Specimen types, and whether a clock's coefficients mean anything on one.

A clock's intercept is fitted on a particular tissue. Applied to a different
one it still returns a number, still correlates with age, and is wrong by an
amount nothing in the arithmetic can see. The measured sizes:

* saliva against buffy coat, same 91 people: clock ages differ by **3.83 to
  16.46 years** (bioRxiv 2025.09.16.673560) while still correlating with each
  other at Spearman 0.45-0.69. Correlation is not agreement, and the check most
  people run is the correlation.
* buffy coat against PBMC, same people: 0.66-0.87 on the clock ages, and a mean
  difference of only -0.06 to 0.39 years. One family, and the contrast with
  saliva is what makes the saliva number a specimen effect rather than noise.
* which clock matters as much as which specimen: on those samples DunedinPACE
  differed by -0.007 years (p = 0.486) where Hannum differed by 12.5.
* unrelated tissues generally: **20-30 years** (PMC12714307).

So this module is a lookup table and two questions: what did the user mean by
the string in ``obs["tissue"]``, and is it the same thing the clock was fitted
on. Both answers are deliberately coarse. A wrong-by-a-decade result is not
improved by a fine-grained ontology; it is prevented by noticing that "saliva"
and "whole blood" are different words.
"""

from __future__ import annotations

import re

#: Families. Within a family the substitution is defensible and gets a note;
#: across families it gets a warning or a refusal, depending on the clock.
BLOOD = "blood"
CORD = "cord_blood"
SALIVA = "saliva"
BUCCAL = "buccal"
PLACENTA = "placenta"
SKIN = "skin"
BRAIN = "brain"
#: Its own family, not a kind of blood. Cell-free DNA is a fragment population
#: shed by dying cells across many tissues, at a coverage profile nothing on an
#: array resembles; array clocks applied to it directly perform poorly
#: (bioRxiv 2025.11.27.690895). Grouping it under blood would make the check
#: pass on exactly the case it should stop.
CFDNA = "cfdna"
OTHER = "other"
ANY = "any"

#: Canonical specimen -> family. The keys are what ``normalise`` returns.
FAMILY: dict[str, str] = {
    "whole blood": BLOOD,
    "buffy coat": BLOOD,
    "pbmc": BLOOD,
    "leukocytes": BLOOD,
    "dried blood spot": BLOOD,
    "blood": BLOOD,
    "cord blood": CORD,
    "neonatal blood spots": CORD,
    "saliva": SALIVA,
    "buccal epithelium": BUCCAL,
    "placenta": PLACENTA,
    "skin": SKIN,
    "fibroblast": SKIN,
    "epithelium": SKIN,
    "brain": BRAIN,
    "cell-free dna": CFDNA,
    "multi-tissue": ANY,
    "cultured cells": OTHER,
    "adipose": OTHER,
    "muscle": OTHER,
    "liver": OTHER,
    "kidney": OTHER,
    "lung": OTHER,
    "breast": OTHER,
    "colon": OTHER,
    "heart": OTHER,
    "spleen": OTHER,
    "prostate": OTHER,
    "sperm": OTHER,
    "oocyte": OTHER,
}

#: Everything a user might actually type, mapped onto a key of ``FAMILY``.
#: Written as an alias table rather than fuzzy matching on purpose: a specimen
#: this table does not know becomes ``None``, which produces "unrecognised" --
#: a true statement -- instead of a confident wrong match.
_ALIASES: dict[str, str] = {
    "wholeblood": "whole blood", "whole_blood": "whole blood", "wb": "whole blood",
    "peripheral blood": "whole blood", "peripheral whole blood": "whole blood",
    "venous blood": "whole blood", "blood": "blood",
    "buffycoat": "buffy coat", "buffy_coat": "buffy coat", "buffy": "buffy coat",
    "pbmc": "pbmc", "pbmcs": "pbmc",
    "peripheral blood mononuclear cells": "pbmc",
    "mononuclear cells": "pbmc",
    "leukocyte": "leukocytes", "leucocytes": "leukocytes", "wbc": "leukocytes",
    "granulocytes": "leukocytes", "cd4t": "leukocytes", "cd8t": "leukocytes",
    "monocytes": "leukocytes", "neutrophils": "leukocytes", "lymphocytes": "leukocytes",
    "purified blood leukocytes": "leukocytes", "sorted monocytes": "leukocytes",
    "b cells": "leukocytes", "t cells": "leukocytes", "nk cells": "leukocytes",
    "dried blood spot": "dried blood spot", "dbs": "dried blood spot",
    "bloodspot": "dried blood spot", "blood spot": "dried blood spot",
    "cord blood": "cord blood", "cordblood": "cord blood",
    "umbilical cord blood": "cord blood",
    "neonatal blood spots": "neonatal blood spots",
    "neonatal blood spot": "neonatal blood spots",
    "saliva": "saliva", "salivary": "saliva", "oral fluid": "saliva",
    "buccal": "buccal epithelium", "buccal epithelium": "buccal epithelium",
    "buccal cells": "buccal epithelium", "buccal swab": "buccal epithelium",
    "cheek swab": "buccal epithelium",
    "placenta": "placenta", "placental": "placenta", "chorionic villi": "placenta",
    "skin": "skin", "dermis": "skin", "epidermis": "skin",
    "fibroblast": "fibroblast", "fibroblasts": "fibroblast",
    "cultured fibroblasts": "fibroblast", "keratinocytes": "skin",
    "epithelium": "epithelium", "epithelial": "epithelium",
    "brain": "brain", "cortex": "brain", "prefrontal cortex": "brain",
    "brain cortex": "brain", "cerebellum": "brain", "hippocampus": "brain",
    "plasma cell free dna": "cell-free dna", "cell free dna": "cell-free dna",
    "cfdna": "cell-free dna", "plasma cfdna": "cell-free dna",
    "circulating cell free dna": "cell-free dna",
    "multi-tissue": "multi-tissue", "multi tissue": "multi-tissue",
    "multitissue": "multi-tissue", "pan-tissue": "multi-tissue", "any": "multi-tissue",
    "cultured cells": "cultured cells", "cell line": "cultured cells",
    "ipsc": "cultured cells", "esc": "cultured cells",
    "cultured primary human cells": "cultured cells",
    "cultured human cells": "cultured cells",
    "cultured mesenchymal stromal cells": "cultured cells",
    "adipose": "adipose", "adipose tissue": "adipose", "fat": "adipose",
    "muscle": "muscle", "skeletal muscle": "muscle",
    "liver": "liver", "kidney": "kidney", "lung": "lung",
    "breast": "breast", "colon": "colon", "heart": "heart", "spleen": "spleen",
    "prostate": "prostate", "sperm": "sperm", "oocyte": "oocyte",
}

#: The concordance figures, so a warning can carry the number rather than an
#: adjective. Keyed by the unordered pair of families.
_MEASURED: dict[frozenset, str] = {
    frozenset({BLOOD, SALIVA}): (
        "saliva clock ages ran 3.83-16.46 years above buffy coat in the same 91 "
        "people, while still correlating with it at Spearman 0.45-0.69 "
        "(bioRxiv 2025.09.16.673560)"),
    frozenset({BLOOD, BUCCAL}): (
        "buccal and blood are different cell populations; clocks fitted on one "
        "carry an unquantified offset on the other"),
    frozenset({BLOOD, CORD}): (
        "cord blood has a distinct cell composition and a near-zero age range; "
        "an adult blood clock has no calibration there"),
    frozenset({BLOOD, CFDNA}): (
        "cell-free DNA is a fragment population shed from many tissues, not a "
        "cell type; array clocks applied to it directly perform poorly "
        "(bioRxiv 2025.11.27.690895)"),
}

_DEFAULT_MISMATCH = ("cross-tissue application of an aging clock has been "
                     "measured at 20-30 years of error (PMC12714307)")

#: Families that are a refusal whatever the clock's own policy says, because the
#: substitution is a category error rather than an offset. A clock's
#: ``tissue_policy`` is a statement about how far its own tissue generalises; it
#: is not a licence to score a specimen that is not a tissue at all.
ALWAYS_REFUSE: frozenset = frozenset({CFDNA})


def normalise(value: object) -> str | None:
    """Map whatever the sample sheet says onto a canonical specimen, or ``None``.

    ``None`` means "not recognised", never "fine". A specimen this table has
    never heard of is reported as unrecognised so the user can either fix the
    label or accept that the check did not run -- both better than a guess.
    """
    if value is None:
        return None
    s = str(value).strip().lower()
    if not s or s in ("na", "nan", "none", "unknown", "not specified", "-"):
        return None
    s = re.sub(r"[\s_/-]+", " ", s).strip()
    if s in _ALIASES:
        return _ALIASES[s]
    if s in FAMILY:
        return s
    # One narrow fallback: a trailing qualifier such as "whole blood (fasting)".
    head = re.split(r"[(,;]", s)[0].strip()
    if head != s:
        return _ALIASES.get(head) or (head if head in FAMILY else None)
    return None


def family(specimen: str | None) -> str | None:
    if specimen is None:
        return None
    return FAMILY.get(specimen)


def compare(sample: object, clock_tissues: tuple[str, ...] | list[str]) -> dict:
    """How a sample's specimen relates to the tissues a clock was fitted on.

    Returns a dict with ``verdict`` in ``exact`` / ``family`` / ``mismatch`` /
    ``unrecognised`` / ``unrestricted``, the normalised specimen, and a
    ``message`` carrying the measured discordance where one is published.
    """
    got = normalise(sample)
    want = [normalise(t) or str(t).strip().lower() for t in (clock_tissues or ())]
    want_fams = {family(t) for t in want} - {None}

    if not want or ANY in want_fams:
        return {"verdict": "unrestricted", "specimen": got, "message": ""}
    if got is None:
        return {"verdict": "unrecognised", "specimen": None,
                "message": (f"specimen {sample!r} is not a recognised type, so the "
                            "tissue check did not run")}

    if got in want:
        return {"verdict": "exact", "specimen": got, "message": ""}

    fam = family(got)
    if fam is not None and fam in want_fams:
        return {"verdict": "family", "specimen": got,
                "message": (f"{got} is not one of the tissues this clock was fitted "
                            f"on ({', '.join(want)}), but is in the same family. "
                            "Buffy coat and PBMC clock ages agree at Spearman "
                            "0.66-0.87, with a mean difference of -0.06 to 0.39 "
                            "years (bioRxiv 2025.09.16.673560)")}

    detail = ""
    for w in want_fams:
        key = frozenset({fam, w})
        if key in _MEASURED:
            detail = _MEASURED[key]
            break
    return {"verdict": "mismatch", "specimen": got,
            "message": (f"fitted on {', '.join(want)}, scored on {got}. "
                        + (detail or _DEFAULT_MISMATCH))}
