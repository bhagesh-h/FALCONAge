"""Unit registry for clinical chemistry, and the refusal to guess.

The clinical clocks are where the field loses most of its reproducibility, and
it is almost always units. Levine's PhenoAge was published with albumin in g/L
and creatinine in umol/L; half the reimplementations feed it g/dL and mg/dL,
which changes the coefficient-weighted sum by two orders of magnitude on those
terms and still returns a plausible-looking age. Both inputs are inside the
clinically normal range for *their own* unit, so no range check can separate
them. The only correct behaviour is to require the caller to say.

``convert`` handles the conversions that are exact ratios. Anything else --
notably anything needing a molar mass -- is listed explicitly rather than
computed, so a wrong factor is a visible line in a table instead of a hidden
constant.
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import UnitConversionError, UnitsNotDeclaredError


@dataclass(frozen=True)
class Marker:
    """One clinical measurement: its canonical unit and its plausible range."""

    name: str
    canonical: str
    aliases: tuple[str, ...]
    #: Physiologically plausible window in the canonical unit. Used for a
    #: warning, never for inferring which unit was meant -- see the module
    #: docstring for why that inference is impossible.
    plausible: tuple[float, float]
    description: str


#: Conversions expressed as multiply-by factors into the canonical unit.
#: Molar conversions carry the molar mass in the comment so the factor can be
#: rederived rather than trusted.
_FACTORS: dict[tuple[str, str], float] = {
    # albumin: canonical g/L
    ("g/dL", "g/L"): 10.0,
    ("g/L", "g/dL"): 0.1,
    # creatinine: canonical umol/L; molar mass 113.12 g/mol
    ("mg/dL", "umol/L"): 88.4017,
    ("umol/L", "mg/dL"): 1.0 / 88.4017,
    # glucose: canonical mmol/L; molar mass 180.156 g/mol
    ("mg/dL", "mmol/L"): 0.0555,
    ("mmol/L", "mg/dL"): 1.0 / 0.0555,
    # C-reactive protein: canonical mg/L
    ("mg/dL", "mg/L"): 10.0,
    ("mg/L", "mg/dL"): 0.1,
    # cell counts
    ("10^3/uL", "10^9/L"): 1.0,
    ("10^9/L", "10^3/uL"): 1.0,
    ("cells/uL", "10^9/L"): 1e-3,
    ("10^9/L", "cells/uL"): 1e3,
    # percentages and volumes are their own canonical form
    ("%", "%"): 1.0,
    ("fL", "fL"): 1.0,
    ("years", "years"): 1.0,
    ("days", "weeks"): 1.0 / 7.0,
    ("weeks", "days"): 7.0,
    ("months", "years"): 1.0 / 12.0,
    ("years", "months"): 12.0,
}


#: The ten PhenoAge markers plus the KDM/HD panel, with the units the papers
#: used. ``canonical`` is what the model code sees; anything else is converted
#: on the way in and recorded in the manifest.
MARKERS: dict[str, Marker] = {
    "albumin": Marker("albumin", "g/L", ("alb",), (20.0, 60.0),
                      "serum albumin"),
    "creatinine": Marker("creatinine", "umol/L", ("creat",), (20.0, 900.0),
                         "serum creatinine"),
    "glucose": Marker("glucose", "mmol/L", ("glu", "fasting_glucose"), (2.0, 35.0),
                      "serum glucose"),
    "crp": Marker("crp", "mg/L", ("c_reactive_protein", "creactive_protein"), (0.01, 300.0),
                  "C-reactive protein; log-transformed by PhenoAge"),
    "lymphocyte_percent": Marker("lymphocyte_percent", "%", ("lymph", "lymphocyte_pct"),
                                 (1.0, 80.0), "lymphocytes as a percentage of leukocytes"),
    "mean_cell_volume": Marker("mean_cell_volume", "fL", ("mcv",), (50.0, 130.0),
                               "mean corpuscular volume"),
    "red_cell_distribution_width": Marker("red_cell_distribution_width", "%", ("rdw",),
                                          (8.0, 30.0), "red cell distribution width"),
    "alkaline_phosphatase": Marker("alkaline_phosphatase", "U/L", ("alp",), (5.0, 1500.0),
                                   "serum alkaline phosphatase"),
    "white_blood_cell_count": Marker("white_blood_cell_count", "10^9/L", ("wbc",),
                                     (0.5, 60.0), "leukocyte count"),
    "age": Marker("age", "years", ("chronological_age", "chrono_age"), (0.0, 130.0),
                  "chronological age"),
    # KDM / homeostatic dysregulation panel additions
    "systolic_blood_pressure": Marker("systolic_blood_pressure", "mmHg", ("sbp",),
                                      (60.0, 260.0), "systolic blood pressure"),
    "total_cholesterol": Marker("total_cholesterol", "mmol/L", ("totchol", "chol"),
                                (1.0, 15.0), "total cholesterol"),
    "hba1c": Marker("hba1c", "%", ("glycated_haemoglobin",), (2.0, 20.0),
                    "glycated haemoglobin"),
    "urea_nitrogen": Marker("urea_nitrogen", "mmol/L", ("bun",), (0.5, 60.0),
                            "blood urea nitrogen"),
    "forced_expiratory_volume": Marker("forced_expiratory_volume", "mL", ("fev", "fev1"),
                                       (200.0, 7000.0), "FEV1"),
    "cytomegalovirus_optical_density": Marker("cytomegalovirus_optical_density", "od",
                                              ("cmv",), (0.0, 10.0),
                                              "CMV antibody optical density"),
}

_ALIAS: dict[str, str] = {}
for _m in MARKERS.values():
    _ALIAS[_m.name] = _m.name
    for _a in _m.aliases:
        _ALIAS[_a] = _m.name


def canonical_name(name: str) -> str:
    """Map a column name or alias to its canonical marker name."""
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    if key not in _ALIAS:
        raise UnitConversionError(
            f"unknown clinical marker {name!r}\n"
            f"  known: {', '.join(sorted(MARKERS))}"
        )
    return _ALIAS[key]


def convert(value, frm: str, to: str):
    """Convert ``value`` from one unit to another, or say why it cannot.

    Works elementwise on scalars and numpy arrays alike.
    """
    if frm == to:
        return value
    try:
        return value * _FACTORS[(frm, to)]
    except KeyError:
        raise UnitConversionError(
            f"no conversion from {frm!r} to {to!r} is defined.\n"
            "  Conversions are enumerated rather than derived, so that a wrong "
            "factor is a visible line in falconage/core/units.py rather than a "
            "hidden constant. Add the pair there, with the molar mass in a "
            "comment if one is involved."
        ) from None


def require_units(declared: dict[str, str] | None, columns: list[str]) -> dict[str, str]:
    """Validate that units were declared for every clinical column.

    Raises
    ------
    UnitsNotDeclaredError
        With the exact dict the caller needs to supply. Making the error
        actionable matters more here than anywhere else in the package: this is
        the failure a user hits first and the one they are most tempted to work
        around by guessing.
    """
    markers = [c for c in columns if c.strip().lower() in _ALIAS]
    if declared is None:
        example = "{\n" + "\n".join(
            f"    {canonical_name(c)!r}: {MARKERS[canonical_name(c)].canonical!r},"
            for c in markers[:4]
        ) + "\n    ...\n}"
        raise UnitsNotDeclaredError(
            "clinical data needs an explicit units= mapping.\n\n"
            f"  {len(markers)} recognised marker column(s): {', '.join(markers[:8])}"
            f"{' ...' if len(markers) > 8 else ''}\n\n"
            "  FALCONAge will not infer units. Albumin at 4.2 is g/dL and albumin "
            "at 42 is g/L; both are clinically normal, and PhenoAge fitted on one "
            "returns nonsense for the other. Pass, for example:\n\n"
            f"    units={example}"
        )

    missing = [c for c in markers if canonical_name(c) not in
               {canonical_name(k) for k in declared}]
    if missing:
        raise UnitsNotDeclaredError(
            "units= is missing entries for: " + ", ".join(missing) + "\n"
            "  Every recognised marker column needs one; drop the column if you "
            "do not have the unit."
        )
    return {canonical_name(k): v for k, v in declared.items()}


def check_plausible(name: str, values) -> str | None:
    """Return a warning string if values sit outside the plausible window.

    A warning, never an error, and never a unit inference. Real cohorts contain
    real outliers, and a haemodialysis creatinine of 1200 umol/L is a patient,
    not a typo.
    """
    import numpy as np

    m = MARKERS[canonical_name(name)]
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    lo, hi = m.plausible
    out = int(((v < lo) | (v > hi)).sum())
    if out == 0:
        return None
    return (f"{m.name}: {out}/{v.size} value(s) outside the plausible "
            f"[{lo}, {hi}] {m.canonical} window (median {float(np.median(v)):.3g}). "
            f"Check the declared unit.")
