"""Helpers shared by the analysis modules."""

from __future__ import annotations


import numpy as np

from ..core.errors import IllegalOperationError




def _check_legal(registry, clock_id: str, op: str) -> None:
    c = registry.get(clock_id)
    if op not in c.legal_operations:
        raise IllegalOperationError(
            f"{op!r} is not defined for {clock_id}, whose output is "
            f"{c.scale_type} ({', '.join(c.unit) or 'no unit'}).\n"
            f"  Legal here: {', '.join(sorted(c.legal_operations))}.\n"
            + ("  A pace of aging is already a rate; subtracting chronological "
               "age from it is a units error, not a conservative choice.\n"
               if c.scale_type == "pace_ratio" else "")
            + ("  This clock is in years and tracks age with a slope near one, "
               "but its origin is not fixed: its offset against chronological "
               "age moves by over a hundred years between cohorts. Subtracting "
               "chronological age from it measures the cohort. Use "
               "method='residual' or method='within_group', which fit inside "
               "the data you give them and are what the published analyses "
               "use.\n" if c.scale_type == "age_years_relative" else "")
            + ("  A log-hazard has no zero point on the age scale; use "
               "cox_hazard or rank it.\n" if c.scale_type == "mortality_log_hazard" else "")
        )


def _bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg, NaN-safe."""
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    v = p[ok]
    n = v.size
    order = np.argsort(v)
    ranked = v[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    q[ok] = out
    return q
