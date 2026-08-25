"""Trait, state and technical variance, decomposed on the same samples.

A published ICC answers "how reproducible is this clock across a cohort". An
individual asking "has my number actually moved" needs a different quantity, and
the gap between the two is why clock ICCs in the literature read as reassuring
while replicate disagreement of up to nine years is also true. Both follow from
the same three variances, and nothing separates them without repeated
measurement.

The design this needs, which is the part that is hard to find::

    subject A --- occasion 1 --- replicate 1     (same DNA, split and re-assayed)
               |              \\_ replicate 2
               \\_ occasion 2 --- replicate 1     (a different draw, weeks later)

Nested random effects, fitted by Henderson's method of moments:

.. math::

    y_{ijk} = \\mu + a_i + s_{ij} + \\varepsilon_{ijk}

for person :math:`i`, occasion :math:`j`, replicate :math:`k`.

Moment estimators are not clipped at zero here. A negative component means the
within-level spread exceeded the between-level spread, which is a real and
reportable state of affairs for a small design or for a clock that is measuring
nothing in this cohort, and rounding it to 0.0 disguises both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..core.errors import FalconError


class VarianceError(FalconError):
    """Raised when a design cannot support a variance decomposition."""


@dataclass
class VarianceComponents:
    """The three variances per clock, the two ICCs, and the design behind them.

    ``design`` is not decoration. These are moment estimators on whatever shape
    the data happened to have, and a reader who cannot see the subject and
    occasion counts cannot tell a decomposition from a coincidence.
    """

    table: pd.DataFrame
    design: dict[str, Any]

    def replicates_needed(self, target_icc: float = 0.9) -> pd.Series:
        """Replicates per occasion to reach a target effective reliability.

        Averaging :math:`r` replicates leaves
        :math:`\\sigma^2_{\\text{eff}} = \\sigma^2_{\\text{trait}} +
        \\sigma^2_{\\text{state}} + \\sigma^2_{\\text{tech}} / r`, so

        .. math::

            r = \\frac{\\sigma^2_{\\text{tech}}}
                     {\\sigma^2_{\\text{trait}}(1/\\text{target} - 1)
                      - \\sigma^2_{\\text{state}}}

        Returns ``inf`` where no number of replicates gets there, and that is the
        informative case rather than an error: replicates average away the
        *technical* term only. A clock whose day-to-day within-person variance
        already exceeds the budget cannot be rescued by running the array again,
        and the honest answer to "how many replicates" is then "a second draw,
        not a second scan".
        """
        t = self.table
        budget = t["var_trait"] * (1.0 / float(target_icc) - 1.0) - t["var_state"].fillna(0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = t["var_tech"] / budget
        r = r.where(budget > 0, np.inf)
        return np.ceil(r.clip(lower=1.0)).rename(f"replicates_for_icc_{target_icc:g}")

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (f"VarianceComponents({len(self.table)} clocks, "
                f"{self.design['n_subjects']} subjects, "
                f"{self.design['n_observations']} observations)")


def variance_components(result, *, subject_col: str,
                        occasion_col: str | None = None,
                        clocks: Sequence[str] | None = None,
                        age_col: str = "age") -> VarianceComponents:
    """Split each clock's variance into trait, state and technical parts.

    Two ICCs come back, and the gap between them is the whole point:

    ``icc``
        :math:`\\sigma^2_{\\text{trait}} / (\\sigma^2_{\\text{trait}} +
        \\sigma^2_{\\text{state}} + \\sigma^2_{\\text{tech}})` on raw scores.
        Comparable to what the literature publishes.
    ``icc_age_adjusted``
        The same after chronological age is regressed out. A cohort spanning
        decades holds most of its between-person variance *in the age term*, so
        a raw ICC largely reports that the clock tracks age, which was never in
        question. The adjusted figure is between-person reliability among people
        of the same age, which is what any individual-level claim rests on. It
        is routinely far lower, and a clock whose raw ICC is 0.98 and whose
        adjusted ICC is 0.4 is not a precise instrument, it is a cohort with a
        wide age range.

    Parameters
    ----------
    occasion_col
        Column identifying the sampling occasion within a subject. Omit when
        every repeat is a technical replicate of one draw: the state term is
        then not identifiable and comes back NaN rather than being folded
        silently into one of its neighbours, and ``icc`` becomes ICC(1,1).
    """
    obs = result.obs
    if subject_col not in obs.columns:
        raise VarianceError(f"no {subject_col!r} column in obs")
    if occasion_col is not None and occasion_col not in obs.columns:
        raise VarianceError(f"no {occasion_col!r} column in obs")

    subj = obs[subject_col].astype(str)
    if subj.nunique() < 3:
        raise VarianceError(
            f"only {subj.nunique()} subject(s); a between-person variance from "
            "fewer than three people is not an estimate")

    nested = occasion_col is not None
    if nested:
        occ = subj.str.cat(obs[occasion_col].astype(str), sep="||")
        if occ.nunique() <= subj.nunique():
            raise VarianceError(
                "every subject has exactly one occasion, so the state and "
                "technical terms are the same stratum.\n"
                "  Drop occasion_col to get the two-way split, or supply data "
                "with more than one draw per person.")
    else:
        occ = pd.Series(subj.to_numpy(), index=subj.index)

    cols = list(result.scores.columns) if clocks is None else [
        c for c in clocks if c in result.scores.columns]
    if not cols:
        raise VarianceError("none of the requested clocks is in this result")

    age = pd.to_numeric(obs[age_col], errors="coerce") if age_col in obs.columns else None

    rows: dict[str, dict[str, float]] = {}
    for cid in cols:
        y = pd.to_numeric(result.scores[cid], errors="coerce")
        ok = y.notna()
        if ok.sum() < 4:
            continue
        comp = _nested_moments(y[ok].to_numpy(dtype=np.float64),
                               subj[ok].to_numpy(), occ[ok].to_numpy(),
                               nested=nested)
        if comp is None:
            continue

        comp["icc_age_adjusted"] = np.nan
        if age is not None:
            a_ok = ok & age.notna()
            if a_ok.sum() >= 4 and age[a_ok].nunique() > 1:
                av = age[a_ok].to_numpy(dtype=np.float64)
                yv = y[a_ok].to_numpy(dtype=np.float64)
                resid = yv - np.polyval(np.polyfit(av, yv, 1), av)
                adj = _nested_moments(resid, subj[a_ok].to_numpy(),
                                      occ[a_ok].to_numpy(), nested=nested)
                if adj is not None:
                    comp["icc_age_adjusted"] = adj["icc"]

        comp["n_observations"] = float(ok.sum())
        rows[cid] = comp

    if not rows:
        raise VarianceError(
            "no clock had enough repeated observations to decompose")

    table = pd.DataFrame.from_dict(rows, orient="index")
    table["n_observations"] = table["n_observations"].astype(int)
    table = table[["var_trait", "var_state", "var_tech", "icc",
                   "icc_age_adjusted", "n_observations"]]
    design = {
        "n_subjects": int(subj.nunique()),
        "n_occasions": int(occ.nunique()),
        "n_observations": int(len(subj)),
        "nested": nested,
        "age_adjusted": age is not None,
    }
    return VarianceComponents(table=table.sort_values("icc", ascending=False),
                              design=design)


def _nested_moments(y: np.ndarray, subject: np.ndarray, occasion: np.ndarray,
                    *, nested: bool) -> dict[str, float] | None:
    """Henderson method-of-moments estimates for a two-level nested design.

    The unbalanced coefficients are the standard ones: with :math:`N`
    observations, :math:`a` subjects, :math:`n_i` per subject and :math:`n_{ij}`
    per occasion,

    .. math::

        E[MS_E] &= \\sigma^2_e \\\\
        E[MS_{B(A)}] &= \\sigma^2_e + k_1 \\sigma^2_s \\\\
        E[MS_A] &= \\sigma^2_e + k_2 \\sigma^2_s + k_3 \\sigma^2_a

    which collapse to the familiar :math:`K` and :math:`JK` when the design is
    balanced. Solved top-down. Returns ``None`` when a stratum has no degrees of
    freedom, which is what "everyone was measured once" looks like
    arithmetically.
    """
    N = y.size
    s_codes, s_uniq = pd.factorize(subject)
    o_codes, o_uniq = pd.factorize(occasion)
    a, b = len(s_uniq), len(o_uniq)

    df_a, df_b, df_e = a - 1, b - a, N - b
    if df_a < 1 or df_e < 1 or (nested and df_b < 1):
        return None

    grand = float(y.mean())
    n_i = np.bincount(s_codes, minlength=a).astype(np.float64)
    n_ij = np.bincount(o_codes, minlength=b).astype(np.float64)
    mean_i = np.bincount(s_codes, weights=y, minlength=a) / n_i
    mean_ij = np.bincount(o_codes, weights=y, minlength=b) / n_ij

    # Which subject each occasion belongs to. Occasion labels already carry the
    # subject prefix, so this mapping is well defined by construction.
    occ_subject = np.zeros(b, dtype=np.int64)
    occ_subject[o_codes] = s_codes

    ss_a = float((n_i * (mean_i - grand) ** 2).sum())
    ss_e = float(((y - mean_ij[o_codes]) ** 2).sum())
    ms_a, ms_e = ss_a / df_a, ss_e / df_e

    if not nested:
        var_tech = ms_e
        k = N / a
        var_trait = (ms_a - ms_e) / k
        total = var_trait + var_tech
        return {"var_trait": var_trait, "var_state": np.nan,
                "var_tech": var_tech,
                "icc": var_trait / total if total > 0 else np.nan}

    ss_b = float((n_ij * (mean_ij - mean_i[occ_subject]) ** 2).sum())
    ms_b = ss_b / df_b

    # Sum over subjects of sum over their occasions of n_ij^2 / n_i.
    per_occ = n_ij ** 2 / n_i[occ_subject]
    sum_nij2_over_ni = float(per_occ.sum())

    k1 = (N - sum_nij2_over_ni) / df_b
    k2 = (sum_nij2_over_ni - float((n_ij ** 2).sum()) / N) / df_a
    k3 = (N - float((n_i ** 2).sum()) / N) / df_a
    if k1 <= 0 or k3 <= 0:
        return None

    var_tech = ms_e
    var_state = (ms_b - ms_e) / k1
    var_trait = (ms_a - ms_e - k2 * var_state) / k3
    total = var_trait + var_state + var_tech
    return {"var_trait": var_trait, "var_state": var_state, "var_tech": var_tech,
            "icc": var_trait / total if total > 0 else np.nan}
