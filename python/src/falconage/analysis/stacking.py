"""Combine several clocks' scores into one, honestly.

Combining clocks has improved robustness and prediction in several designs: a
meta-clock built from many clocks' shared signal (Liu et al. 2020,
doi:10.1111/acel.13229), ensembles of penalised models (Haghani et al. 2025,
doi:10.1007/s11357-025-01808-1; Anand et al. 2025,
doi:10.1186/s12864-025-12271-9), and multi-omic stacks (Jiang et al. 2026,
doi:10.1093/bib/bbag271). The general method is the super learner: base
predictions made out of fold, weights chosen by cross-validation (van der Laan
et al. 2007, doi:10.2202/1544-6115.1309).

Four rules specific to this package:

* **Units.** A plain mean is allowed only across clocks of one scale type. A
  learned stack standardises its inputs and takes the scale of its target.
* **Out of fold.** Every reported number comes from weights fitted without the
  sample, and the comparison is against the best single clock and the plain
  mean measured the same way.
* **Circularity.** A stack fitted on the samples later compared between groups
  has seen the answer. Fit it on another set, or use :func:`consensus` to ask
  whether a difference holds across clocks instead.
* **Correlated inputs.** First-generation clocks are strongly correlated, so
  the weights come from non-negative least squares or ridge, and are reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from ..core.errors import AnalysisError, IllegalOperationError
from .learners import folds, metrics

__all__ = ["StackFit", "stack_clocks"]


def _cindex(risk: np.ndarray, t: np.ndarray, e: np.ndarray) -> float:
    """Harrell's concordance: of comparable pairs, how many the risk orders right."""
    num = den = 0.0
    for i in np.where(e == 1)[0]:
        later = t > t[i]
        den += later.sum()
        num += (risk[i] > risk[later]).sum() + 0.5 * (risk[i] == risk[later]).sum()
    return float(num / den) if den else float("nan")


@dataclass
class StackFit:
    """A fitted combination of clocks."""

    clocks: list[str]
    method: str
    target: str
    intercept: float
    #: One weight per clock, on the standardised score (mean 0, SD 1 in the
    #: training data): comparable across clocks whatever their units.
    weights: pd.Series
    means: pd.Series
    sds: pd.Series
    predictions: pd.Series
    metrics: dict[str, float]
    #: Out-of-fold metrics of each single clock and of the plain mean.
    comparison: pd.DataFrame
    scale_type: str = "age_years"
    notes: list[str] = field(default_factory=list)

    def predict(self, result) -> pd.Series:
        """Apply the fitted stack to a scored result holding the same clocks."""
        missing = [c for c in self.clocks if c not in result.scores.columns]
        if missing:
            raise AnalysisError(f"the result lacks {', '.join(missing)}; a stack scores "
                                "only when every one of its clocks was scored")
        Z = (result.scores[self.clocks] - self.means) / self.sds
        return pd.Series(self.intercept + Z.to_numpy() @ self.weights.to_numpy(),
                         index=result.scores.index, name="stack")

    def to_clock(self, clock_id: str, outdir, *, registry=None, name: str | None = None,
                 tissue=("whole blood",)) -> Path:
        """Collapse the stack to one linear clock and write it for ``add_clock``.

        Possible when every base clock is a weighted sum of CpGs followed only
        by constant shifts and scalings: the stack is then itself a weighted sum
        over the union of their CpGs, so it gets coverage checks and technical
        standard errors like any other clock.
        """
        import yaml

        from ..models import ops
        from ..registry import load

        if self.method == "cox":
            raise AnalysisError("a Cox stack is a log hazard, not a clock in years; "
                                "keep it with StackFit.predict()")
        reg = registry or load()
        beta: dict[str, float] = {}
        const = self.intercept
        refs: dict[str, float] = {}
        for c in self.clocks:
            clk = reg.get(c)
            if clk.preprocess or not reg.has_coefficient_vector(c):
                raise AnalysisError(f"{c} is not a plain weighted sum of CpGs, so the "
                                    "stack cannot be written as one coefficient file")
            # Its postprocess must be affine: f(z) = a + b z.
            f = (lambda z, chain=clk.postprocess: float(
                ops.apply_chain(np.array([z], float), chain, ops.POSTPROCESS, xp=np)[0])
                if chain else z)
            a, b = f(0.0), f(1.0) - f(0.0)
            if abs(f(2.0) - (a + 2 * b)) > 1e-9:
                raise AnalysisError(f"{c} ends in a nonlinear transform, so the stack "
                                    "cannot be written as one coefficient file")
            feats, coefs = reg.coefficients(c)
            k = self.weights[c] / self.sds[c]
            const += k * (a + b * reg.intercept(c) - self.means[c])
            for fid, w in zip(feats, coefs):
                beta[fid] = beta.get(fid, 0.0) + k * b * w
            for fid, v in (reg.reference_values(c) or {}).items():
                refs.setdefault(fid, v)
        d = Path(outdir)
        d.mkdir(parents=True, exist_ok=True)
        coef = pd.Series({"(Intercept)": const, **beta}, name="coefficient")
        coef.to_csv(d / f"{clock_id}.csv", index_label="feature_id")
        entry = {
            "id": clock_id, "name": name or f"Stack of {', '.join(self.clocks)}",
            "data_type": reg.get(self.clocks[0]).data_type, "tissue": list(tissue),
            "platform": sorted({p for c in self.clocks for p in reg.get(c).platform}),
            "training_target": [self.target], "scale_type": self.scale_type,
            "unit": ["years" if self.scale_type == "age_years" else "unitless"],
            "model_type": f"linear (stack of {len(self.clocks)} clocks, {self.method})",
            "coefficients": f"{clock_id}.csv",
            "notes": " ".join(self.notes),
        }
        if refs:
            pd.Series(refs, name="value").reindex(list(beta)).dropna().to_csv(
                d / f"{clock_id}_reference.csv", index_label="feature_id")
            entry["reference"] = f"{clock_id}_reference.csv"
        out = d / f"{clock_id}.yaml"
        out.write_text(yaml.safe_dump(entry, sort_keys=False), encoding="utf-8")
        return out

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"StackFit({self.method}, {len(self.clocks)} clocks, {self.metrics})"


def _fit(Z: np.ndarray, y: np.ndarray, method: str, ridge: float):
    if method == "nnls":
        yc = y - y.mean()
        w, _ = nnls(Z, yc)
        return float(y.mean()), w
    if method == "ridge":
        yc = y - y.mean()
        w = np.linalg.solve(Z.T @ Z + ridge * len(y) * np.eye(Z.shape[1]), Z.T @ yc)
        return float(y.mean()), w
    raise AnalysisError(f"unknown method {method!r}")


def stack_clocks(result, target: str = "age", *, clocks=None, method: str = "nnls",
          event_col: str | None = None, outer_folds: int = 5, groups: str | None = None,
          ridge: float = 0.01, seed: int = 20261122) -> StackFit:
    """Fit a combination of clocks, judged out of fold.

    Parameters
    ----------
    result
        A scored :class:`~falconage.score.FalconResult`.
    target
        The ``obs`` column to predict: an age, a phenotype, or with
        ``method="cox"`` the follow-up time (``event_col`` then names the event
        indicator).
    clocks
        Which scored clocks to combine; by default every one scored on every
        sample with the target.
    method
        ``"mean"`` (only across clocks of one scale type), ``"nnls"``
        (non-negative weights, the default), ``"ridge"``, or ``"cox"``.
    groups
        An ``obs`` column whose members always share a fold.

    Returns
    -------
    StackFit
        The weights, the out-of-fold predictions and metrics, and the same
        metrics for each single clock and the plain mean, so the gain (or its
        absence) is visible.
    """
    from ..registry import load

    reg = getattr(result, "registry", None) or load()
    S = result.scores if clocks is None else result.scores[list(clocks)]
    if target not in result.obs.columns:
        raise AnalysisError(f"no {target!r} column in obs")
    y = pd.to_numeric(result.obs[target], errors="coerce")
    keep = y.notna() & S.notna().all(axis=1)
    if method == "cox":
        if not event_col or event_col not in result.obs.columns:
            raise AnalysisError("method='cox' needs event_col naming the event indicator")
        ev = pd.to_numeric(result.obs[event_col], errors="coerce")
        keep &= ev.notna()
    S = S.loc[keep, S.notna().loc[keep].all()]
    S = S.loc[:, S.std() > 0]
    if S.shape[1] < 2:
        raise AnalysisError("a stack needs at least two clocks with varying scores")
    names = list(S.columns)
    scales = {reg.get(c).scale_type for c in names}
    yv = y[keep].to_numpy()
    n = len(yv)
    if n < 2 * outer_folds:
        raise AnalysisError(f"{n} samples; {outer_folds} folds need at least {2 * outer_folds}")

    if method == "mean":
        if len(scales) > 1:
            raise IllegalOperationError(
                f"a plain mean across scale types {sorted(scales)} has no unit: averaging "
                "years with a pace or a log hazard is the arithmetic FALCONAge exists "
                "to refuse. Use method='nnls' or 'ridge', which standardise each clock "
                "and take the target's scale.")
    g = None if groups is None else result.obs.loc[keep, groups].to_numpy()
    rng = np.random.default_rng(seed)
    fold = folds(n, outer_folds, rng, g)
    Sv = S.to_numpy()
    pred = np.empty(n)
    single = {c: np.empty(n) for c in names}
    plain = np.empty(n)
    for f in range(outer_folds):
        te = fold == f
        mu, sd = Sv[~te].mean(axis=0), Sv[~te].std(axis=0)
        Ztr, Zte = (Sv[~te] - mu) / sd, (Sv[te] - mu) / sd
        if method == "mean":
            pred[te] = Sv[te].mean(axis=1)
        elif method == "cox":
            from .association import _cox_breslow
            beta, _ = _cox_breslow(Ztr, yv[~te], ev[keep].to_numpy()[~te])
            pred[te] = Zte @ beta
        else:
            b0, w = _fit(Ztr, yv[~te], method, ridge)
            pred[te] = b0 + Zte @ w
        # each clock alone, calibrated to the target on the training part
        for j, c in enumerate(names):
            a1, a0 = np.polyfit(Sv[~te, j], yv[~te], 1)
            single[c][te] = a0 + a1 * Sv[te, j]
        plain[te] = Sv[te].mean(axis=1) if len(scales) == 1 else np.nan

    mu, sd = Sv.mean(axis=0), Sv.std(axis=0)
    Z = (Sv - mu) / sd
    if method == "mean":
        b0, w = 0.0, np.full(len(names), 1.0 / len(names))
        b0 = float(mu @ w)
        w = w * sd                       # mean of raw scores in standardised form
    elif method == "cox":
        from .association import _cox_breslow
        w, _ = _cox_breslow(Z, yv, ev[keep].to_numpy())
        b0 = 0.0
    else:
        b0, w = _fit(Z, yv, method, ridge)

    if method == "cox":
        e = ev[keep].to_numpy()
        mets = {"n": float(n), "c_index": _cindex(pred, yv, e)}
        comp = pd.DataFrame({c: {"c_index": _cindex(Sv[:, j], yv, e)}
                             for j, c in enumerate(names)}).T
        scale = "mortality_log_hazard"
    else:
        mets = metrics(pred, yv)
        comp = pd.DataFrame({c: metrics(single[c], yv) for c in names}).T
        if not np.isnan(plain).any():
            comp.loc["plain mean"] = metrics(plain, yv)
        scale = (next(iter(scales)) if method == "mean" else
                 ("age_years" if target.lower() in ("age", "chronological_age") else "relative_score"))
    comp = comp.sort_values(comp.columns[1] if method != "cox" else "c_index",
                            ascending=method == "cox" and False or method != "cox")
    notes = [f"Stack of {len(names)} clocks by {method}; every reported number is out "
             f"of fold ({outer_folds} folds"
             + (f", grouped by {groups!r}" if groups else "") + ").",
             "Do not test a group difference on the samples the stack was fitted on."]
    return StackFit(
        clocks=names, method=method, target=target, intercept=float(b0),
        weights=pd.Series(w, index=names, name="weight"),
        means=pd.Series(mu, index=names), sds=pd.Series(sd, index=names),
        predictions=pd.Series(pred, index=S.index, name="stack"),
        metrics=mets, comparison=comp, scale_type=scale, notes=notes)
