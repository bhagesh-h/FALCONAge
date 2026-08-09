"""The scoring loop and its result object."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import pandas as pd

# Re-exported so a caller holding a result can reach the versions that made
# it without importing two modules. Aliased to themselves so the re-export is
# explicit rather than an import that only looks unused.
from .._version import REGISTRY_VERSION as REGISTRY_VERSION
from .._version import __version__ as __version__
from ..core.backend import resolve
from ..core.container import FalconData
from ..core.errors import FalconError, FeatureCoverageError, ScoringError, WeightsUnavailableError
from ..core.logging import WarningCollector, get_logger
from ..core.manifest import RunManifest
from ..models import build
from ..registry import load as load_registry

__all__ = ["FalconResult", "combine", "score"]

log = get_logger("score")


@dataclass
class FalconResult:
    """Scores plus everything needed to say where they came from.

    ``scores`` is samples x clocks. Everything else -- coverage, warnings,
    weight digests, the resolved config -- hangs off the manifest, so a result
    written to disk and read back is self-describing.
    """

    scores: pd.DataFrame
    obs: pd.DataFrame
    manifest: RunManifest
    registry: Any
    coverage: dict[str, dict[str, Any]] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)

    # -- shapes ------------------------------------------------------------
    def wide(self) -> pd.DataFrame:
        return self.scores

    def long(self) -> pd.DataFrame:
        """One row per sample per clock, with the scale and provenance attached.

        The scale column is not decoration: it is what stops a reader averaging
        a mortality log-hazard with an age in years because both were numbers in
        a column called ``value``.
        """
        rows = []
        for cid in self.scores.columns:
            c = self.registry.get(cid)
            cov = self.coverage.get(cid, {})
            for sid, v in self.scores[cid].items():
                rows.append({
                    "sample_id": sid,
                    "clock": cid,
                    "value": v,
                    "unit": ", ".join(c.unit) or "",
                    "scale_type": c.scale_type,
                    "generation": c.generation,
                    "predicts": ", ".join(c.predicts),
                    "n_features": c.n_features,
                    "coverage": cov.get("coverage"),
                    "mass_coverage": cov.get("mass_coverage"),
                    "n_imputed": cov.get("n_imputed"),
                    "availability": c.availability,
                    "registry_version": self.manifest.registry_version,
                    "falconage_version": self.manifest.falconage_version,
                })
        return pd.DataFrame(rows)

    def qc(self) -> pd.DataFrame:
        rows = []
        for cid, cov in self.coverage.items():
            c = self.registry.get(cid)
            rows.append({"clock": cid, "n_features": c.n_features, **cov})
        for cid, why in self.skipped.items():
            rows.append({"clock": cid, "skipped": why})
        return pd.DataFrame(rows)

    def summary(self) -> pd.DataFrame:
        d = self.scores.describe().T
        d["scale_type"] = [self.registry.get(c).scale_type for c in d.index]
        d["coverage"] = [self.coverage.get(c, {}).get("coverage") for c in d.index]
        d["mass_coverage"] = [self.coverage.get(c, {}).get("mass_coverage")
                              for c in d.index]
        return d

    # -- how to read it ----------------------------------------------------
    #: The limit that belongs with every number this object holds. It is on the
    #: documentation site, which is not where a person looks while reading a
    #: table in a notebook.
    CAVEAT = (
        "Aging clocks are population-research instruments. A score is "
        "interpretable against a comparison group; it is not a diagnostic "
        "statement about one person, and no clock here is validated for a "
        "clinical decision. Reliability varies sharply by clock, and technical "
        "and biological reliability are separate properties -- see "
        "result.interpretation()."
    )

    def interpretation(self) -> pd.DataFrame:
        """One row per clock: what the number is, and what may be done with it.

        Everything needed to read a score, in the object holding the score.
        Scale and unit say what the quantity is; ``legal_operations`` says which
        arithmetic the package will perform on it; reliability says how much of
        a within-person change is likely to be the assay; ``caveats`` carries
        any documented disagreement between the clock's paper and the
        coefficients that circulate for it.
        """
        rows = []
        for cid in self.scores.columns:
            c = self.registry.get(cid)
            cov = self.coverage.get(cid, {})
            r = c.reliability
            rows.append({
                "clock": cid,
                "predicts": ", ".join(c.predicts),
                "scale_type": c.scale_type,
                "unit": ", ".join(c.unit) or "",
                "legal_operations": ", ".join(sorted(c.legal_operations)),
                "coverage": cov.get("coverage"),
                "mass_coverage": cov.get("mass_coverage"),
                "technical_icc": r.technical_icc,
                "biological_icc": r.biological_icc,
                "reliability_note": r.note,
                "caveats": " ".join(c.known_discrepancies),
                "tier": c.availability,
            })
        return pd.DataFrame(rows).set_index("clock")

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (f"FalconResult({self.scores.shape[0]} samples x "
                f"{self.scores.shape[1]} clocks, {len(self.skipped)} skipped, "
                f"{len(self.manifest.warnings)} warning(s))\n"
                f"  {self.CAVEAT.splitlines()[0][:96]}...\n"
                "  .interpretation() for scale, units, reliability and caveats "
                "per clock")

    def write(self, outdir):
        from ..io import write_results

        return write_results(self, outdir)


def _resolve_clocks(registry, data, clocks, min_coverage: float) -> tuple[list[str], dict[str, str]]:
    """Turn the ``clocks=`` argument into a concrete list, and say what it dropped."""
    skipped: dict[str, str] = {}

    if clocks == "compatible":
        chosen = [c.id for c in registry.compatible_with(data, min_coverage=min_coverage)]
        for c in registry:
            if c.id not in chosen and c.data_type == data.modality:
                if c.availability == "C":
                    skipped[c.id] = "scaffold: coefficients not distributed (see README section 6)"
                elif not registry.has_coefficients(c.id):
                    skipped[c.id] = "tier B: coefficients not bundled and no primary source traced"
                else:
                    skipped[c.id] = f"below the {min_coverage:.0%} feature-coverage floor"
        return chosen, skipped

    if clocks == "all":
        return [c.id for c in registry if c.data_type == data.modality], skipped

    if isinstance(clocks, str):
        clocks = [clocks]
    return list(clocks), skipped


def score(data: FalconData, clocks: str | Sequence[str] = "compatible", *,
          device: str = "auto", dtype: str | None = None,
          imputation: str = "reference", min_coverage: float = 0.8,
          reference=None, registry=None, caller: str = "python") -> FalconResult:
    """Score a dataset against one or more clocks.

    Parameters
    ----------
    clocks
        ``"compatible"`` scores everything this dataset can support and reports
        the rest as skipped with a reason. ``"all"`` attempts every clock of the
        right modality and fails loudly on the ones that cannot run. A list
        names them explicitly, and every name in it must work -- an explicit
        request is never silently dropped.
    imputation
        Passed to the per-clock feature alignment. See
        :func:`falconage.models.linear.align`; the short version is that zero is
        never a fill value.
    min_coverage
        The fraction of a clock's features that must be present. Below it, the
        clock is skipped (``clocks="compatible"``) or raises (explicit list).
    reference
        A fitted :class:`~falconage.models.clinical.KDMReference` or
        :class:`~falconage.models.clinical.HDReference`, for the two clinical
        clocks that have no fixed coefficients.

    Returns
    -------
    FalconResult
        Scores, per-clock coverage, and a manifest recording the device, the
        dtype, every coefficient digest and every warning.
    """
    reg = registry if registry is not None else load_registry()
    warns = WarningCollector()

    explicit = not isinstance(clocks, str)
    wanted, skipped = _resolve_clocks(reg, data, clocks, min_coverage)
    if not wanted:
        raise ScoringError(
            "no clocks to score.\n"
            f"  The dataset is {data.modality} with {data.n_features} features"
            + (f" on {data.platform}" if data.platform else "")
            + f", and none of the {len(reg)} registry entries reached the "
              f"{min_coverage:.0%} coverage floor.\n"
              "  falconage clocks list --tier A  shows what ships with coefficients."
        )

    manifest = RunManifest(caller=caller)
    scores: dict[str, pd.Series] = {}
    coverage: dict[str, dict[str, Any]] = {}

    for cid in wanted:
        c = reg.get(cid)
        spec = resolve(device, dtype, requires_fp64=c.requires_fp64)
        manifest.device, manifest.dtype, manifest.backend = spec.device, spec.dtype, spec.backend

        # A cohort-centred clock has no answer for one sample, and the failure
        # is silent: centring a single row against itself makes every value
        # zero, so the model returns its intercept -- the same confident number
        # for anybody. Refused rather than warned about, because a warning next
        # to a plausible number is read as a caveat on a result rather than as
        # notice that there is no result.
        if c.requires_cohort and data.n_samples < max(c.min_samples, 2):
            msg = (f"{cid} centres each feature across the samples it is given, "
                   f"so it is undefined for {data.n_samples} sample(s); it needs "
                   f"at least {max(c.min_samples, 2)}.\n"
                   "  Score the whole cohort in one call. Scoring samples "
                   "one at a time gives every one of them the same answer, "
                   "because each is centred against itself.")
            if explicit:
                raise ScoringError(msg)
            skipped[cid] = msg.splitlines()[0]
            warns.warn(msg.splitlines()[0], clock=cid, category="cohort")
            continue

        try:
            model = build(reg, cid)
            values, alignment = model.predict(
                data, spec, imputation=imputation, min_coverage=min_coverage
            ) if not c.formula else model.predict(data, spec, reference=reference)
        except (WeightsUnavailableError, FeatureCoverageError, FalconError) as exc:
            if explicit:
                raise
            skipped[cid] = str(exc).splitlines()[0]
            warns.warn(f"skipped: {str(exc).splitlines()[0]}", clock=cid, category="skipped")
            continue

        scores[cid] = values
        manifest.weights[cid] = reg.weight_record(cid)

        # Coverage is not validity. The mammalian array carries 96% of
        # Horvath2013's CpGs by design, so a zebra scores at high coverage and
        # gets a confident number out of a clock fitted on people. Nothing in
        # the arithmetic can catch that; only the declared species can.
        if (c.species and data.species and c.species != data.species
                and "multi" not in c.species.lower()):
            warns.warn(
                f"trained on {c.species}, scored on {data.species}. Feature "
                "coverage says nothing about whether the coefficients transfer "
                "across species -- set data.species deliberately if this is "
                "intended.",
                clock=cid, category="species")

        if alignment is not None:
            coverage[cid] = {
                "coverage": round(alignment.coverage, 6),
                "mass_coverage": (None if alignment.mass_coverage is None
                                  else round(alignment.mass_coverage, 6)),
                "n_present": int(alignment.present.sum()),
                "n_imputed": alignment.n_imputed,
                "imputation": alignment.imputation,
            }
            if alignment.coverage < 0.95:
                warns.warn(
                    f"{alignment.coverage:.1%} feature coverage; "
                    f"{alignment.n_imputed} value(s) imputed",
                    clock=cid, category="coverage")
            # Worth its own warning, separate from the count. The count can look
            # fine while the weights do not, and that combination is the one a
            # user is least likely to check for. Fires only when the two
            # measures actually disagree -- repeating a coverage warning in
            # different units teaches people to skim both.
            mc = alignment.mass_coverage
            if mc is not None and mc < 0.95 and mc < alignment.coverage - 0.02:
                worst = ", ".join(f"{f} ({s:.1%})"
                                  for f, s in alignment.missing_mass[:3])
                warns.warn(
                    f"features present cover {alignment.coverage:.1%} of the "
                    f"probe list but only {mc:.1%} of the model's total "
                    f"|coefficient|. Heaviest absent: {worst}",
                    clock=cid, category="coefficient_mass")
        else:
            coverage[cid] = {"coverage": 1.0, "mass_coverage": None,
                             "n_present": c.n_features or 0,
                             "n_imputed": 0, "imputation": "n/a"}

        for d in c.known_discrepancies:
            warns.warn(d, clock=cid, category="discrepancy")

    if not scores:
        raise ScoringError(
            "every requested clock was skipped.\n  " +
            "\n  ".join(f"{k}: {v}" for k, v in list(skipped.items())[:6]))

    df = pd.DataFrame(scores, index=data.sample_ids)
    manifest.coverage = coverage
    manifest.skipped = skipped
    manifest.warnings = warns.records
    manifest.config = {"clocks": clocks if isinstance(clocks, str) else list(clocks),
                       "imputation": imputation, "min_coverage": min_coverage,
                       "modality": data.modality, "platform": data.platform,
                       "n_samples": data.n_samples, "n_features": data.n_features}
    manifest.finish()

    log.info("scored %d clock(s) on %d sample(s); %d skipped",
             df.shape[1], df.shape[0], len(skipped))
    return FalconResult(scores=df, obs=data.obs.copy(), manifest=manifest,
                        registry=reg, coverage=coverage, skipped=skipped)


def combine(results: Iterable[FalconResult], *, keys: Sequence[str] | None = None,
            dataset_col: str = "dataset") -> FalconResult:
    """Stack per-dataset results into one, for a benchmark across studies.

    Datasets are scored separately and combined afterwards, never merged before
    scoring. Two reasons, and the second is the one that bites:

    * a 27K study and an EPIC study have different probe spaces, and the outer
      join of ten of them is a mostly-NaN matrix that no clock should see;
    * feature coverage is a property of a dataset. Merged, a clock that covers
      99% of one study and 40% of another reports a single meaningless average,
      and the AA2 test silently compares cases in the well-covered study against
      controls in the badly-covered one.

    Per-clock coverage is kept per dataset in ``uns`` and the manifests are
    merged, so a combined result still says which study contributed what.
    """
    results = list(results)
    if not results:
        raise ScoringError("combine() needs at least one result")
    names = list(keys) if keys else [
        str(r.obs[dataset_col].iloc[0]) if dataset_col in r.obs.columns else f"dataset{i}"
        for i, r in enumerate(results)]

    scores = pd.concat([r.scores for r in results], axis=0)
    obs = pd.concat([r.obs.assign(**{dataset_col: n}) if dataset_col not in r.obs.columns
                     else r.obs for r, n in zip(results, names)], axis=0)
    obs = obs.reindex(scores.index)

    m = RunManifest(caller=results[0].manifest.caller)
    m.device = results[0].manifest.device
    m.dtype = results[0].manifest.dtype
    m.backend = results[0].manifest.backend
    for r, n in zip(results, names):
        m.weights.update(r.manifest.weights)
        m.warnings.extend({**w, "dataset": n} for w in r.manifest.warnings)
        for cid, cov in r.coverage.items():
            m.coverage[f"{n}:{cid}"] = cov
        for cid, why in r.skipped.items():
            m.skipped[f"{n}:{cid}"] = why
    m.config = {"combined_from": names, "n_datasets": len(results)}
    m.finish()

    # Coverage on the combined object is the per-clock minimum across datasets:
    # a clock is only as usable as its worst study, and reporting the mean would
    # hide exactly the case the split above exists to expose.
    merged_cov: dict[str, dict[str, Any]] = {}
    for r in results:
        for cid, cov in r.coverage.items():
            cur = merged_cov.get(cid)
            if cur is None or (cov.get("coverage", 1) or 1) < (cur.get("coverage", 1) or 1):
                merged_cov[cid] = dict(cov)

    return FalconResult(scores=scores, obs=obs, manifest=m, registry=results[0].registry,
                        coverage=merged_cov,
                        skipped={k.split(":", 1)[-1]: v for k, v in m.skipped.items()})
