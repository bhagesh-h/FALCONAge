"""The clock registry: metadata, availability tiers, and coefficient resolution.

The registry is a data artefact, not code. It ships inside the wheel as one
YAML file plus a directory of coefficient CSVs, it carries its own version
independent of the package version, and every result records which version
produced it. A bug fix in the scoring loop must not silently change which
numbers a result came from, and a coefficient correction must be visible even
when no code changed.

Three availability tiers, and the difference is about redistribution rights
rather than about difficulty:

``A``
    Coefficients ship in the wheel. Extracted from a named source, checksummed,
    and usable offline.
``B``
    Architecture and metadata ship; coefficients are fetched from the authors'
    URL on first use and cached. Scoring one offline raises.
``C``
    Scaffold only. The architecture is implemented and testable, the
    coefficients are research-use-only or have no traceable public source, and
    FALCONAge will not redistribute them. The error names where to obtain a
    file and which open clocks answer the same question.
"""

from __future__ import annotations

import csv
import difflib
import functools
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .._version import REGISTRY_VERSION
from ..core.errors import ClockNotFoundError, RegistryError, WeightsUnavailableError

DATA_DIR = Path(__file__).with_name("data")

#: Which downstream operations each scale admits. Enforced in analysis/, not
#: here, but declared here because it is a property of the clock.
LEGAL_OPS: dict[str, set[str]] = {
    "age_years":            {"acceleration", "residual", "correlate", "difference", "mean"},
    # Years, and a slope of about one against chronological age -- but no fixed
    # origin. `predicted - chronological` is therefore not a quantity; the
    # residual, fitted inside the dataset at hand, is.
    #
    # This is not a hypothetical category. Ying's DamAge and AdaptAge carry
    # intercepts of +543.43 and -511.97, and measured across three healthy
    # cohorts in the test corpus their median bias against chronological age
    # swings 162 years each -- while Horvath's swings 15. The two move as near
    # mirror images (r = -0.975) and their sum swings only 33, which says the
    # cause is one dataset-level offset amplified by those two huge opposing
    # intercepts rather than two separately broken clocks. Their slopes are
    # fine: 0.967 for DamAge pooled, better than DNAmPhenoAge's 1.199.
    #
    # So the clock works and its zero does not travel. Calling the scale
    # `relative_score` would be the other error -- it would forbid the residual
    # and the group difference, which are exactly the operations the paper and
    # the field use, and it would contradict a `unit` of years and a
    # `training_target` of chronological age, both of which are accurate.
    "age_years_relative":   {"residual", "correlate", "difference", "mean"},
    "gestational_weeks":    {"acceleration", "residual", "correlate", "difference", "mean"},
    "mortality_log_hazard": {"correlate", "rank", "hazard_ratio", "mean"},
    "pace_ratio":           {"correlate", "rank", "difference", "mean"},
    "telomere_kb":          {"acceleration", "correlate", "difference", "mean"},
    "proportion":           {"correlate", "difference", "mean", "compositional"},
    "divisions":            {"correlate", "rank", "difference", "mean"},
    "relative_score":       {"correlate", "rank"},
}


@dataclass(frozen=True)
class CoefficientSource:
    file: str | None = None
    url: str | None = None
    sha256: str | None = None
    provenance: str = ""
    primary_source_traced: bool = False
    redistributable: bool | None = None
    why: str = ""
    obtain: str = ""
    alternatives: tuple[str, ...] = ()


@dataclass(frozen=True)
class Reliability:
    """How repeatable a clock is, split into the two things that word means.

    These are separate properties and they do not track together, which is the
    entire reason both fields exist rather than one called ``icc``:

    * **technical** -- the same biological sample, assayed twice. Measures the
      assay and the model's sensitivity to array noise.
    * **biological** -- the same person, sampled again a short time later, with
      nothing clinically relevant having happened in between. Measures whether
      the number is a property of the person or of the morning.

    A clock can be excellent on the first and poor on the second, and the
    clocks most used in intervention work are exactly where that gap is widest
    (bioRxiv 2025.10.13.682176). A single reliability figure hides that.

    Most entries have neither value. That is honest: an unpublished ICC is not
    a good ICC, and ``None`` here means "not established in this registry",
    never "fine".
    """

    technical_icc: float | None = None
    biological_icc: float | None = None
    source: str = ""
    note: str = ""

    @property
    def known(self) -> bool:
        return self.technical_icc is not None or self.biological_icc is not None


@dataclass(frozen=True)
class Clock:
    """One registry entry."""

    id: str
    name: str
    year: int | None
    species: str
    data_type: str
    generation: str
    tissue: tuple[str, ...]
    platform: tuple[str, ...]
    predicts: tuple[str, ...]
    training_target: tuple[str, ...]
    unit: tuple[str, ...]
    scale_type: str
    model_type: str
    population: str
    n_features: int | None
    requires_fp64: bool
    availability: str
    citation: str
    doi: str
    notes: str
    coefficient_source: CoefficientSource
    preprocess: tuple[dict[str, Any], ...] = ()
    postprocess: tuple[dict[str, Any], ...] = ()
    formula: str | None = None
    requires_covariates: tuple[str, ...] = ()
    requires_reference: bool = False
    #: True when the clock's preprocessing centres or scales against the cohort
    #: it is given, so a single sample has no defined answer.
    #:
    #: The transcriptomic clocks median-centre across all samples within a
    #: dataset, which means the score for one sample depends on which other
    #: samples were scored beside it. Run one sample alone and the centring is
    #: against itself: every value becomes zero and the clock returns its
    #: intercept, confidently, for anybody. That is a refusal rather than a
    #: warning, and nothing in the arithmetic can notice it -- which is why it
    #: is a declared property of the clock instead of a check inside a model.
    requires_cohort: bool = False
    #: Smallest cohort the centring is meaningful over. Only read when
    #: ``requires_cohort`` is set.
    min_samples: int = 1
    #: How hard to push back when the sample's specimen is not one of
    #: :attr:`tissue`: ``"warn"`` (default), ``"refuse"``, or ``"allow"``.
    #:
    #: There is no second ``trained_tissue`` field. ``tissue`` already records
    #: what the clock was fitted on -- "placenta", "buccal epithelium",
    #: "multi-tissue" -- and duplicating it under another name would give two
    #: places for the same fact to be wrong. Only the policy is new.
    #:
    #: ``"refuse"`` is for clocks whose training tissue has no counterpart
    #: elsewhere: a placenta clock on blood, a cord-blood clock on an adult.
    #: Those are not offsets, they are category errors. ``"allow"`` is for the
    #: genuinely pan-tissue clocks, where the check has nothing to say.
    tissue_policy: str = "warn"
    known_discrepancies: tuple[str, ...] = ()
    reliability: Reliability = field(default_factory=Reliability)

    @property
    def legal_operations(self) -> set[str]:
        return LEGAL_OPS.get(self.scale_type, {"correlate"})

    @property
    def is_scaffold(self) -> bool:
        return self.availability == "C"

    @property
    def ships_coefficients(self) -> bool:
        return self.availability == "A" and self.coefficient_source.file is not None

    def cite(self, style: str = "plain") -> str:
        if style == "bibtex":
            key = f"{self.id}{self.year or ''}"
            return (f"@article{{{key},\n  title = {{{self.name}}},\n"
                    f"  year = {{{self.year}}},\n  note = {{{self.citation}}},\n"
                    f"  doi = {{{self.doi}}}\n}}")
        return f"{self.citation} {self.doi}".strip()

    def __repr__(self) -> str:  # pragma: no cover - display only
        n = f"{self.n_features} features" if self.n_features else "features unknown"
        return (f"Clock({self.id!r}, {self.year}, {self.scale_type}, "
                f"tier {self.availability}, {n})")


#: Policies that are a property of the tissue list rather than of the clock, so
#: the YAML does not have to repeat them 161 times. An explicit
#: ``tissue_policy:`` in an entry always wins.
_LEGAL_POLICIES = ("allow", "warn", "refuse")


def _tissue_policy(cid: str, entry: dict[str, Any]) -> str:
    """Resolve a clock's tissue policy, defaulting from its declared tissues.

    A clock that lists no tissue, or lists multi-tissue, has nothing to enforce
    and gets ``"allow"``. Everything else gets ``"warn"`` unless the entry says
    otherwise. Deriving the common case means the only rows that carry the field
    are the ones where the answer is not obvious from the data already there.
    """
    given = entry.get("tissue_policy")
    if given is not None:
        if given not in _LEGAL_POLICIES:
            raise RegistryError(
                f"{cid}: tissue_policy {given!r}; expected one of {_LEGAL_POLICIES}")
        return str(given)
    tis = {str(t).strip().lower() for t in _tuple(entry.get("tissue"))}
    if not tis or {"multi-tissue", "multi tissue", "pan-tissue"} & tis:
        return "allow"
    return "warn"


def _tuple(v: Any) -> tuple:
    if v is None:
        return ()
    if isinstance(v, (list, tuple)):
        return tuple(v)
    return (v,)


class ClockRegistry:
    """Loaded catalogue. Immutable except for user-registered local weights."""

    def __init__(self, clocks: dict[str, Clock], version: str, path: Path) -> None:
        self._clocks = clocks
        self.version = version
        self.path = path
        #: clock_id -> (path, sha256) supplied by the user for a tier C clock.
        self._local: dict[str, tuple[Path, str]] = {}

    # -- construction ------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: Path | None = None) -> ClockRegistry:
        p = Path(path) if path else DATA_DIR / "clocks.yaml"
        if not p.exists():
            raise RegistryError(f"registry not found at {p}")
        doc = yaml.safe_load(p.read_text(encoding="utf-8"))
        if doc.get("schema_version") != 1:
            raise RegistryError(
                f"{p}: schema_version {doc.get('schema_version')!r}; this build "
                "reads schema 1 only"
            )
        version = str(doc.get("registry_version", REGISTRY_VERSION))

        clocks: dict[str, Clock] = {}
        for cid, e in (doc.get("clocks") or {}).items():
            cs = e.get("coefficient_source") or {}
            clocks[cid] = Clock(
                id=cid,
                name=e.get("name") or cid,
                year=e.get("year"),
                species=e.get("species") or "Homo sapiens",
                data_type=e.get("data_type") or "dna_methylation",
                generation=e.get("generation") or "other",
                tissue=_tuple(e.get("tissue")),
                platform=_tuple(e.get("platform")),
                predicts=_tuple(e.get("predicts")),
                training_target=_tuple(e.get("training_target")),
                unit=_tuple(e.get("unit")),
                scale_type=e.get("scale_type") or "relative_score",
                model_type=e.get("model_type") or "linear",
                population=e.get("population") or "",
                n_features=e.get("n_features"),
                requires_fp64=bool(e.get("requires_fp64", False)),
                availability=e.get("availability") or "B",
                citation=e.get("citation") or "",
                doi=e.get("doi") or "",
                notes=e.get("notes") or "",
                coefficient_source=CoefficientSource(
                    file=cs.get("file"), url=cs.get("url"), sha256=cs.get("sha256"),
                    provenance=cs.get("provenance") or "",
                    primary_source_traced=bool(cs.get("primary_source_traced", False)),
                    redistributable=cs.get("redistributable"),
                    why=cs.get("why") or "", obtain=cs.get("obtain") or "",
                    alternatives=_tuple(cs.get("alternatives")),
                ),
                preprocess=tuple(e.get("preprocess") or ()),
                postprocess=tuple(e.get("postprocess") or ()),
                formula=e.get("formula"),
                requires_covariates=_tuple(e.get("requires_covariates")),
                requires_reference=bool(e.get("requires_reference", False)),
                requires_cohort=bool(e.get("requires_cohort", False)),
                min_samples=int(e.get("min_samples", 1) or 1),
                tissue_policy=_tissue_policy(cid, e),
                known_discrepancies=_tuple(e.get("known_discrepancies")),
                reliability=Reliability(**{
                    k: v for k, v in (e.get("reliability") or {}).items()
                    if k in ("technical_icc", "biological_icc", "source", "note")
                }),
            )
        return cls(clocks, version, p)

    # -- lookup ------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._clocks)

    def __contains__(self, cid: object) -> bool:
        return cid in self._clocks

    def __iter__(self):
        return iter(self._clocks.values())

    def list(self) -> list[str]:
        return sorted(self._clocks)

    def get(self, clock_id: str) -> Clock:
        try:
            return self._clocks[clock_id]
        except KeyError:
            near = difflib.get_close_matches(clock_id, self._clocks, n=5, cutoff=0.5)
            raise ClockNotFoundError(clock_id, near) from None

    def search(self, term: str) -> list[Clock]:
        """Substring match over id, name, predicts, notes and citation."""
        t = term.lower()
        return [c for c in self._clocks.values() if t in " ".join(
            [c.id, c.name, " ".join(c.predicts), c.notes, c.citation]).lower()]

    def filter(self, **criteria: Any) -> list[Clock]:
        """Exact-match filter; tuple-valued fields match on membership.

        >>> reg.filter(availability="A", scale_type="age_years")
        """
        out = []
        for c in self._clocks.values():
            ok = True
            for k, want in criteria.items():
                if not hasattr(c, k):
                    raise RegistryError(f"no such clock field: {k!r}")
                have = getattr(c, k)
                if isinstance(have, tuple):
                    ok &= want in have
                else:
                    ok &= (have == want)
                if not ok:
                    break
            if ok:
                out.append(c)
        return sorted(out, key=lambda c: c.id)

    def compatible_with(self, data, *, min_coverage: float = 0.8) -> list[Clock]:
        """Clocks this dataset can actually be scored on.

        Compatibility is coverage, not platform. A clock trained on 450K runs
        perfectly well on EPIC data that happens to carry its probes, and fails
        on 450K data that has been filtered down to 20,000 probes. Only the
        feature list can answer the question, so tier A clocks are checked
        against it and tier B/C clocks are reported by declared platform, with
        the difference made explicit in the reason string.
        """
        out = []
        for c in self._clocks.values():
            if c.data_type != data.modality:
                continue
            if c.formula:
                need = set(c.requires_covariates)
                if need <= set(data.obs.columns) | set(data.X.columns):
                    out.append(c)
                continue
            if not self.has_coefficients(c.id):
                continue
            feats = self.feature_ids(c.id)
            if data.coverage(feats) >= min_coverage:
                out.append(c)
        return sorted(out, key=lambda c: c.id)

    # -- coefficients ------------------------------------------------------
    def has_coefficients(self, clock_id: str) -> bool:
        c = self.get(clock_id)
        if clock_id in self._local:
            return True
        return c.ships_coefficients and (DATA_DIR / c.coefficient_source.file).exists()

    def register_local_weights(self, clock_id: str, path: str | Path,
                               sha256: str | None = None) -> str:
        """Supply a coefficient file the package does not distribute.

        Validates against the scaffold -- two columns, no duplicate feature ids,
        finite values, and the declared feature count when the registry has one.
        A file that does not match is rejected with the mismatch named, which
        also makes this the way to check a coefficient set somebody handed you.

        Returns the SHA-256 of the accepted file; it goes into the run manifest
        as ``weights_source: user_supplied``, so a result computed from a
        licensed copy is distinguishable from one computed from a redistributed
        set.
        """
        c = self.get(clock_id)
        p = Path(path).expanduser()
        if not p.exists():
            raise RegistryError(f"no such file: {p}")

        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if sha256 and digest != sha256:
            raise RegistryError(
                f"{p}: SHA-256 is {digest}, you asserted {sha256}")

        feats, coefs = _read_coefficients(p)
        if len(feats) != len(set(feats)):
            raise RegistryError(f"{p}: duplicate feature ids")
        if not np.isfinite(coefs).all():
            raise RegistryError(f"{p}: contains non-finite coefficients")
        if c.n_features and len(feats) != c.n_features:
            raise RegistryError(
                f"{p}: {len(feats)} coefficients, but the registry declares "
                f"{c.n_features} for {clock_id}. Either this is the wrong file "
                "or the registry entry is stale -- both are worth knowing."
            )
        self._local[clock_id] = (p, digest)
        return digest

    def coefficients(self, clock_id: str) -> tuple[list[str], np.ndarray]:
        """Feature ids and coefficients, in the file's order.

        Raises
        ------
        WeightsUnavailableError
            For a tier C clock with nothing registered, or a tier B clock whose
            fetch has not happened. The message names the remedy.
        """
        c = self.get(clock_id)

        if clock_id in self._local:
            return _read_coefficients(self._local[clock_id][0])

        if c.availability == "A" and c.coefficient_source.file:
            p = DATA_DIR / c.coefficient_source.file
            if not p.exists():
                raise WeightsUnavailableError(
                    clock_id, f"{clock_id}: registry declares {p.name} but it is "
                              "missing from the installed package")
            return _read_coefficients(p)

        raise WeightsUnavailableError(clock_id, self.unavailable_message(clock_id))

    def unavailable_message(self, clock_id: str) -> str:
        """The text a user sees when a clock cannot be scored. Worth care."""
        c = self.get(clock_id)
        cs = c.coefficient_source

        if c.availability == "C":
            alts = [self.get(a) for a in cs.alternatives if a in self]
            alt_lines = "\n".join(
                f"    {a.id:<16}{a.data_type.replace('_', ' '):<20}"
                f"{a.n_features or '?'} features, tier {a.availability}"
                for a in alts)
            return (
                f"{clock_id} is a scaffold-only clock.\n\n"
                "  Its architecture is implemented and tested, but its coefficients "
                f"are\n  not distributed with FALCONAge.\n\n"
                f"  Why: {cs.why}\n\n"
                f"  Obtain them from: {cs.obtain}\n"
                f"  Then: falconage.registry.load().register_local_weights"
                f"({clock_id!r}, <path>)\n\n"
                + (f"  Open alternatives predicting the same thing:\n{alt_lines}\n"
                   if alt_lines else "")
            )

        # A tier B entry that has been looked into says so. The default text
        # below is true of most of them and useless for the ones where somebody
        # has already done the work: telling a reader "no source has been
        # established" when the registry holds the DOI, the file name and the
        # reason it is still not shipped wastes the search they are about to
        # repeat.
        if cs.why or cs.obtain:
            found = f"{clock_id} is tier B: its coefficients are not bundled.\n\n"
            if cs.why:
                found += f"  Why: {cs.why}\n\n"
            if cs.obtain:
                found += f"  The material is at: {cs.obtain}\n\n"
            found += (
                "  Registering it yourself is one call:\n"
                f"    falconage.registry.load().register_local_weights({clock_id!r}, <path>)\n"
                "  and the run manifest will record the file's SHA-256 as user-supplied."
            )
            return found

        return (
            f"{clock_id} is tier B: its coefficients are not bundled and no "
            "primary\n  source has been established for it yet.\n\n"
            "  FALCONAge v1.0 ships coefficients for the clocks whose source it "
            "could\n  name and check. Copying a coefficient set out of another "
            "package without\n  knowing where that package got it is how the "
            "eleven known paper-versus-\n  implementation discrepancies spread in "
            "the first place.\n\n"
            f"  falconage clocks list --tier A   shows the {len(self.filter(availability='A'))} "
            "that do ship.\n"
            "  Contributions of a traced extractor are welcome: see CONTRIBUTING.md."
        )

    @functools.lru_cache(maxsize=256)  # noqa: B019 - registry is process-lifetime
    def feature_ids(self, clock_id: str) -> tuple[str, ...]:
        return tuple(self.coefficients(clock_id)[0])

    def weight_record(self, clock_id: str) -> dict[str, Any]:
        """What the run manifest records about this clock's coefficients."""
        c = self.get(clock_id)
        if clock_id in self._local:
            p, digest = self._local[clock_id]
            return {"source": "user_supplied", "path": str(p), "sha256": digest,
                    "tier": c.availability}
        return {"source": "bundled", "path": c.coefficient_source.file or "",
                "sha256": c.coefficient_source.sha256 or "", "tier": c.availability,
                "provenance": c.coefficient_source.provenance,
                "primary_source_traced": c.coefficient_source.primary_source_traced}

    # -- reporting ---------------------------------------------------------
    def summary(self):
        import pandas as pd

        return pd.DataFrame([{
            "id": c.id, "year": c.year, "generation": c.generation,
            "scale_type": c.scale_type, "n_features": c.n_features,
            "availability": c.availability, "data_type": c.data_type,
            "traced": c.coefficient_source.primary_source_traced,
        } for c in self._clocks.values()]).set_index("id").sort_index()

    def untraced(self) -> list[Clock]:
        """Clocks whose coefficients have no established primary source."""
        return [c for c in self._clocks.values()
                if not c.coefficient_source.primary_source_traced]


def _read_coefficients(path: Path) -> tuple[list[str], np.ndarray]:
    feats: list[str] = []
    vals: list[float] = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rdr = csv.reader(fh)
        header = next(rdr, None)
        if header is None:
            raise RegistryError(f"{path}: empty coefficient file")
        if [h.strip().lower() for h in header[:2]] != ["feature_id", "coefficient"]:
            # Accept the header the upstream packages use, but say so, because a
            # silently accepted column order is how a coefficient column gets
            # read as a feature id.
            if len(header) >= 2 and header[0].strip().lower() in ("cpgmarker", "cpg", "probe"):
                pass
            else:
                raise RegistryError(
                    f"{path}: header is {header[:2]}, expected "
                    "['feature_id', 'coefficient']")
        for row in rdr:
            if len(row) < 2 or not row[0].strip():
                continue
            feats.append(row[0].strip())
            vals.append(float(row[1]))
    return feats, np.asarray(vals, dtype=np.float64)


@functools.lru_cache(maxsize=4)
def load(path: str | None = None) -> ClockRegistry:
    """Load the packaged registry. Cached for the life of the process."""
    return ClockRegistry.from_yaml(Path(path) if path else None)


# ---------------------------------------------------------------------------
# the evidence pack
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _evidence_doc() -> dict:
    p = DATA_DIR / "evidence.yaml"
    if not p.exists():
        return {"sources": {}, "general": [], "clocks": {}}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def evidence(clock_id: str | None = None) -> pd.DataFrame:
    """What a score on this clock has been shown to predict, and how strongly.

    The step between "DunedinPACE = 1.08" and something a reader can act on.
    Every row carries the study it came from, that study's design, and a DOI --
    a row without one is not admitted, because an uncheckable effect size looks
    identical to a checkable one and is worth less than nothing.

    Only effect sizes from studies that tested many clocks on one cohort under
    one protocol are seeded. A single-clock paper reports its own clock
    favourably and its numbers are not comparable to the row above.

    Parameters
    ----------
    clock_id
        One clock, or ``None`` for the whole table including the cross-clock
        findings, which are marked with clock ``*``.
    """
    doc = _evidence_doc()
    src = doc.get("sources", {})
    rows = []

    def _rows(cid: str, entries):
        for e in entries or ():
            s = src.get(e.get("source"), {})
            rows.append({
                "clock": cid,
                "outcome": e.get("outcome", ""),
                "measure": e.get("measure", ""),
                "value": e.get("value"),
                "per": e.get("per", ""),
                "n": e.get("n") or s.get("n"),
                "followup_years": e.get("followup"),
                "population": e.get("population", ""),
                "note": e.get("note", ""),
                "citation": s.get("citation", ""),
                "doi": s.get("doi", ""),
                "design": s.get("design", ""),
            })

    if clock_id is None:
        _rows("*", doc.get("general"))
        for cid, entries in (doc.get("clocks") or {}).items():
            _rows(cid, entries)
    else:
        _rows(clock_id, (doc.get("clocks") or {}).get(clock_id))

    out = pd.DataFrame(rows)
    if not out.empty:
        missing = out[out["doi"] == ""]
        if len(missing):
            raise RegistryError(
                f"{len(missing)} evidence row(s) have no DOI: "
                f"{sorted(set(missing['clock']))}. Every row must be checkable.")
    return out
