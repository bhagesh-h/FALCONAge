"""Longitudinal interventional studies, and how to get them.

WHY THIS IS SEPARATE FROM `download`. `download` dispatches on the SHAPE of an
accession: `GSE182991` is a GEO series because it starts with GSE. An
intervention study has no such shape. "CALERIE" is not an accession, the
dataset behind it may sit in GEO or behind an application form, and which of
those it is cannot be read off the name. So the name is resolved against a
curated list first, and only then handed to the transfer layer.

WHY THE REFUSALS ARE THE INTERESTING PART. Three of the four access routes are
not a download. A TruDiagnostic partner dataset needs approval from two parties
and a controlled-access one needs IRB review, so the useful thing a tool can do
is name the route and stop, rather than emit a 403 and let the user guess
whether they typed the name wrong. That is the same reasoning the scoring path
uses for a clock whose coefficients cannot be redistributed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ..core.errors import DownloadError

DATA = Path(__file__).with_name("data") / "interventions.yaml"


@dataclass(frozen=True)
class InterventionStudy:
    """One longitudinal interventional dataset."""

    id: str
    name: str
    intervention: str
    category: str
    design: str
    access: str
    accession: str | None = None
    platform: str = ""
    size: str = ""
    paired: bool = True
    notes: str = ""
    citation: str = ""
    doi: str | None = None

    @property
    def downloadable(self) -> bool:
        """Whether this one can be fetched without a human in the loop."""
        return self.access == "open" and bool(self.accession)


@dataclass(frozen=True)
class InterventionCatalogue:
    studies: tuple[InterventionStudy, ...]
    routes: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __iter__(self):
        return iter(self.studies)

    def __len__(self) -> int:
        return len(self.studies)

    def __contains__(self, study_id: str) -> bool:
        return any(s.id == study_id for s in self.studies)

    def get(self, study_id: str) -> InterventionStudy:
        key = study_id.strip().lower()
        for s in self.studies:
            if s.id == key:
                return s
        known = ", ".join(sorted(s.id for s in self.studies))
        raise DownloadError(
            f"{study_id!r} is not a catalogued intervention study.\n"
            f"  Known: {known}\n"
            "  falconage interventions list  shows what each one is.")

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "id": s.id, "name": s.name, "intervention": s.intervention,
            "category": s.category, "design": s.design, "access": s.access,
            "accession": s.accession or "", "platform": s.platform,
            "size": s.size, "downloadable": s.downloadable,
            "citation": s.citation, "doi": s.doi or "",
        } for s in self.studies])

    def how_to_get(self, study_id: str) -> str:
        """The actual next step for this study, whatever that step is."""
        s = self.get(study_id)
        route = self.routes.get(s.access, {})
        lines = [f"{s.name} ({s.id})",
                 f"  intervention  {s.intervention}",
                 f"  design        {s.design}",
                 f"  route         {route.get('label', s.access)}"]
        if s.accession:
            lines.append(f"  accession     {s.accession}")
        if s.size:
            lines.append(f"  size          {s.size}")
        how = " ".join(str(route.get("how", "")).split())
        if how:
            lines.append(f"  how           {how}")
        if s.downloadable:
            lines.append(f"  command       falconage download {s.accession}")
        return "\n".join(lines)


def load(path: str | Path | None = None) -> InterventionCatalogue:
    """The catalogue that ships with the package."""
    spec = yaml.safe_load(Path(path or DATA).read_text(encoding="utf-8"))
    studies = tuple(
        InterventionStudy(
            id=e["id"], name=e["name"], intervention=e["intervention"],
            category=e.get("category", ""), design=e.get("design", ""),
            access=e.get("access", "open"), accession=e.get("accession"),
            platform=e.get("platform", ""), size=e.get("size", ""),
            paired=bool(e.get("paired", True)),
            notes=" ".join(str(e.get("notes", "")).split()),
            citation=e.get("citation", ""), doi=e.get("doi"),
        )
        for e in spec["studies"]
    )
    return InterventionCatalogue(studies=studies,
                                 routes=spec.get("access_routes", {}))


def interventions(category: str | None = None,
                  downloadable: bool | None = None) -> pd.DataFrame:
    """The catalogue as a table.

    >>> fa.download.interventions()
    >>> fa.download.interventions(downloadable=True)
    """
    df = load().to_frame()
    if category:
        df = df[df["category"] == category]
    if downloadable is not None:
        df = df[df["downloadable"] == downloadable]
    return df.reset_index(drop=True)


def download_intervention(study_id: str, *, cache_dir: Path | None = None,
                          dry_run: bool = False, **kw):
    """Fetch a catalogued intervention study, or say exactly why it cannot.

    An application form is not a failure the caller can retry, so a study behind
    one raises with the form's URL rather than returning an empty result that
    reads like a network problem.
    """
    from . import download as _download           # late, to avoid a cycle

    cat = load()
    s = cat.get(study_id)
    if not s.downloadable:
        raise DownloadError(
            f"{s.name} is not a direct download.\n\n"
            + cat.how_to_get(s.id)
            + "\n\n  Once the files are local, read them directly: "
              "falconage.io.read(path).")
    return _download(s.accession, cache_dir=cache_dir, dry_run=dry_run, **kw)
