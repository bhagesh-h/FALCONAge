#!/usr/bin/env python3
"""Emit ``overlap.csv``: every clock, what it was trained on, and who it shares that with.

WHY THIS EXISTS. The registry answers "what is clock X" one clock at a time.
The question people actually arrive with is the other way round: *which clocks
were trained on the thing I have?* Whole blood from adults. Sorted monocytes.
A mortality endpoint rather than chronological age. Answering that meant reading
175 YAML entries, so it did not get answered and clocks got picked by fame
instead of by fit.

This writes one row per clock with the training attributes normalised into
classes you can filter on, plus the overlap columns that say how many other
clocks share each attribute and which ones. Two clocks with the same
``profile_key`` were fitted on the same kind of data, and that is the column to
sort on when choosing a panel.

WHAT "OVERLAP" MEANS HERE, IN TWO SENSES, BOTH PRESENT
------------------------------------------------------
*Training overlap*: same tissue, population, target and species. Registry
metadata, exact.

*Feature overlap*: the literal shared biomarkers, meaning the CpGs or clinical
markers two clocks actually have in common, as a Jaccard index. Only computable
for the clocks that ship coefficient vectors, which is 43 of 175; the rest get
blank rather than zero, because "no shared features" and "we cannot see this
clock's features" are different facts.

VERIFICATION. Every row carries ``verify_url``, and every one resolves or is
reported as not resolving. Registry DOIs are the primary source. Three
independent catalogues are cross-referenced so a reader can check the training
target against somebody who is not us:

- TranslAGE (translage.io), which publishes a "Trained Phenotype" per clock.
- biolearn (github.com/bio-learn/biolearn).
- methylCIPHER (github.com/HigginsChenLab/methylCIPHER).

Their snapshots live in ``python/tools/data/`` so this runs offline and so a
change in someone else's catalogue shows up as a diff rather than as a silent
change in our output. Refresh them with ``--refresh``.

WHAT IS NOT HERE, AND IT IS THE ONE PEOPLE ASK FOR FIRST. **Health status of
the training cohort is not a registry field**, and it is not recoverable from
one either: 5 of 175 entries mention health in free-text notes. There is
therefore no "trained on healthy individuals" column, because the honest value
for almost every row would be "unstated" and a column that is 97% unstated
invites the reader to treat the other 3% as a sample. ``training_cohort_note``
carries the free text where it exists and is empty otherwise. Filter on
``tissue_class`` and ``target_class``, which are real, and read the paper behind
``verify_url`` for cohort health.

Usage
-----
    python python/tools/build_overlap.py                 # writes overlap.csv
    python python/tools/build_overlap.py --check-urls    # + HTTP status per row
    python python/tools/build_overlap.py --refresh       # re-fetch the catalogues
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ROOT = HERE.parent.parent
OUT = ROOT / "overlap.csv"

TRANSLAGE_URL = "https://www.translage.io/data/datatables/clock_metadata.csv"
BIOLEARN_TREE = "https://api.github.com/repos/bio-learn/biolearn/git/trees/master?recursive=1"
METHYLCIPHER_R = "https://api.github.com/repos/HigginsChenLab/methylCIPHER/contents/R"

CATALOGUE_URLS = {
    "translage": "https://www.translage.io",
    "biolearn": "https://github.com/bio-learn/biolearn",
    "methylcipher": "https://github.com/HigginsChenLab/methylCIPHER",
}


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------
# Each class is a coarsening of the registry's own free-text vocabulary, chosen
# so that the classes are the distinctions people filter on. The raw value is
# kept in its own column beside every class, so a coarsening that turns out to
# be wrong for someone's question can be worked around rather than argued with.

def tissue_class(tissues: list[str]) -> str:
    t = {x.lower() for x in tissues}
    if not t:
        return "unstated"
    if "multi-tissue" in t:
        return "multi_tissue"
    if t & {"whole blood"}:
        return "whole_blood"
    if t & {"blood", "purified blood leukocytes", "sorted monocytes", "b cells",
            "cd4+ t cells", "cord blood", "plasma cell-free dna",
            "peripheral blood mononuclear cells"}:
        return "blood_derived"
    if any("brain" in x or "cortex" in x or "neuron" in x for x in t):
        return "brain"
    if t & {"placenta"} or any("cord" in x for x in t):
        return "perinatal_tissue"
    if any("cultur" in x or "fibroblast" in x for x in t):
        return "cell_culture"
    return "other_tissue"


def population_class(pop: str) -> str:
    p = (pop or "").lower()
    if not p:
        return "unstated"
    if "mice" in p or "mammalian" in p or "mouse" in p:
        return "non_human"
    if "cell culture" in p or "cell cultures" in p:
        return "cell_culture"
    if "pregnan" in p or "newborn" in p or "prenatal" in p:
        return "perinatal"
    if "children" in p or "adolescent" in p:
        return "children"
    if "centenarian" in p or "older" in p:
        return "older_adults"
    if "all ages" in p:
        return "all_ages"
    if "adult" in p or "human" in p:
        return "adults"
    return "other"


#: Training target to the class that decides what a score means. The split that
#: matters most is chronological_age against everything else: a clock fitted to
#: age is fitted to a quantity you already know, and §5.2 of the science page is
#: the argument for why that is a weaker thing to have measured.
#: Order is significant: the first rule that matches wins, so the specific
#: categories come before the general ones. ``disease`` in particular has to
#: precede everything, because it is the one class that says the training cohort
#: was *not* healthy, and that is the closest thing this table has to the health
#: filter the registry does not record.
_TARGET_RULES = [
    ("disease", ("carcinoma", "cancer", "alzheimer", "cardiovascular disease",
                 "depressive disorder", "dementia", "diabetes", "tumour", "tumor")),
    # A PC clock is refitted to reproduce another clock's output, so what it was
    # trained on is that clock, not a phenotype. Collapsing these into
    # chronological_age would misreport a dozen rows as primary age predictors.
    ("clock_output", ("clock output", "grimage output", "dnamtl output")),
    ("sex_or_chromosome", ("sex", "x-chromosome", "y-chromosome", "chromosome")),
    ("chronological_age", ("chronological age", "relative age", "gestational age",
                           "age acceleration", "electronic medical record age",
                           "retroelement methylation age")),
    ("mortality", ("mortality", "lifespan", "survival")),
    ("pace_of_aging", ("pace of aging", "rate of aging")),
    ("cell_composition", ("cell-type proportion", "cell-type-specific",
                          "cell type proportion")),
    ("mitotic", ("population doubling", "replicative history", "cell division",
                 "mitotic", "cell passage", "senescence")),
    ("phenotype", ("phenotypic age", "biological age", "frailty", "grip strength",
                   "gait speed", "intrinsic capacity", "disability", "vo2max",
                   "body fat", "physical")),
    ("exposure", ("smoking", "alcohol", "body mass index", "waist-to-hip",
                  "education", "diet", "stress", "pack years")),
    ("protein_or_analyte", ("leptin", "cystatin", "adrenomedullin", "gdf15",
                            "growth differentiation factor", "pai-1",
                            "plasminogen activator", "timp", "b2m",
                            "beta-2-microglobulin", "beta-2 microglobulin",
                            "interleukin", "hemoglobin", "a1c", "telomere",
                            "cholesterol", "hdl", "protein", "crp")),
]


def target_class(targets: list[str]) -> str:
    joined = " ; ".join(targets).lower()
    if not joined or joined == "not applicable":
        return "unstated"
    # Matched with and without separators, because the same analyte is written
    # "GDF-15" in one entry and "gdf15" in another and neither spelling is more
    # correct than the other.
    flat = re.sub(r"[^a-z0-9]", "", joined)
    for label, needles in _TARGET_RULES:
        if any(nd in joined or re.sub(r"[^a-z0-9]", "", nd) in flat
               for nd in needles):
            return label
    return "other"


def _norm(name: str) -> str:
    """Loose key for matching our ids against someone else's clock names."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


#: Our clock id to the name a third-party catalogue publishes it under, where the
#: two genuinely differ rather than merely being punctuated differently.
#:
#: These are not typos to be normalised away. Horvath's 2013 pan-tissue clock is
#: "Horvath1" everywhere else and "horvath2013" here; his 2018 skin-and-blood
#: clock is "Horvath2"; PhenoAge is filed under its published name and under
#: DNAmPhenoAge. Matching without this map silently reported the field's five
#: most-used clocks as appearing in no external catalogue, which was wrong in a
#: direction that made our own registry look more isolated than it is.
CATALOGUE_ALIASES = {
    "horvath2013": ("Horvath1", "Horvathv1"),
    "skinandblood": ("Horvath2", "Horvathv2"),
    "dnamphenoage": ("PhenoAge",),
    "zhangen": ("Zhang_EN", "Zhang2019", "Zhang"),
    "zhangblup": ("Zhang_BLUP", "Zhang2019", "Zhang"),
    "zhangmortality": ("Zhang_10", "ZhangMortality"),
    "corticalclock": ("DNAmClockCortical",),
    "epitoc1": ("EpiTOC1", "EpiToc"),
    "epitoc2": ("EpiTOC2", "EpiToc2"),
    "hrsinchphenoage": ("HRSInCHPhenoAge",),
    "grimage": ("GrimAgeV1",),
    "grimage2": ("GrimAgeV2",),
}


def _permuted_keys(name: str) -> set[str]:
    """Every word order a catalogue might have used for a compound clock name.

    biolearn files `mccartneyalcohol` as `AlcoholMcCartney`, and the two are the
    same clock written the other way round. Splitting on camel case and
    separators and then trying each ordering matches the whole McCartney family
    without an entry per clock, which is nine rows this would otherwise miss.
    """
    from itertools import permutations

    parts = [p for p in re.split(r"[^A-Za-z0-9]+|(?<=[a-z0-9])(?=[A-Z])", str(name)) if p]
    if not 1 < len(parts) <= 3:
        return {_norm(name)}
    return {_norm("".join(order)) for order in permutations(parts)} | {_norm(name)}


# ---------------------------------------------------------------------------
# external catalogues
# ---------------------------------------------------------------------------
def refresh_catalogues() -> None:
    DATA.mkdir(parents=True, exist_ok=True)

    def get(url: str) -> bytes:
        req = urllib.request.Request(
            url, headers={"User-Agent": "falconage-overlap/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read()

    (DATA / "translage_clock_metadata.csv").write_bytes(get(TRANSLAGE_URL))
    tree = get(BIOLEARN_TREE).decode("utf-8", "replace")
    models = sorted(set(re.findall(
        r'"path":\s*"biolearn/test/data/expected_model_outputs/([^"]+)\.csv"', tree)))
    (DATA / "biolearn_models.txt").write_text("\n".join(models) + "\n")
    r_dir = get(METHYLCIPHER_R).decode("utf-8", "replace")
    files = sorted(set(re.findall(r'"name":\s*"([^"]+)\.R"', r_dir)))
    (DATA / "methylcipher_files.txt").write_text("\n".join(files) + "\n")
    print(f"  refreshed: translage, {len(models)} biolearn, {len(files)} methylCIPHER")


def load_catalogues() -> tuple[dict[str, str], set[str], set[str]]:
    """TranslAGE phenotype by normalised name, plus biolearn and methylCIPHER keys."""
    translage: dict[str, str] = {}
    p = DATA / "translage_clock_metadata.csv"
    if p.exists():
        with p.open(newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                pheno = (row.get("Trained Phenotype") or "").strip()
                for key in (row.get("Clock Name"), row.get("Friendly Name")):
                    if key:
                        for k in _permuted_keys(key):
                            translage.setdefault(k, pheno)

    def keys(path: Path, strip: tuple[str, ...] = ()) -> set[str]:
        if not path.exists():
            return set()
        out: set[str] = set()
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            # methylCIPHER files its clocks as calcHorvath1.R and
            # Horvath1_CpGs.R; both name the same clock and neither is the
            # clock's name, so the affixes come off before matching.
            for suffix in strip:
                if line.endswith(suffix):
                    line = line[: -len(suffix)]
                    break
            if line.lower().startswith("calc"):
                line = line[4:]
            out |= _permuted_keys(line)
        return out

    biolearn = keys(DATA / "biolearn_models.txt")
    cipher = keys(DATA / "methylcipher_files.txt",
                  strip=("_CpGs", "_CpG", "_data", "_parameters"))
    return translage, biolearn, cipher


#: Column, what it is, where it came from, and what an empty cell means.
#:
#: Written here rather than in a hand-kept spreadsheet so it cannot drift from
#: the file it describes: ``main`` asserts that this covers every emitted column
#: and nothing else, and fails the build rather than shipping a dictionary that
#: documents a column that is gone or omits one that is new.
#:
#: "blank_means" is a real field and not padding. Five of these columns use an
#: empty cell to carry information, and in every case the wrong reading of a
#: blank is the plausible one: that the value is zero, or that the check failed.
COLUMN_DOCS: dict[str, tuple[str, str, str]] = {
    # column: (description, source, blank_means)
    "clock_id": (
        "FALCONAge's identifier for the clock. The value to pass to "
        "fa.score(clocks=[...]) and the key every other FALCONAge output joins on.",
        "registry key", "never blank"),
    "name": (
        "The clock's name as the registry records it. Usually identical to "
        "clock_id; kept separate because third-party catalogues file clocks "
        "under a display name and the two are not always the same string.",
        "registry: name", "never blank"),
    "year": (
        "Publication year of the primary source.",
        "registry: year", "year not recorded"),
    "species": (
        "The species the clock was fitted on. Not the species it can be run on: "
        "the mammalian array carries 96% of Horvath2013's CpGs, so a zebra "
        "scores at high coverage against a clock fitted on people.",
        "registry: species", "never blank"),
    "data_type": (
        "The measurement the clock consumes: dna_methylation, "
        "clinical_chemistry, proteomic or transcriptomic.",
        "registry: data_type", "never blank"),
    "generation": (
        "The clock's generation in the field's own taxonomy. First generation "
        "is fitted to chronological age, second to a mortality or phenotype "
        "composite, third to a rate of change.",
        "registry: generation", "generation not classified"),
    "tissue": (
        "Tissues the clock was trained on, verbatim from the registry, "
        "semicolon-separated where there is more than one.",
        "registry: tissue", "tissue not recorded"),
    "tissue_class": (
        "tissue coarsened to one filterable class. multi_tissue wins over "
        "whole_blood when both are present: a clock fitted across tissues is "
        "not a blood clock, and filing it as one puts it on blood shortlists it "
        "does not belong on.",
        "derived from tissue", "never blank; absent tissue becomes 'unstated'"),
    "population": (
        "The training population as recorded, which is an AGE GROUP and not a "
        "health status. There is no health-status field in the registry.",
        "registry: population", "population not recorded"),
    "population_class": (
        "population coarsened to one filterable class.",
        "derived from population", "never blank; absent becomes 'unstated'"),
    "training_target": (
        "What the model was actually fitted to predict, verbatim. The single "
        "most important column for deciding what a score means, and the one "
        "most often assumed rather than read.",
        "registry: training_target", "target not recorded"),
    "target_class": (
        "training_target coarsened to one filterable class. 'disease' takes "
        "priority over every other rule, because it is the only class that says "
        "the training cohort was not healthy. 'clock_output' marks a model "
        "refitted to reproduce another clock rather than to predict a "
        "phenotype, which is what the PC clocks are.",
        "derived from training_target", "never blank; absent becomes 'unstated'"),
    "translage_trained_phenotype": (
        "The trained phenotype TranslAGE publishes for this clock. An "
        "independent check on training_target by a group that is not us; where "
        "the two disagree, read the paper.",
        "external: translage.io", "clock not listed in TranslAGE"),
    "predicts": (
        "What the clock's output is a prediction of, which is not always what "
        "it was trained on. A PC clock is trained on another clock's output and "
        "predicts age.",
        "registry: predicts", "not recorded"),
    "platform": (
        "Array or assay platforms the clock was trained on.",
        "registry: platform", "platform not recorded"),
    "n_features": (
        "Number of features (CpGs, or markers for a clinical clock) in the "
        "published model. Where this disagrees with the shipped coefficient "
        "count the registry records the discrepancy on the clock entry.",
        "registry: n_features", "feature count not published"),
    "unit": (
        "The unit of the score. Years, a proportion, a ratio, a log-hazard, or "
        "unitless.",
        "registry: unit", "no unit declared"),
    "scale_type": (
        "The formal scale of the output, which decides which downstream "
        "operations are defined. This is the field that stops age acceleration "
        "being computed on a pace of aging.",
        "registry: scale_type", "never blank"),
    "legal_operations": (
        "The downstream operations FALCONAge will perform on this clock's "
        "output. Anything absent here raises IllegalOperationError rather than "
        "returning a number.",
        "registry: derived from scale_type", "never blank"),
    "model_type": (
        "The fitted model form: elastic net, principal-component linear, "
        "reference-based deconvolution, neural network, and so on.",
        "registry: model_type", "model form not recorded"),
    "availability": (
        "Whether the coefficients are distributable. 'bundled' ships inside the "
        "wheel and runs offline; 'licensed' is research-use-only and must be "
        "supplied by the user; 'untraced' is catalogued metadata whose "
        "coefficients have not been traced to a primary source.",
        "registry: availability", "never blank"),
    "ships_coefficients": (
        "yes if this clock's weights are inside the package and it can score "
        "today. This is the column to filter on when building something that "
        "has to run.",
        "registry: derived", "never blank"),
    "profile_key": (
        "species|tissue_class|population_class|target_class. Two clocks sharing "
        "this string were fitted on the same kind of data. Sort on it to see "
        "the natural groupings.",
        "derived", "never blank"),
    "n_same_profile": (
        "How many clocks share this profile_key, counting this one.",
        "computed", "never blank"),
    "peers_same_profile": (
        "The other clocks sharing this profile_key, semicolon-separated. "
        "Symmetric by construction: if A lists B, B lists A.",
        "computed", "no other clock shares this training profile"),
    "n_same_tissue_class": (
        "How many clocks in the registry share this tissue_class.",
        "computed", "never blank"),
    "n_same_target_class": (
        "How many clocks in the registry share this target_class.",
        "computed", "never blank"),
    "feature_overlap_partner": (
        "The clock sharing the largest proportion of this clock's features, "
        "among those whose coefficients ship. Nearby clocks surface here: the "
        "two Lee placental variants find each other, as do stemTOC and "
        "stemTOCvitro.",
        "computed from shipped coefficients",
        "two different things, and feature_overlap_jaccard separates them: blank "
        "with a blank jaccard means this clock's features are not distributed; "
        "blank with a jaccard of 0 means they are, and it shares no feature with "
        "any other clock that also ships them"),
    "feature_overlap_jaccard": (
        "Jaccard index against feature_overlap_partner: shared features divided "
        "by the union. 1.0 would be an identical feature set.",
        "computed from shipped coefficients",
        "NOT zero. Blank means this clock's features are not distributed. A "
        "clock that ships coefficients and genuinely shares none reads 0."),
    "feature_overlap_shared": (
        "Count of features shared with feature_overlap_partner.",
        "computed from shipped coefficients",
        "features not distributed; see feature_overlap_note"),
    "feature_overlap_note": (
        "Why the three feature_overlap columns are empty, where they are.",
        "computed", "the overlap columns are populated and need no explanation"),
    "external_catalogues": (
        "Which independent catalogues also list this clock, of TranslAGE, "
        "biolearn and methylCIPHER. More than one is corroboration that the "
        "clock exists as described by someone other than us.",
        "external", "listed in none of the three catalogues checked"),
    "external_catalogue_urls": (
        "Where to check those catalogues, in the same order as "
        "external_catalogues.",
        "external", "no external catalogue lists this clock"),
    "training_cohort_note": (
        "Registry free text where it says something about the health of the "
        "training cohort. Populated for a handful of clocks only. THIS IS NOT A "
        "HEALTH-STATUS COLUMN and must not be filtered as one: health status is "
        "not a registry field, 5 of 175 entries mention it, and an empty cell "
        "here means nothing was written down, not that the cohort was healthy.",
        "registry: notes, filtered", "nothing recorded about cohort health"),
    "verify_url": (
        "Where to check this row. The clock's DOI where one exists, otherwise "
        "the coefficient source. Populated for all 175 clocks.",
        "registry: doi, else coefficient_source.url", "never blank"),
    "verify_url_kind": (
        "Whether verify_url is a DOI or a coefficient source URL.",
        "derived", "never blank"),
    "verify_url_status": (
        "Result of actually resolving verify_url. '200 ok' served directly. "
        "'403 publisher blocks bots; DOI registered in Crossref' means the link "
        "is good and the publisher refuses automated requests, which is the "
        "case for 44 rows, 32 of them the journal Aging. Anything ending "
        "'unresolved' is a genuinely broken reference.",
        "computed with --check-urls",
        "never blank; reads 'not checked' when the URL pass was skipped"),
    "coefficient_source_url": (
        "Where the coefficients themselves came from, where that is a URL "
        "distinct from the paper.",
        "registry: coefficient_source.url", "no separate coefficient source URL"),
    "citation": (
        "The primary reference, as the registry records it.",
        "registry: citation", "no citation recorded"),
}


def write_dictionary(rows: list[dict], order: list[str], path: Path) -> None:
    """Write the data dictionary beside the table, from the table itself.

    Vocabularies and examples are read off the emitted rows rather than typed
    out, so a class that stops being produced stops being documented as
    possible.
    """
    missing = [c for c in order if c not in COLUMN_DOCS]
    extra = [c for c in COLUMN_DOCS if c not in order]
    if missing or extra:
        raise SystemExit(
            "COLUMN_DOCS is out of step with the table.\n"
            f"  columns with no description: {missing}\n"
            f"  descriptions for absent columns: {extra}")

    out = []
    for i, col in enumerate(order, 1):
        values = [str(r.get(col, "")) for r in rows]
        filled = [v for v in values if v != ""]
        distinct = sorted(set(filled))
        desc, source, blank = COLUMN_DOCS[col]
        out.append({
            "position": i,
            "column": col,
            "description": desc,
            "source": source,
            "value_type": ("integer" if all(v.isdigit() for v in filled) and filled
                           else "number" if _all_numeric(filled)
                           else "url" if filled and all(v.startswith("http") for v in filled)
                           else "list (semicolon-separated)"
                           if any("; " in v for v in filled) else "text"),
            "n_distinct": len(distinct),
            "n_blank": len(values) - len(filled),
            "possible_values": ("; ".join(distinct) if 0 < len(distinct) <= 15
                                else f"{len(distinct)} distinct values"),
            "example": filled[0][:160] if filled else "",
            "blank_means": blank,
        })

    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "position", "column", "description", "source", "value_type",
            "n_distinct", "n_blank", "possible_values", "example", "blank_means"])
        w.writeheader()
        w.writerows(out)


def _all_numeric(values: list[str]) -> bool:
    if not values:
        return False
    try:
        for v in values:
            float(v)
    except ValueError:
        return False
    return True


def _crossref_registered(doi: str, timeout: float = 20.0) -> bool:
    """Is this DOI registered, independently of whether the publisher serves it?"""
    try:
        req = urllib.request.Request(
            f"https://api.crossref.org/works/{doi}",
            headers={"User-Agent": "falconage-overlap/1.0 (mailto:bhunakun@uni-bonn.de)"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def check_url(url: str, timeout: float = 20.0) -> str:
    """Resolution status for one URL, as a short phrase, never raising.

    HEAD first because a DOI resolver answers it without transferring the
    article, then GET, because a number of publishers answer HEAD with 403 and
    GET with 200 and the first answer is not about the link being broken.

    A bare ``403`` in this column would be read as a dead link, and for this
    corpus it usually is not: 32 of the 44 non-200 rows are the journal *Aging*
    (prefix 10.18632), which refuses automated requests and serves the article
    to a browser. So when a DOI does not resolve over HTTP, the DOI itself is
    checked against Crossref, and the distinction that matters -- registered but
    bot-blocked, against genuinely absent -- is recorded rather than collapsed.
    """
    last = "unreachable"
    for method in ("HEAD", "GET"):
        try:
            req = urllib.request.Request(
                url, method=method,
                headers={"User-Agent": "Mozilla/5.0 (compatible; falconage-overlap/1.0)"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                if r.status == 200:
                    return "200 ok"
                last = str(r.status)
        except Exception as exc:  # noqa: BLE001 - any failure is a status here
            code = getattr(exc, "code", None)
            last = str(code) if code else type(exc).__name__

    m = re.search(r"doi\.org/(10\.[^/]+/\S+)", url)
    if m and _crossref_registered(m.group(1)):
        return f"{last} publisher blocks bots; DOI registered in Crossref"
    return f"{last} unresolved"


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--dictionary", default=None,
                    help="where to write the column dictionary "
                         "(default: <out>_col_desc.csv beside it)")
    ap.add_argument("--check-urls", action="store_true",
                    help="resolve every verify_url and record the HTTP status")
    ap.add_argument("--refresh", action="store_true",
                    help="re-fetch the external catalogue snapshots first")
    args = ap.parse_args(argv)

    if args.refresh:
        refresh_catalogues()

    import falconage as fa

    reg = fa.registry.load()
    translage, biolearn, cipher = load_catalogues()
    clocks = [reg.get(cid) for cid in reg.list()]

    # -- per-clock attributes ------------------------------------------------
    rows = []
    features: dict[str, set[str]] = {}
    for c in clocks:
        tis = list(c.tissue or [])
        tgt = list(c.training_target or [])
        tc, pc, gc = tissue_class(tis), population_class(c.population), target_class(tgt)

        doi = (c.doi or "").strip()
        src = (getattr(c.coefficient_source, "url", None) or "").strip()
        if doi:
            verify = doi if doi.startswith("http") else f"https://doi.org/{doi}"
            verify_kind = "doi"
        else:
            verify, verify_kind = src, "coefficient_source"

        # Every name this clock might be filed under elsewhere: our id, our
        # name, the aliases above, and each of those in any word order.
        lookups: set[str] = set()
        for label in (c.id, c.name, *CATALOGUE_ALIASES.get(c.id, ())):
            lookups |= _permuted_keys(label)

        cats = []
        if lookups & set(translage):
            cats.append("translage")
        if lookups & biolearn:
            cats.append("biolearn")
        if lookups & cipher:
            cats.append("methylcipher")
        phenotype = next((translage[k] for k in sorted(lookups)
                          if translage.get(k)), "")

        note = (c.notes or "").strip()
        health = ""
        if re.search(r"health|disease|patient|case-control|cancer", note, re.I):
            health = note[:300]

        if reg.has_coefficient_vector(c.id):
            try:
                feats, _ = reg.coefficients(c.id)
                features[c.id] = {str(f) for f in feats
                                  if str(f).lower() not in
                                  ("intercept", "(intercept)", "_intercept")}
            except Exception:  # pragma: no cover - defensive
                pass

        rows.append({
            "clock_id": c.id,
            "name": c.name,
            "year": c.year,
            "species": c.species,
            "data_type": c.data_type,
            "generation": c.generation,
            "tissue": "; ".join(tis),
            "tissue_class": tc,
            "population": c.population or "",
            "population_class": pc,
            "training_target": "; ".join(tgt),
            "target_class": gc,
            "predicts": "; ".join(c.predicts or []),
            "platform": "; ".join(c.platform or []),
            "n_features": c.n_features if c.n_features is not None else "",
            "unit": "; ".join(c.unit or []),
            "scale_type": c.scale_type,
            "legal_operations": "; ".join(sorted(c.legal_operations)),
            "model_type": c.model_type or "",
            "availability": c.availability,
            "ships_coefficients": "yes" if c.ships_coefficients else "no",
            "profile_key": f"{c.species}|{tc}|{pc}|{gc}",
            "translage_trained_phenotype": phenotype,
            "external_catalogues": "; ".join(cats),
            "external_catalogue_urls": "; ".join(CATALOGUE_URLS[c_] for c_ in cats),
            "training_cohort_note": health,
            "verify_url": verify,
            "verify_url_kind": verify_kind,
            "coefficient_source_url": src,
            "citation": (c.citation or "").strip(),
        })

    # -- overlap counts ------------------------------------------------------
    def tally(field: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in rows:
            out[r[field]] = out.get(r[field], 0) + 1
        return out

    by_tissue, by_target, by_profile = tally("tissue_class"), tally("target_class"), tally("profile_key")
    peers: dict[str, list[str]] = {}
    for r in rows:
        peers.setdefault(r["profile_key"], []).append(r["clock_id"])

    for r in rows:
        same = [x for x in peers[r["profile_key"]] if x != r["clock_id"]]
        r["n_same_tissue_class"] = by_tissue[r["tissue_class"]]
        r["n_same_target_class"] = by_target[r["target_class"]]
        r["n_same_profile"] = by_profile[r["profile_key"]]
        r["peers_same_profile"] = "; ".join(sorted(same))

    # -- feature overlap, for the clocks whose features we can actually see ---
    for r in rows:
        cid = r["clock_id"]
        mine = features.get(cid)
        if not mine:
            r["feature_overlap_partner"] = ""
            r["feature_overlap_jaccard"] = ""
            r["feature_overlap_shared"] = ""
            r["feature_overlap_note"] = "features not distributed for this clock"
            continue
        best, best_j, best_n = "", 0.0, 0
        for other, theirs in features.items():
            if other == cid:
                continue
            shared = len(mine & theirs)
            if not shared:
                continue
            j = shared / len(mine | theirs)
            if j > best_j:
                best, best_j, best_n = other, j, shared
        r["feature_overlap_partner"] = best
        r["feature_overlap_jaccard"] = f"{best_j:.4f}" if best else "0"
        r["feature_overlap_shared"] = best_n if best else 0
        r["feature_overlap_note"] = ""

    # -- URL check -----------------------------------------------------------
    if args.check_urls:
        seen: dict[str, str] = {}
        for i, r in enumerate(rows, 1):
            u = r["verify_url"]
            if u not in seen:
                seen[u] = check_url(u)
                print(f"  [{i}/{len(rows)}] {seen[u]:>6}  {u}", flush=True)
            r["verify_url_status"] = seen[u]
    else:
        for r in rows:
            r["verify_url_status"] = "not checked"

    order = [
        "clock_id", "name", "year", "species", "data_type", "generation",
        "tissue", "tissue_class", "population", "population_class",
        "training_target", "target_class", "translage_trained_phenotype",
        "predicts", "platform", "n_features", "unit", "scale_type",
        "legal_operations", "model_type", "availability", "ships_coefficients",
        "profile_key", "n_same_profile", "peers_same_profile",
        "n_same_tissue_class", "n_same_target_class",
        "feature_overlap_partner", "feature_overlap_jaccard",
        "feature_overlap_shared", "feature_overlap_note",
        "external_catalogues", "external_catalogue_urls",
        "training_cohort_note", "verify_url", "verify_url_kind",
        "verify_url_status", "coefficient_source_url", "citation",
    ]
    rows.sort(key=lambda r: (r["tissue_class"], r["target_class"], r["clock_id"]))

    out = Path(args.out)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=order, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    doc = Path(args.dictionary) if args.dictionary else out.with_name(
        out.stem + "_col_desc.csv")
    write_dictionary(rows, order, doc)

    print(f"wrote {out}: {len(rows)} clock(s), {len(order)} column(s)")
    print(f"wrote {doc}: one row per column, with its vocabulary and what a blank means")
    print(f"  feature overlap computable for {len(features)} clock(s) that ship coefficients")
    print(f"  cross-referenced against {sum(1 for r in rows if r['external_catalogues'])} "
          "clock(s) in at least one external catalogue")
    return 0


if __name__ == "__main__":
    sys.exit(main())
