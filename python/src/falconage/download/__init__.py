"""Fetch public data from an accession.

``fa.download("GSE182991")`` and the files appear, cached, checksummed and with
a normalised sample table beside them. The point is not to save typing -- it is
that the alternative, everybody writing their own GEO scraper, is why two
analyses of the same accession routinely start from different files.

Sources are dispatched on the shape of the accession. Credentialed archives
(dbGaP, EGA, Synapse, UK Biobank) are documented and deliberately not automated:
the access agreement is between the user and the archive, and a tool that made
it one function call would be inviting people to breach it.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .._version import __version__
from ..core.config import default_cache_dir
from ..core.errors import ChecksumMismatchError, DownloadError
from ..core.logging import get_logger

__all__ = ["DownloadResult", "cache_info", "clear_cache", "download", "resolve_source"]

log = get_logger("download")
UA = f"FALCONAge/{__version__} (+https://github.com/bhagesh-h/FALCONAge)"
CHUNK = 1 << 20

GEO_SERIES = re.compile(r"^GSE\d+$", re.I)
GEO_SAMPLE = re.compile(r"^GSM\d+$", re.I)
ARRAYEXPRESS = re.compile(r"^E-[A-Z]{4}-\d+$", re.I)
SRA = re.compile(r"^(SR[APRSX]\d+|PRJ[A-Z]{2}\d+|ERR\d+|DRR\d+)$", re.I)
DOI = re.compile(r"^10\.\d{4,9}/\S+$")
HF = re.compile(r"^[\w.-]+/[\w.-]+$")
PRIDE = re.compile(r"^PXD\d+$", re.I)
METABOLIGHTS = re.compile(r"^MTBLS\d+$", re.I)
GDC_UUID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
GDC_PROJECT = re.compile(
    r"^(TCGA|TARGET|CPTAC|MMRF|HCMI|CGCI|GENIE|NCICCR|REBC|VAREPOP|WCDT|"
    r"ORGANOID|MP2PRT|CDDP|EXCEPTIONAL)[-_][A-Z0-9]+$", re.I)
# Figshare mints DOIs under its own prefix and Zenodo under 10.5281, so the two
# are distinguishable without asking either API which one owns the record.
FIGSHARE_DOI = re.compile(r"^10\.6084/m9\.figshare\.\S+$", re.I)


@dataclass
class DownloadResult:
    accession: str
    source: str
    files: list[Path] = field(default_factory=list)
    samples: pd.DataFrame | None = None
    cache_dir: Path | None = None
    notes: list[str] = field(default_factory=list)
    #: Populated by sra(): one row per sequencing run, with FASTQ URLs.
    #: Separate from `samples` because these are reads awaiting alignment and
    #: methylation calling, not a sample table a clock can be pointed at.
    run_table: pd.DataFrame | None = None

    def __repr__(self) -> str:  # pragma: no cover - display only
        n = len(self.files)
        total = sum(f.stat().st_size for f in self.files if f.exists())
        return (f"DownloadResult({self.accession}, {self.source}, {n} file(s), "
                f"{total / 1e6:.1f} MB)")

    def read(self, **kw):
        """Read the first file that a FALCONAge reader recognises."""
        from ..io import read

        for f in self.files:
            try:
                return read(f, **kw)
            except Exception:
                continue
        raise DownloadError(
            f"none of the {len(self.files)} downloaded file(s) could be read as a "
            "matrix. Call the specific reader on the one you want; "
            f"{', '.join(f.name for f in self.files[:4])}")


def resolve_source(accession: str) -> str:
    a = accession.strip()
    if GEO_SERIES.match(a):
        return "geo_series"
    if GEO_SAMPLE.match(a):
        return "geo_sample"
    if ARRAYEXPRESS.match(a):
        return "arrayexpress"
    if SRA.match(a):
        return "sra"
    if PRIDE.match(a):
        return "pride"
    if METABOLIGHTS.match(a):
        return "metabolights"
    # Both GDC forms before the Hugging Face rule: TCGA-BRCA does not look like
    # owner/name, but a project id with an underscore would, and being wrong
    # here sends someone to a 404 on huggingface.co.
    if GDC_UUID.match(a) or GDC_PROJECT.match(a):
        return "gdc"
    if a.lower().startswith(("http://", "https://")):
        return "url"
    if FIGSHARE_DOI.match(a):
        return "figshare"
    if DOI.match(a):
        return "zenodo"
    if HF.match(a):
        return "huggingface"
    raise DownloadError(
        f"cannot tell what {accession!r} is.\n"
        "  Recognised: GSE/GSM (GEO), E-MTAB-* (ArrayExpress), SRP/SRR/PRJNA (SRA),\n"
        "  PXD* (PRIDE), MTBLS* (MetaboLights), a GDC file UUID or project id,\n"
        "  a DOI (Figshare or Zenodo), owner/name (Hugging Face), or a full https URL.\n"
        "  dbGaP, EGA, Synapse and UK Biobank need credentials and are documented\n"
        "  rather than automated -- the access agreement is yours, not the tool's."
    )


# ---------------------------------------------------------------------------
# transfer
# ---------------------------------------------------------------------------
def _cache_path(url: str, root: Path) -> Path:
    """Content-addressed by URL, so two accessions sharing a file share the copy."""
    h = hashlib.sha256(url.encode()).hexdigest()[:16]
    name = url.rstrip("/").split("/")[-1] or "download"
    return root / "downloads" / h[:2] / f"{h}_{name}"


def fetch(url: str, *, cache_dir: Path | None = None, sha256: str | None = None,
          retries: int = 4, force: bool = False, quiet: bool = False) -> Path:
    """Download one URL into the cache, resuming and verifying.

    Resumable because the files are large and the archives are not always fast:
    GEO's FTP frontend drops connections often enough that a non-resuming
    fetcher makes a 500 MB series a matter of luck.
    """
    root = Path(cache_dir or default_cache_dir())
    dest = _cache_path(url, root)
    if dest.exists() and not force:
        if sha256 and _sha256(dest) != sha256:
            log.warning("cached copy of %s fails its digest; refetching", dest.name)
            dest.unlink()
        else:
            return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    for attempt in range(1, retries + 1):
        offset = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        if offset:
            req.add_header("Range", f"bytes={offset}-")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                if offset and resp.status != 206:
                    offset = 0
                    part.unlink(missing_ok=True)
                with part.open("ab" if offset else "wb") as fh:
                    while True:
                        block = resp.read(CHUNK)
                        if not block:
                            break
                        fh.write(block)
            break
        except urllib.error.HTTPError as exc:
            if exc.code in (400, 401, 403, 404, 410):
                raise DownloadError(f"{url}: HTTP {exc.code} {exc.reason}") from exc
            if attempt == retries:
                raise DownloadError(f"{url}: {exc}") from exc
            time.sleep(2 ** attempt)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt == retries:
                raise DownloadError(f"{url}: {exc}") from exc
            time.sleep(2 ** attempt)

    if sha256:
        got = _sha256(part)
        if got != sha256:
            part.unlink(missing_ok=True)
            raise ChecksumMismatchError(
                f"{url}\n  expected {sha256}\n  got      {got}")
    part.replace(dest)
    if not quiet:
        log.info("fetched %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
    return dest


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def _get_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# GEO
# ---------------------------------------------------------------------------
def _geo_dir(acc: str, kind: str) -> str:
    stub = acc[:-3] + "nnn"
    return f"https://ftp.ncbi.nlm.nih.gov/geo/{kind}/{stub}/{acc}/"


def geo_series(acc: str, *, want: str = "matrix", cache_dir: Path | None = None,
               dry_run: bool = False) -> DownloadResult:
    """Fetch a GEO series.

    ``want="matrix"`` takes the series matrix, which carries the metadata and
    (usually) the values in one file -- the right default, because it is the
    only path that works for the majority of series that publish no IDATs.
    ``want="suppl"`` lists the supplementary files instead. ``want="both"``
    takes everything, which for a large series is gigabytes and is why
    ``dry_run=True`` exists.
    """
    acc = acc.upper()
    res = DownloadResult(accession=acc, source="geo",
                         cache_dir=Path(cache_dir or default_cache_dir()))
    urls: list[str] = []

    if want in ("matrix", "both"):
        urls.append(_geo_dir(acc, "series") + f"matrix/{acc}_series_matrix.txt.gz")
    if want in ("suppl", "both"):
        listing = _geo_dir(acc, "series") + "suppl/filelist.txt"
        try:
            p = fetch(listing, cache_dir=cache_dir, quiet=True)
            for line in p.read_text(encoding="utf-8", errors="replace").splitlines()[1:]:
                cols = line.split("\t")
                if len(cols) >= 2 and cols[0].strip() == "File":
                    urls.append(_geo_dir(acc, "series") + "suppl/" + cols[1].strip())
        except DownloadError as exc:
            res.notes.append(f"no supplementary listing: {exc}")

    if dry_run:
        res.notes.append(f"{len(urls)} file(s) would be fetched: "
                         + ", ".join(u.split("/")[-1] for u in urls))
        return res

    for u in urls:
        try:
            res.files.append(fetch(u, cache_dir=cache_dir))
        except DownloadError as exc:
            res.notes.append(str(exc))

    if not res.files:
        raise DownloadError(
            f"{acc}: nothing could be fetched.\n"
            "  Some series publish no series matrix; try want='suppl'.\n"
            f"  Browse: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={acc}")

    for f in res.files:
        if "series_matrix" in f.name:
            res.samples = geo_sample_table(f)
            break
    return res


def geo_sample(acc: str, *, cache_dir: Path | None = None,
               pattern: str | None = None) -> DownloadResult:
    """Fetch one GSM's supplementary files -- typically an IDAT pair."""
    acc = acc.upper()
    base = _geo_dir(acc, "samples")
    res = DownloadResult(accession=acc, source="geo",
                         cache_dir=Path(cache_dir or default_cache_dir()))
    try:
        html = urllib.request.urlopen(
            urllib.request.Request(base + "suppl/", headers={"User-Agent": UA}),
            timeout=60).read().decode("utf-8", "replace")
    except Exception as exc:
        raise DownloadError(f"{acc}: cannot list {base}suppl/ ({exc})") from exc

    names = sorted(set(re.findall(r'href="([^"/?][^"]*)"', html)))
    names = [n for n in names if not n.startswith(("http", "/"))]
    if pattern:
        names = [n for n in names if re.search(pattern, n)]
    for n in names:
        res.files.append(fetch(base + "suppl/" + n, cache_dir=cache_dir))
    if not res.files:
        raise DownloadError(f"{acc}: no supplementary files matched")
    return res


def geo_sample_table(series_matrix: Path) -> pd.DataFrame:
    """Normalise a series matrix's characteristics block into a sample table."""
    from ..io.methylation import read_series_matrix

    try:
        return read_series_matrix(series_matrix).obs
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# other sources
# ---------------------------------------------------------------------------
def huggingface(repo: str, *, filenames: Iterable[str] = (), revision: str = "main",
                repo_type: str = "datasets", cache_dir: Path | None = None
                ) -> DownloadResult:
    """Fetch named files from a Hugging Face repo at a pinned revision.

    ``revision`` defaults to ``main`` but a commit SHA is the right choice for
    anything a result will cite: a moving reference lets the data change under
    the number.
    """
    res = DownloadResult(accession=repo, source="huggingface")
    base = f"https://huggingface.co/{repo_type}/{repo}/resolve/{revision}/"
    for fn in filenames:
        res.files.append(fetch(base + fn, cache_dir=cache_dir))
    if not res.files:
        res.notes.append(
            "no filenames given; list them with "
            f"https://huggingface.co/api/{repo_type}/{repo}/tree/{revision}?recursive=true")
    return res


def zenodo(doi: str, *, cache_dir: Path | None = None) -> DownloadResult:
    """Fetch every file in a Zenodo record, resolved from its DOI."""
    rec = doi.rstrip("/").split(".")[-1]
    meta = _get_json(f"https://zenodo.org/api/records/{rec}")
    res = DownloadResult(accession=doi, source="zenodo")
    for f in meta.get("files", []):
        url = f.get("links", {}).get("self")
        digest = (f.get("checksum") or "").replace("md5:", "")
        if url:
            res.files.append(fetch(url, cache_dir=cache_dir))
            res.notes.append(f"{f.get('key')}: zenodo md5 {digest}")
    return res


def arrayexpress(acc: str, *, cache_dir: Path | None = None) -> DownloadResult:
    """Fetch an ArrayExpress/BioStudies accession's files."""
    meta = _get_json(f"https://www.ebi.ac.uk/biostudies/api/v1/studies/{acc}")
    res = DownloadResult(accession=acc, source="arrayexpress")
    base = f"https://www.ebi.ac.uk/biostudies/files/{acc}/"
    for f in meta.get("files", []) or []:
        path = f.get("path")
        if path:
            res.files.append(fetch(base + path, cache_dir=cache_dir))
    return res


def figshare(doi: str, *, cache_dir: Path | None = None) -> DownloadResult:
    """Fetch a Figshare article's files, resolved from its DOI.

    Split from :func:`zenodo` rather than folded into it because the two APIs
    disagree about everything except being REST: Figshare keys the article by
    the numeric tail of the DOI, calls the list ``/files`` on a separate
    endpoint, and names the download link ``download_url``. A single "DOI"
    handler that guessed between them would fail on whichever it guessed wrong,
    with a KeyError.
    """
    art = doi.rstrip("/").split(".")[-1].lstrip("v")
    res = DownloadResult(accession=doi, source="figshare")
    meta = _get_json(f"https://api.figshare.com/v2/articles/{art}/files")
    for f in meta if isinstance(meta, list) else []:
        url = f.get("download_url")
        if url:
            res.files.append(fetch(url, cache_dir=cache_dir))
            res.notes.append(f"{f.get('name')}: figshare md5 {f.get('computed_md5')}")
    if not res.files:
        raise DownloadError(
            f"{doi}: Figshare article {art} lists no downloadable files.\n"
            "  A DOI that is a Zenodo record rather than a Figshare one goes "
            "to zenodo(); pass it explicitly if the shape fooled the router.")
    return res


def pride(acc: str, *, cache_dir: Path | None = None,
          extensions: Iterable[str] = ()) -> DownloadResult:
    """Fetch a PRIDE proteomics project's files.

    ``extensions`` filters by suffix, and filtering is close to mandatory: a
    PRIDE project routinely carries tens of gigabytes of RAW instrument files
    next to the few megabytes of processed intensities a clock can use. Called
    without it this lists what is there and refuses, rather than starting a
    transfer nobody sized.
    """
    meta = _get_json(
        "https://www.ebi.ac.uk/pride/ws/archive/v2/files/byProject"
        f"?accession={acc}")
    entries = meta if isinstance(meta, list) else meta.get("_embedded", {}).get("files", [])
    res = DownloadResult(accession=acc, source="pride")

    wanted = tuple(e.lower() for e in extensions)
    picked = []
    for f in entries:
        name = f.get("fileName") or ""
        urls = [loc.get("value") for loc in (f.get("publicFileLocations") or [])
                if str(loc.get("value", "")).startswith("http")]
        if not urls:
            continue
        if wanted and not name.lower().endswith(wanted):
            continue
        picked.append((name, urls[0], f.get("fileSizeBytes") or 0))

    if not wanted:
        total = sum(s for _, _, s in picked)
        raise DownloadError(
            f"{acc}: {len(picked)} files, {total / 1e9:.1f} GB in total.\n"
            "  Name the extensions you want -- most of this is raw instrument "
            "output that no aging clock reads:\n"
            f"    fa.download({acc!r}, extensions=['.txt', '.tsv'])\n"
            "  Present: "
            + ", ".join(sorted({("." + n.rsplit(".", 1)[-1]) for n, _, _ in picked
                                if "." in n})[:12]))

    for name, url, _ in picked:
        res.files.append(fetch(url, cache_dir=cache_dir))
        res.notes.append(name)
    return res


def metabolights(acc: str, *, cache_dir: Path | None = None) -> DownloadResult:
    """Fetch a MetaboLights study's metadata and processed tables.

    Only the ISA-Tab files (``i_``, ``s_``, ``a_``, ``m_``): those carry the
    sample table and the quantified metabolites. The raw spectra beside them
    are large and are not what a metabolomic clock consumes.
    """
    meta = _get_json(
        f"https://www.ebi.ac.uk/metabolights/ws/studies/{acc}/files")
    res = DownloadResult(accession=acc, source="metabolights")
    base = f"https://www.ebi.ac.uk/metabolights/ws/studies/{acc}/download/"
    for f in meta.get("study", []) or []:
        name = f.get("file") or ""
        if name.startswith(("i_", "s_", "a_", "m_")):
            res.files.append(fetch(base + name, cache_dir=cache_dir))
            res.notes.append(name)
    if not res.files:
        raise DownloadError(
            f"{acc}: no ISA-Tab files found. The study may be private, or the "
            "accession may not exist.")
    return res


def gdc(uuid_or_project: str, *, cache_dir: Path | None = None,
        data_type: str = "Methylation Beta Value") -> DownloadResult:
    """Fetch from the NCI Genomic Data Commons: one file UUID, or a project.

    A project id (``TCGA-BRCA``) resolves to every open-access file of
    ``data_type`` in it, which for methylation is one beta matrix per aliquot
    and can be several hundred files. Controlled-access files are not reachable
    without a token and are excluded by the filter rather than failing one at a
    time.
    """
    res = DownloadResult(accession=uuid_or_project, source="gdc")

    if re.match(r"^[0-9a-f]{8}-[0-9a-f]{4}-", uuid_or_project, re.I):
        ids = [uuid_or_project]
    else:
        filt = {"op": "and", "content": [
            {"op": "in", "content": {"field": "cases.project.project_id",
                                     "value": [uuid_or_project]}},
            {"op": "in", "content": {"field": "data_type", "value": [data_type]}},
            {"op": "in", "content": {"field": "access", "value": ["open"]}},
        ]}
        q = ("https://api.gdc.cancer.gov/files?filters="
             + urllib.parse.quote(json.dumps(filt))
             + "&fields=file_id,file_name,file_size&size=2000&format=JSON")
        meta = _get_json(q)
        hits = meta.get("data", {}).get("hits", [])
        if not hits:
            raise DownloadError(
                f"{uuid_or_project}: no open-access {data_type!r} files.\n"
                "  Check the project id, or name a different data_type "
                "(for example 'Masked Somatic Mutation').")
        ids = [h["file_id"] for h in hits]
        res.notes.append(
            f"{len(ids)} open files, "
            f"{sum(h.get('file_size', 0) for h in hits) / 1e9:.2f} GB")

    for fid in ids:
        res.files.append(
            fetch(f"https://api.gdc.cancer.gov/data/{fid}", cache_dir=cache_dir))
    return res


def sra(acc: str, *, cache_dir: Path | None = None) -> DownloadResult:
    """Resolve an SRA/ENA accession to its run table and file URLs.

    Deliberately stops at the file list. SRA holds reads, and reads are not
    something a clock can score: they need alignment and methylation calling
    first, which is a pipeline this package does not contain and should not
    pretend to. What it can usefully do is resolve the accession, so the run
    table and the FASTQ URLs are in hand for whichever aligner does that work.

    Uses ENA's portal API rather than NCBI's: it returns the FASTQ URLs
    directly, where NCBI returns SRA-format blobs that need the toolkit.
    """
    url = ("https://www.ebi.ac.uk/ena/portal/api/filereport"
           f"?accession={acc}&result=read_run&format=tsv"
           "&fields=run_accession,sample_accession,fastq_ftp,fastq_bytes,"
           "library_strategy,scientific_name")
    with urllib.request.urlopen(url, timeout=60) as fh:  # noqa: S310 - fixed EBI host
        text = fh.read().decode("utf-8", "replace")

    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        raise DownloadError(f"{acc}: ENA returned no runs. Is the accession right?")

    header = lines[0].split("\t")
    runs = [dict(zip(header, ln.split("\t"))) for ln in lines[1:]]

    res = DownloadResult(accession=acc, source="sra")
    res.notes.append(f"{len(runs)} run(s); FASTQ URLs in .run_table")
    res.notes.append(
        "Reads, not methylation calls. Align and call first -- Bismark or "
        "bwa-meth, then a beta matrix -- and bring that here.")
    res.run_table = pd.DataFrame(runs)
    return res


CREDENTIALED = {
    "dbgap": "https://www.ncbi.nlm.nih.gov/gap/ -- needs an approved dbGaP request and the SRA toolkit with a repository key",
    "ega": "https://ega-archive.org/ -- needs a DAC-approved account and pyega3",
    "synapse": "https://www.synapse.org/ -- needs an account and synapseclient; some datasets add per-study conditions",
    "ukbiobank": "https://www.ukbiobank.ac.uk/ -- needs an approved application; data is delivered, not downloaded",
}


def download(accession: str, *, cache_dir: Path | None = None, dry_run: bool = False,
             **kw) -> DownloadResult:
    """Fetch by accession, dispatching on its shape.

    >>> fa.download("GSE182991")
    >>> fa.download("GSE182991", want="suppl", dry_run=True)
    """
    a = accession.strip()
    low = a.lower()
    if low in CREDENTIALED:
        raise DownloadError(
            f"{a} is a credentialed archive and FALCONAge does not automate it.\n"
            f"  {CREDENTIALED[low]}\n"
            "  Once the files are local, read them directly: "
            "falconage.io.read(path).")

    src = resolve_source(a)
    if src == "geo_series":
        return geo_series(a, cache_dir=cache_dir, dry_run=dry_run, **kw)
    if src == "geo_sample":
        return geo_sample(a, cache_dir=cache_dir, **kw)
    if src == "huggingface":
        return huggingface(a, cache_dir=cache_dir, **kw)
    if src == "zenodo":
        return zenodo(a, cache_dir=cache_dir)
    if src == "arrayexpress":
        return arrayexpress(a, cache_dir=cache_dir)
    if src == "figshare":
        return figshare(a, cache_dir=cache_dir)
    if src == "pride":
        return pride(a, cache_dir=cache_dir, **kw)
    if src == "metabolights":
        return metabolights(a, cache_dir=cache_dir)
    if src == "gdc":
        return gdc(a, cache_dir=cache_dir, **kw)
    if src == "sra":
        return sra(a, cache_dir=cache_dir)
    if src == "url":
        r = DownloadResult(accession=a, source="url")
        r.files.append(fetch(a, cache_dir=cache_dir, **kw))
        return r
    raise DownloadError(f"{src} is recognised but has no handler")


def cache_info(cache_dir: Path | None = None) -> pd.DataFrame:
    root = Path(cache_dir or default_cache_dir()) / "downloads"
    if not root.exists():
        return pd.DataFrame(columns=["file", "bytes", "modified"])
    rows = [{"file": str(p.relative_to(root)), "bytes": p.stat().st_size,
             "modified": pd.Timestamp(p.stat().st_mtime, unit="s")}
            for p in root.rglob("*") if p.is_file()]
    return pd.DataFrame(rows).sort_values("bytes", ascending=False)


def clear_cache(cache_dir: Path | None = None, *, confirm: bool = False) -> int:
    """Delete the download cache. Returns the bytes freed."""
    root = Path(cache_dir or default_cache_dir()) / "downloads"
    if not root.exists():
        return 0
    total = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    if not confirm:
        raise DownloadError(
            f"clear_cache would delete {total / 1e6:.1f} MB from {root}. "
            "Pass confirm=True.")
    import shutil

    shutil.rmtree(root)
    return total
