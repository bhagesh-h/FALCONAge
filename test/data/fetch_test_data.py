#!/usr/bin/env python3
"""Fetch the FALCONAge public test corpus described in ``datasets.yaml``.

Nothing here imports FALCONAge. The corpus is what the package will be tested
against, so a fetcher that needed the package installed would be circular, and
one that broke whenever the package broke would be useless at exactly the moment
it is needed. The standard library plus PyYAML is the whole dependency set.

``fetch_test_data.R`` is the same program in R and must produce a byte-identical
``checksums.sha256``. That equality is the corpus-level form of the R/Python
conformance gate in the design notes, and Dockerfile.testdata
asserts it at build time.

Usage
-----
    python fetch_test_data.py --dry-run              # what it would cost
    python fetch_test_data.py                        # fetch the default groups
    python fetch_test_data.py --groups bench,mouse   # fetch two of them
    python fetch_test_data.py --verify               # re-check what is on disk
    python fetch_test_data.py --self-test            # manifest sanity, no network
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

try:
    import yaml
except ImportError:  # pragma: no cover - the message is the whole point
    sys.exit("PyYAML is required: pip install PyYAML")

__version__ = "1.0.0"

USER_AGENT = f"FALCONAge-testdata/{__version__} (+https://github.com/bhagesh-h/FALCONAge)"
CHUNK = 1 << 20  # 1 MiB. Large enough that the syscall overhead disappears,
                 # small enough that a progress line still moves.


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Entry:
    """One file: where it goes, where it comes from, and what it should be."""

    path: str
    url: str
    bytes: int
    source: str
    group: str
    sha256: str | None = None
    note: str = ""


@dataclass(frozen=True)
class Manifest:
    schema_version: int
    budget_bytes: int
    expected_total_bytes: int
    expected_total_files: int
    sources: dict[str, dict[str, Any]]
    groups: list[dict[str, Any]]
    entries: list[Entry]
    digest: str

    def group_ids(self) -> list[str]:
        return [g["id"] for g in self.groups]

    def default_group_ids(self) -> list[str]:
        return [g["id"] for g in self.groups if g.get("default", False)]


def load_manifest(path: Path) -> Manifest:
    raw_bytes = path.read_bytes()
    doc = yaml.safe_load(raw_bytes.decode("utf-8"))

    if doc.get("schema_version") != 1:
        raise SystemExit(
            f"{path}: schema_version {doc.get('schema_version')!r} is not 1; "
            "this fetcher only understands schema 1"
        )

    entries: list[Entry] = []
    for group in doc["groups"]:
        for f in group["files"]:
            entries.append(
                Entry(
                    path=f["path"],
                    url=f["url"],
                    bytes=int(f["bytes"]),
                    # `sha256` is omitted, never null, when the publisher gives
                    # us none -- so that R's yaml reader and this one agree on
                    # what absence looks like.
                    sha256=f.get("sha256"),
                    source=f["source"],
                    group=group["id"],
                    note=f.get("note", ""),
                )
            )

    return Manifest(
        schema_version=int(doc["schema_version"]),
        budget_bytes=int(doc["budget_bytes"]),
        expected_total_bytes=int(doc["expected_total_bytes"]),
        expected_total_files=int(doc["expected_total_files"]),
        sources=doc["sources"],
        groups=doc["groups"],
        entries=entries,
        # The digest is of the file, not of the parse. Two implementations that
        # parse differently still agree here, which is what makes it useful as a
        # provenance field rather than a checksum of an opinion.
        digest=hashlib.sha256(raw_bytes).hexdigest(),
    )


def select(manifest: Manifest, spec: str) -> list[Entry]:
    """Resolve a ``--groups`` specification to the entries it names."""
    if spec == "default":
        wanted = manifest.default_group_ids()
    elif spec == "all":
        wanted = manifest.group_ids()
    else:
        wanted = [g.strip() for g in spec.split(",") if g.strip()]
        unknown = [g for g in wanted if g not in manifest.group_ids()]
        if unknown:
            raise SystemExit(
                f"unknown group(s): {', '.join(unknown)}\n"
                f"available: {', '.join(manifest.group_ids())}"
            )
    return [e for e in manifest.entries if e.group in wanted]


# ---------------------------------------------------------------------------
# hashing and formatting
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def human(n: int) -> str:
    """SI units, because that is what publishers quote file sizes in."""
    for unit in ("B", "kB", "MB", "GB"):
        if abs(n) < 1000 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000.0
    return f"{n:.1f} GB"


# ---------------------------------------------------------------------------
# transfer
# ---------------------------------------------------------------------------
class TransferError(RuntimeError):
    pass


def _open(url: str, offset: int = 0, timeout: int = 60):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    if offset:
        req.add_header("Range", f"bytes={offset}-")
    return urllib.request.urlopen(req, timeout=timeout)


def download(entry: Entry, dest: Path, *, retries: int = 4, quiet: bool = False) -> None:
    """Fetch one entry to ``dest``, resuming a previous partial transfer.

    The partial file is ``dest.part``. It survives a failure on purpose: a
    dropped connection 80 MB into a 94 MB parquet should cost the last 14 MB,
    not the whole thing. The rename to ``dest`` happens only after the digest
    checks pass, so a file that exists at its final name is a file that was
    verified -- there is no state in which a truncated download looks complete.
    """
    part = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, retries + 1):
        offset = part.stat().st_size if part.exists() else 0
        try:
            with _open(entry.url, offset) as resp:
                # A server that ignores Range answers 200 with the whole file.
                # Appending to the partial would then produce a corrupt double.
                if offset and resp.status != 206:
                    offset = 0
                    part.unlink(missing_ok=True)

                mode = "ab" if offset else "wb"
                got = offset
                with part.open(mode) as out:
                    while True:
                        block = resp.read(CHUNK)
                        if not block:
                            break
                        out.write(block)
                        got += len(block)
                        if not quiet and entry.bytes:
                            pct = min(100.0, 100.0 * got / entry.bytes)
                            print(
                                f"\r    {entry.path}  {pct:5.1f}%  {human(got)}",
                                end="",
                                file=sys.stderr,
                                flush=True,
                            )
            if not quiet:
                print("", file=sys.stderr)
            break

        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            if isinstance(exc, urllib.error.HTTPError) and exc.code in (400, 401, 403, 404, 410):
                # Retrying a 404 just wastes the user's time; it will still be
                # a 404 in eight seconds.
                raise TransferError(f"{entry.url}: HTTP {exc.code} {exc.reason}") from exc
            if attempt == retries:
                raise TransferError(f"{entry.url}: {exc}") from exc
            back = 2 ** attempt
            print(
                f"    {entry.path}: {exc} -- retry {attempt}/{retries - 1} in {back}s",
                file=sys.stderr,
            )
            time.sleep(back)

    actual = part.stat().st_size
    digest = sha256_file(part)

    if entry.sha256 is not None and digest != entry.sha256:
        part.unlink(missing_ok=True)
        raise TransferError(
            f"{entry.path}: SHA-256 mismatch\n"
            f"  expected {entry.sha256}\n"
            f"  got      {digest}\n"
            "  The publisher gives a digest for this file, so this is a hard "
            "error: the bytes are not the bytes the manifest describes."
        )

    if actual != entry.bytes:
        # A warning, not an error. GEO publishes no digests and a submitter can
        # replace a supplementary file in place; the honest response is to say
        # the manifest is stale, not to refuse to work.
        print(
            f"    note: {entry.path} is {human(actual)}, manifest says "
            f"{human(entry.bytes)} -- the source has changed since 2026-08-07",
            file=sys.stderr,
        )

    part.replace(dest)


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------
def write_checksums(out: Path, records: dict[str, str]) -> None:
    """Write ``sha256sum``-format lines, sorted by path.

    This file is the conformance artefact: the R implementation writes the same
    bytes for the same corpus. Hence no timestamp, no tool name, no locale-
    dependent formatting -- two spaces between digest and path, LF endings,
    ASCII sort, exactly as coreutils does it, so `sha256sum -c` also works.
    """
    lines = [f"{records[p]}  {p}\n" for p in sorted(records)]
    (out / "checksums.sha256").write_text("".join(lines), encoding="utf-8", newline="\n")


def write_provenance(out: Path, manifest: Manifest, entries: Sequence[Entry],
                     records: dict[str, str]) -> None:
    """Write the human- and machine-readable record of what was fetched.

    Unlike checksums.sha256 this is informational, and the two implementations
    are allowed to differ in whitespace. It carries no timestamp either: a
    provenance file that changes on every run cannot be diffed against the last
    one, which is the only thing anybody ever wants to do with it.
    """
    doc = {
        "manifest_sha256": manifest.digest,
        "schema_version": manifest.schema_version,
        "files": [
            {
                "path": e.path,
                "url": e.url,
                "source": e.source,
                "group": e.group,
                "bytes_expected": e.bytes,
                "sha256_expected": e.sha256,
                "sha256_observed": records.get(e.path),
                "checksum_authority": (
                    "publisher" if e.sha256 is not None else "trust-on-first-use"
                ),
            }
            for e in sorted(entries, key=lambda x: x.path)
            if e.path in records
        ],
        "sources": {
            k: {
                "name": v.get("name"),
                "homepage": v.get("homepage"),
                "licence": v.get("licence"),
                "citation": v.get("citation"),
            }
            for k, v in sorted(manifest.sources.items())
        },
    }
    text = json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    (out / "provenance.json").write_text(text, encoding="utf-8", newline="\n")


def read_checksums(out: Path) -> dict[str, str]:
    f = out / "checksums.sha256"
    if not f.exists():
        return {}
    records: dict[str, str] = {}
    for line in f.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, _, path = line.partition("  ")
        records[path] = digest
    return records


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def cmd_plan(manifest: Manifest, entries: Sequence[Entry], out: Path,
             max_bytes: int) -> int:
    by_group: dict[str, list[Entry]] = {}
    for e in entries:
        by_group.setdefault(e.group, []).append(e)

    have = {e.path for e in entries if (out / e.path).exists()}
    todo = [e for e in entries if e.path not in have]

    print(f"manifest    {manifest.digest[:16]}...  schema {manifest.schema_version}")
    print(f"destination {out}")
    print()
    print(f"{'group':<14}{'files':>6}{'size':>12}   what it is for")
    print("-" * 78)
    for gid, es in by_group.items():
        meta = next(g for g in manifest.groups if g["id"] == gid)
        total = sum(e.bytes for e in es)
        print(f"{gid:<14}{len(es):>6}{human(total):>12}   {meta['title']}")
    print("-" * 78)

    selected = sum(e.bytes for e in entries)
    remaining = sum(e.bytes for e in todo)
    print(f"{'selected':<14}{len(entries):>6}{human(selected):>12}")
    if have:
        print(f"{'already here':<14}{len(have):>6}{human(selected - remaining):>12}")
        print(f"{'to download':<14}{len(todo):>6}{human(remaining):>12}")
    print(f"{'ceiling':<14}{'':>6}{human(max_bytes):>12}"
          f"   {100.0 * selected / max_bytes:.0f}% used")

    if selected > max_bytes:
        print()
        print(f"REFUSED: the selection is {human(selected)}, the ceiling is "
              f"{human(max_bytes)}.")
        print(f"         Over by {human(selected - max_bytes)}. Narrow --groups, "
              "or raise --max-bytes if you mean it.")
        return 1
    return 0


def cmd_fetch(manifest: Manifest, entries: Sequence[Entry], out: Path, *,
              max_bytes: int, force: bool, quiet: bool) -> int:
    total = sum(e.bytes for e in entries)
    if total > max_bytes:
        print(f"REFUSED: selection is {human(total)}, ceiling is {human(max_bytes)}",
              file=sys.stderr)
        return 1

    out.mkdir(parents=True, exist_ok=True)
    records = read_checksums(out)
    fetched = skipped = 0

    for i, e in enumerate(entries, 1):
        dest = out / e.path
        if dest.exists() and not force:
            # Present is not the same as correct. If the file has a published
            # digest, check it; a half-written file from an interrupted earlier
            # run must not be silently accepted as done.
            if e.sha256 is not None:
                digest = records.get(e.path) or sha256_file(dest)
                if digest != e.sha256:
                    print(f"[{i}/{len(entries)}] {e.path}: on disk but wrong digest, refetching")
                    dest.unlink()
                else:
                    records[e.path] = digest
                    skipped += 1
                    continue
            else:
                records.setdefault(e.path, sha256_file(dest))
                skipped += 1
                continue

        print(f"[{i}/{len(entries)}] {e.path}  ({human(e.bytes)}, {e.source})")
        download(e, dest, quiet=quiet)
        records[e.path] = sha256_file(dest)
        fetched += 1

    write_checksums(out, records)
    write_provenance(out, manifest, entries, records)

    on_disk = sum((out / e.path).stat().st_size for e in entries if (out / e.path).exists())
    print()
    print(f"fetched {fetched}, already present {skipped}, "
          f"{human(on_disk)} in {out}")
    print(f"wrote {out / 'checksums.sha256'} and {out / 'provenance.json'}")
    return 0


def cmd_verify(manifest: Manifest, entries: Sequence[Entry], out: Path) -> int:
    recorded = read_checksums(out)
    if not recorded:
        print(f"no checksums.sha256 in {out} -- nothing has been fetched here",
              file=sys.stderr)
        return 1

    missing: list[str] = []
    bad: list[str] = []
    ok = 0

    for e in entries:
        dest = out / e.path
        if not dest.exists():
            missing.append(e.path)
            continue
        digest = sha256_file(dest)

        # Published digest beats recorded digest. A recorded one only says the
        # file has not changed since it was fetched, which is worth much less
        # if what was fetched was wrong.
        expected = e.sha256 or recorded.get(e.path)
        if expected is None:
            bad.append(f"{e.path}: no digest to check against")
        elif digest != expected:
            authority = "publisher" if e.sha256 else "first fetch"
            bad.append(f"{e.path}: differs from the {authority} digest")
        else:
            ok += 1

    print(f"verified {ok}/{len(entries)} files against {out / 'checksums.sha256'}")
    for m in missing:
        print(f"  MISSING  {m}")
    for b in bad:
        print(f"  BAD      {b}")
    return 0 if not missing and not bad else 1


def cmd_inventory(manifest: Manifest, out: Path, readme: Path, check: bool) -> int:
    """Rewrite the file table in README.md from what is actually on disk.

    The table was previously marked ``BEGIN GENERATED`` with nothing generating
    it, which is worse than a hand-written table: a stale block that claims to
    be current is trusted, and this one had drifted into reporting binary
    megabytes under an SI label. Sizes here come from ``stat``, digests from
    ``checksums.sha256``, and ``--check`` fails rather than writes so CI notices
    the drift instead of a reader.
    """
    recorded = read_checksums(out)
    if not recorded:
        print(f"no checksums.sha256 in {out} -- fetch the corpus first", file=sys.stderr)
        return 1

    rows, total = [], 0
    for path in sorted(recorded):
        dest = out / path
        if not dest.exists():
            print(f"  MISSING  {path}", file=sys.stderr)
            return 1
        size = dest.stat().st_size
        total += size
        rows.append(f"| `{path}` | {human(size)} | `{recorded[path][:12]}` |")

    # provenance.json is written by this script rather than fetched, so it has
    # no publisher digest and is not in checksums.sha256. Listing it silently
    # among the fetched files is what made the old count read 34.
    prov = out / "provenance.json"
    extra = (f"\n\nAlongside them, `provenance.json` ({human(prov.stat().st_size)}) is written by "
             f"the fetcher rather than downloaded, so it carries no publisher digest and is not "
             f"in `checksums.sha256`.") if prov.exists() else ""

    block = (
        f"\n{len(rows)} fetched files, {human(total)} on disk, against the "
        f"{human(manifest.expected_total_bytes)} the manifest declares.\n"
        f"Digests are the first 12 characters of the SHA-256 in `checksums.sha256`; "
        f"`verify` checks the full value.\n\n"
        "| File | Size | SHA-256 |\n|---|---:|---|\n"
        + "\n".join(rows) + extra + "\n"
    )

    text = readme.read_text(encoding="utf-8")
    start = "<!-- BEGIN GENERATED: inventory -->"
    end = "<!-- END GENERATED: inventory -->"
    if start not in text or end not in text:
        print(f"no inventory markers in {readme}", file=sys.stderr)
        return 1
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    new = f"{head}{start}\n{block}\n{end}{tail}"

    if check:
        if new != text:
            print(f"{readme} inventory is stale; run without --check to rewrite")
            return 1
        print(f"{readme} inventory is current: {len(rows)} files, {human(total)}")
        return 0

    readme.write_text(new, encoding="utf-8", newline="\n")
    print(f"rewrote the inventory in {readme}: {len(rows)} files, {human(total)}")
    return 0


def cmd_self_test(manifest: Manifest) -> int:
    """Check the manifest against itself. No network, no disk."""
    problems: list[str] = []

    n = len(manifest.entries)
    total = sum(e.bytes for e in manifest.entries)

    if n != manifest.expected_total_files:
        problems.append(f"file count is {n}, manifest declares {manifest.expected_total_files}")
    if total != manifest.expected_total_bytes:
        problems.append(
            f"byte total is {total}, manifest declares {manifest.expected_total_bytes} "
            f"(off by {total - manifest.expected_total_bytes})"
        )
    if total > manifest.budget_bytes:
        problems.append(f"whole corpus is {human(total)}, over the {human(manifest.budget_bytes)} ceiling")

    seen: set[str] = set()
    for e in manifest.entries:
        if e.path in seen:
            problems.append(f"duplicate destination path: {e.path}")
        seen.add(e.path)
        if e.source not in manifest.sources:
            problems.append(f"{e.path}: source {e.source!r} is not declared under sources:")
        if not e.url.startswith("https://"):
            problems.append(f"{e.path}: url is not https")
        if e.sha256 is not None and len(e.sha256) != 64:
            problems.append(f"{e.path}: sha256 is {len(e.sha256)} characters, not 64")
        if os.path.isabs(e.path) or ".." in Path(e.path).parts:
            problems.append(f"{e.path}: destination escapes the output directory")

    for g in manifest.groups:
        declared = g.get("bytes")
        actual = sum(e.bytes for e in manifest.entries if e.group == g["id"])
        if declared is not None and int(declared) != actual:
            problems.append(f"group {g['id']}: declares {declared} bytes, files sum to {actual}")

    print(f"manifest {manifest.digest[:16]}...")
    print(f"  {len(manifest.groups)} groups, {n} files, {human(total)} "
          f"({100.0 * total / manifest.budget_bytes:.0f}% of the ceiling)")
    print(f"  {sum(1 for e in manifest.entries if e.sha256)} files carry a publisher digest, "
          f"{sum(1 for e in manifest.entries if not e.sha256)} are trust-on-first-use")

    if problems:
        print()
        for p in problems:
            print(f"  PROBLEM  {p}")
        return 1
    print("  self-test passed")
    return 0


# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(
        prog="fetch_test_data.py",
        description="Fetch the FALCONAge public test corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage\n-----\n")[-1],
    )
    ap.add_argument("--manifest", type=Path, default=here / "datasets.yaml")
    ap.add_argument("--out", type=Path, default=here,
                    help="destination directory (default: alongside the manifest)")
    ap.add_argument("--groups", default="default",
                    help="'default', 'all', or a comma-separated list of group ids")
    ap.add_argument("--max-bytes", type=int, default=None,
                    help="override the ceiling declared in the manifest")
    ap.add_argument("--dry-run", action="store_true", help="show the plan and stop")
    ap.add_argument("--verify", action="store_true", help="re-check files already on disk")
    ap.add_argument("--self-test", action="store_true", help="check the manifest, touch nothing")
    ap.add_argument("--inventory", action="store_true",
                    help="rewrite the file table in README.md from what is on disk")
    ap.add_argument("--check", action="store_true",
                    help="with --inventory: fail if the table is stale instead of rewriting it")
    ap.add_argument("--force", action="store_true", help="refetch files that are already present")
    ap.add_argument("--quiet", action="store_true", help="no per-file progress")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = ap.parse_args(argv)

    if not args.manifest.exists():
        raise SystemExit(f"manifest not found: {args.manifest}")

    manifest = load_manifest(args.manifest)

    if args.self_test:
        return cmd_self_test(manifest)

    if args.inventory:
        return cmd_inventory(manifest, args.out.resolve(),
                             args.manifest.parent / "README.md", args.check)

    entries = select(manifest, args.groups)
    if not entries:
        raise SystemExit(f"--groups {args.groups} selected nothing")

    out = args.out.resolve()
    max_bytes = args.max_bytes if args.max_bytes is not None else manifest.budget_bytes

    if args.verify:
        return cmd_verify(manifest, entries, out)
    if args.dry_run:
        return cmd_plan(manifest, entries, out, max_bytes)

    rc = cmd_plan(manifest, entries, out, max_bytes)
    if rc:
        return rc
    print()
    return cmd_fetch(manifest, entries, out, max_bytes=max_bytes,
                     force=args.force, quiet=args.quiet)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        # A partial transfer is kept, not cleaned up: the next run resumes it.
        print("\ninterrupted; partial transfers kept, rerun to resume", file=sys.stderr)
        sys.exit(130)
    except TransferError as exc:
        sys.exit(f"\ndownload failed: {exc}")
