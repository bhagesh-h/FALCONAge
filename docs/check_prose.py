#!/usr/bin/env python3
"""Fail on the punctuation habits that make documentation read as machine-written.

WHY THIS EXISTS. Every page in this repository was written with an em dash doing
the work of a colon, a comma, a full stop and a pair of parentheses, roughly
four hundred times, plus eighteen horizontal rules separating sections that
already had headings. None of it was wrong, and all of it together made a
scientific catalogue read like generated filler. Reading for it does not work,
because the habit is invisible to the person with it. Counting does.

WHAT IS ALLOWED, AND WHY

``| — |``
    An em dash alone in a table cell is standard notation for "not applicable",
    and the generated catalogue uses it in 140 cells. It is data, not prose.

Fenced blocks and indented output
    A hyphen inside captured program output or a shell command is part of the
    thing being documented. Rewriting it would corrupt the example.

``--`` inside a word or flag
    ``--dry-run`` is an option, not punctuation.

Usage
-----
    python docs/check_prose.py            # fail on any offender
    python docs/check_prose.py --list     # print them and exit 0
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

#: Prose a reader sees, which is prose that ships.
#:
#: Asked from git rather than globbed off disk. A working tree holds private
#: notes, a fetched corpus and generated output alongside the documentation,
#: and none of that is anybody's to read or this script's to police. Generated
#: API pages are excluded on top of that: their text comes from docstrings, so
#: the page is the wrong place to fix them.
def sources() -> list[Path]:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "ls-files", "--", "*.md", "*.qmd", "*.Rmd"],
            cwd=ROOT, capture_output=True, text=True, timeout=30, check=True).stdout
    except Exception as exc:                      # pragma: no cover
        raise SystemExit(
            f"cannot ask git which files are tracked ({exc}); this check reads "
            "the shipped documentation and will not guess at it from disk"
        ) from exc

    skip = {"reference", "man", "_site", ".quarto"}
    files = []
    for rel in out.splitlines():
        rel = rel.strip()
        if not rel:
            continue
        p = ROOT / rel
        if p.exists() and not (set(Path(rel).parts) & skip):
            files.append(p)
    return sorted(set(files))


CELL = re.compile(r"\|\s*—\s*\|")          # "not applicable" in a table
EM = re.compile(r"—")
INLINE_TRIPLE = re.compile(r"(?<!-) --- (?!-)")
RULE = re.compile(r"^-{3,}$")


def prose_lines(text: str):
    """Yield (lineno, line) outside fenced code, indented output and tables."""
    fence = False
    for i, line in enumerate(text.splitlines(), start=1):
        if line.lstrip().startswith("```"):
            fence = not fence
            continue
        if fence or line.startswith("    ") or line.startswith("\t"):
            continue
        yield i, line


def check(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    rel = path.relative_to(ROOT).as_posix()
    problems: list[str] = []

    lines = text.splitlines()
    in_yaml = bool(lines and lines[0].strip() == "---")
    yaml_end = 0
    if in_yaml:
        for i, ln in enumerate(lines[1:], start=2):
            if ln.strip() == "---":
                yaml_end = i
                break

    for n, line in prose_lines(text):
        bare = CELL.sub("| |", line)
        if EM.search(bare):
            problems.append(f"{rel}:{n}: em dash in prose. Use the mark it is "
                            f"standing in for: a colon, a comma, a full stop, "
                            f"or a pair of parentheses.\n      {line.strip()[:96]}")
        if INLINE_TRIPLE.search(line):
            problems.append(f"{rel}:{n}: ` --- ` renders as an em dash.\n"
                            f"      {line.strip()[:96]}")

    fence = False
    for n, line in enumerate(lines, start=1):
        if line.lstrip().startswith("```"):
            fence = not fence
            continue
        if fence or n <= yaml_end:
            continue
        if RULE.match(line.strip()):
            problems.append(f"{rel}:{n}: horizontal rule. The heading below it "
                            f"already separates the sections.")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--list", action="store_true",
                    help="report and exit 0, for a first pass over new prose")
    args = ap.parse_args(argv)

    files = sources()
    problems: list[str] = []
    for p in files:
        problems += check(p)

    if problems:
        print(f"{len(problems)} prose issue(s) across {len(files)} file(s):\n")
        for p in problems:
            print(f"  {p}")
        if args.list:
            return 0
        print("\nEach of these is punctuation standing in for punctuation. "
              "docs/check_prose.py --list to review without failing.")
        return 1

    print(f"{len(files)} prose file(s) clean: no em dash outside a table cell, "
          f"no ` --- `, no decorative horizontal rule")
    return 0


if __name__ == "__main__":
    sys.exit(main())
