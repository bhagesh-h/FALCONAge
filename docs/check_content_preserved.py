#!/usr/bin/env python3
"""Prove that reorganising the documentation lost nothing.

WHY. The site's pages are being split, merged and moved (todo item 27). A
reorganisation is exactly when a paragraph goes missing without anyone
noticing: it was on the old page, the new page reads well without it, and no
build fails. So before anything moves, every block of every page is recorded
here; afterwards, every recorded block has to exist somewhere in the new set of
pages.

WHAT A BLOCK IS. A paragraph, a list item, a table row, a whole code block, or
a heading. Blocks are compared after normalising whitespace, heading level,
heading ids and link targets, so demoting a heading or re-pointing a link at a
page's new path is not a loss, and changing a word is.

WHEN A BLOCK IS CHANGED ON PURPOSE. A correction (item 28) makes the old block
disappear. ``--accept`` records every currently missing block as amended, with
the reason given, so the correction is a line in the inventory rather than a
silent difference. Review the list ``--check`` prints before accepting it.

Usage
-----
    python docs/check_content_preserved.py --write            # record the current pages
    python docs/check_content_preserved.py --check            # fail on any missing block
    python docs/check_content_preserved.py --accept "reason"  # record missing blocks as amended
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
INVENTORY = DOCS / "content-inventory.json"
#: Generated or third-party trees that are not the site's own prose.
SKIP_DIRS = {"_site", ".quarto", "reference", "r", "figures", "images", "downloads"}


def pages() -> list[Path]:
    out = []
    for p in sorted(DOCS.rglob("*")):
        if p.suffix not in (".qmd", ".md") or not p.is_file():
            continue
        if SKIP_DIRS.intersection(p.relative_to(DOCS).parts[:-1]):
            continue
        out.append(p)
    return out


def normalise(text: str) -> str:
    s = text.strip()
    s = re.sub(r"^#{1,6}\s+", "", s)                     # heading level
    s = re.sub(r"\s*\{[#.][^}]*\}\s*$", "", s)             # {#id} {.class} attributes
    s = re.sub(r"\]\([^)]*\)", "]()", s)                   # link and image targets
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def blocks(path: Path) -> list[tuple[str, str]]:
    """(heading, text) for every block on a page, front matter excluded."""
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[tuple[str, str]] = []
    if lines and lines[0].strip() == "---":
        end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), 0)
        # A page title is a heading, so a section heading that becomes the
        # title of its own page has not been lost.
        for line in lines[1:end]:
            m = re.match(r"^title:\s*(.+)$", line)
            if m:
                out.append(("", m.group(1).strip().strip("'\"")))
        lines = lines[end + 1:]
    heading = ""
    buf: list[str] = []
    fence: str | None = None
    code: list[str] = []
    in_comment = False

    def flush():
        if buf:
            out.append((heading, "\n".join(buf)))
            buf.clear()

    for line in lines:
        st = line.strip()
        if fence is not None:
            code.append(line)
            if st.startswith(fence) and st.strip("`~") == "":
                out.append((heading, "\n".join(code)))
                code, fence = [], None
            continue
        if in_comment:
            in_comment = "-->" not in st
            continue
        if st.startswith("<!--"):
            flush()
            in_comment = "-->" not in st
            continue
        m = re.match(r"^(`{3,}|~{3,})", st)
        if m:
            flush()
            fence = m.group(1)
            code = [line]
            continue
        if not st or st.startswith(":::"):
            flush()
            continue
        if re.match(r"^#{1,6}\s", st):
            flush()
            heading = normalise(st)
            out.append((heading, st))
            continue
        if st.startswith("|"):
            flush()
            if not re.fullmatch(r"\|[\s:|-]+\|?", st):
                out.append((heading, st))
            continue
        if re.match(r"^([-*+]|\d+[.)])\s", st):
            flush()
        buf.append(line)
    flush()
    return [(h, t) for h, t in out if normalise(t)]


def digest(text: str) -> str:
    return hashlib.sha1(normalise(text).encode("utf-8")).hexdigest()[:16]


def current() -> dict[str, tuple[str, str, str]]:
    """hash -> (page, heading, first words), over the pages as they are now."""
    found: dict[str, tuple[str, str, str]] = {}
    for p in pages():
        rel = p.relative_to(DOCS).as_posix()
        for h, t in blocks(p):
            found.setdefault(digest(t), (rel, h, normalise(t)[:90]))
    return found


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    g.add_argument("--accept", metavar="REASON")
    args = ap.parse_args(argv)

    now = current()
    if args.write:
        inv = {"blocks": {k: list(v) for k, v in sorted(now.items())}, "amended": {}}
        INVENTORY.write_text(json.dumps(inv, indent=0, sort_keys=True) + "\n",
                             encoding="utf-8", newline="\n")
        print(f"recorded {len(now)} blocks from {len(pages())} pages "
              f"in {INVENTORY.relative_to(ROOT)}")
        return 0

    inv = json.loads(INVENTORY.read_text(encoding="utf-8"))
    missing = {k: v for k, v in inv["blocks"].items()
               if k not in now and k not in inv["amended"]}
    if args.accept:
        for k, v in missing.items():
            inv["amended"][k] = {"was": v, "reason": args.accept}
        INVENTORY.write_text(json.dumps(inv, indent=0, sort_keys=True) + "\n",
                             encoding="utf-8", newline="\n")
        print(f"recorded {len(missing)} block(s) as amended: {args.accept}")
        return 0

    if missing:
        print(f"{len(missing)} block(s) of the recorded pages are missing:")
        for k, (page, heading, text) in sorted(missing.items(), key=lambda kv: kv[1]):
            print(f"  {page} :: {heading[:50]}\n      {text}")
        return 1
    print(f"all {len(inv['blocks'])} recorded blocks are present "
          f"({len(inv['amended'])} amended on purpose)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
