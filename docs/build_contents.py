#!/usr/bin/env python3
"""Rebuild the in-page contents list of the two long documents from their headings.

WHY THESE ARE GENERATED. `science.qmd` and `architecture.qmd` are long enough to
need a contents list in the body, and both carried a hand-written one. Both had
drifted: the science list pointed at ``#reference-list`` for a section renamed
to References, and at ``#design-blueprint`` under the label "Design blueprint
for the new tool"; the architecture list pointed at
``#algorithms---the-complete-operation-catalogue`` and ``#what-v11-added-and-
where-it-sits``, neither of which is an id on the page. Four dead links out of
twenty-three, in the navigation aid, on the two pages nobody reads end to end.

Nothing catches that by eye, and the failure is silent: a browser given an
anchor it cannot find scrolls nowhere and reports nothing. So the list is
derived from the headings it describes, the same way the catalogue and the
gallery are, and `docs/check_links.py` verifies the result against the rendered
site.

Usage
-----
    python docs/build_contents.py           # rewrite both lists
    python docs/build_contents.py --check   # fail if either is stale (for CI)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGES = ("docs/science.qmd", "docs/architecture.qmd")

BEGIN = "<!-- BEGIN GENERATED: contents -->"
END = "<!-- END GENERATED: contents -->"


def slug(heading: str) -> str:
    """Pandoc's identifier for a heading text, as Quarto emits it.

    Numbers and punctuation are dropped from the front, ``&`` and friends are
    dropped entirely, and runs of whitespace become single hyphens. A literal
    " - " survives as "---", which is why the appendix anchors look the way
    they do. Verified against the ids in ``_site``.
    """
    h = re.sub(r"\s*\{#[^}]+\}\s*$", "", heading).strip()
    h = re.sub(r"[*`]", "", h).lower()
    h = re.sub(r"[^\w\s.-]", "", h)
    h = re.sub(r"\s+", "-", h.strip())
    return re.sub(r"^[^a-z]+", "", h)


def headings(body: str) -> list[str]:
    """The ``##`` headings that are actually sections of the document.

    A heading inside ``::: {.panel-tabset}`` is a tab label, not a section:
    Quarto turns it into a tab and emits no id for it, so listing one in the
    contents produces a link to an anchor that cannot exist. Two of these were
    in the architecture page's list, pointing at ``#python`` and ``#r``.

    Also skips fenced code, where a line beginning ``## `` is a shell comment.
    """
    out, fence, divs = [], False, 0
    tabset = None
    for line in body.splitlines():
        s = line.strip()
        if s.startswith("```"):
            fence = not fence
            continue
        if fence:
            continue
        if s.startswith(":::"):
            if re.match(r":::+\s*\{", s):
                divs += 1
                if "panel-tabset" in s and tabset is None:
                    tabset = divs
            else:
                if tabset is not None and divs == tabset:
                    tabset = None
                divs = max(0, divs - 1)
            continue
        m = re.match(r"^## (.+)$", line)
        if m and tabset is None:
            out.append(m.group(1).strip())
    return out


def render(text: str) -> str:
    """The contents block for one document, from its own ``##`` headings."""
    body = text.split(END, 1)[-1] if END in text else text
    heads = headings(body)

    numbered, appendices, tail = [], [], []
    for h in heads:
        label = re.sub(r"^\d+\.\s*", "", h)
        label = re.sub(r"\s*\{#[^}]+\}\s*$", "", label)
        if h.lower().startswith("appendix"):
            appendices.append((label, slug(h)))
        elif re.match(r"^\d+\.", h):
            numbered.append((label, slug(h)))
        else:
            tail.append((label, slug(h)))

    out = [BEGIN, "", "Contents", ""]
    for i, (label, anchor) in enumerate(numbered, start=1):
        out.append(f"{i}. [{label}](#{anchor})")
    for label, anchor in tail:
        out.append(f"- [{label}](#{anchor})")
    if appendices:
        out += ["", "Appendices", ""]
        for label, anchor in appendices:
            short = re.sub(r"^Appendix\s+", "", label)
            out.append(f"- [{short}](#{anchor})")
    out += ["", END]
    return "\n".join(out)


def apply(path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8")
    block = render(text)
    if BEGIN in text:
        new = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), block.replace("\\", "\\\\"),
                     text, flags=re.S)
    else:
        # First run: replace the hand-written list, which starts at the line
        # "Contents" and ends at the first heading after it.
        m = re.search(r"(?m)^Contents\n.*?(?=^## )", text, re.S)
        if not m:
            raise SystemExit(f"{path}: no 'Contents' block and no generated markers")
        new = text[:m.start()] + block + "\n\n" + text[m.end():]
    return text, new


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if either list is out of date")
    args = ap.parse_args(argv)

    stale = []
    for rel in PAGES:
        p = ROOT / rel
        old, new = apply(p)
        if old == new:
            continue
        if args.check:
            stale.append(rel)
        else:
            p.write_text(new, encoding="utf-8", newline="\n")

    if args.check:
        if stale:
            print("stale contents list in: " + ", ".join(stale))
            print("  run docs/build_contents.py")
            return 1
        print(f"the contents list in {len(PAGES)} long document(s) matches its headings")
        return 0

    for rel in PAGES:
        n = len(re.findall(r"(?m)^## ", (ROOT / rel).read_text(encoding="utf-8")))
        print(f"wrote the contents list for {rel} ({n} sections)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
