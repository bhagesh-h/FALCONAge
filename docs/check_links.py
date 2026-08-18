#!/usr/bin/env python3
"""Every internal link in the rendered site points at something that exists.

WHY THIS EXISTS. Renaming a heading moves its anchor. Quarto does not warn: it
emits ``<a href="#counting-probes-is-not-weighing-them">`` for a section that is
now called something else, the page builds clean, the link silently scrolls
nowhere, and the only way to find out is for a reader to click it. Eighteen
headings were renamed at once here; without this the odds of getting all of
them right by hand were poor and the odds of *knowing* were zero.

Checked against ``_site``, not against the ``.qmd`` sources, because that is
what a reader loads and because Quarto rewrites relative paths on the way out:
``../science.qmd#x`` in the source is ``../science.html#x`` in the output, and
only the second one has to resolve.

Also checked, because it lives in the same place and nothing else looks at it:
every page carries a ``<link rel="icon">`` whose target exists. The site was
deployed for months with a blank browser tab -- the ``favicon:`` key was in the
design and never made it into the generated config -- and no test noticed,
because a missing icon breaks nothing that a build can see.

WHAT IS NOT CHECKED. External URLs, which would make this a network test with a
network test's flakiness, and ``mailto:``. The generated API trees under
``reference/`` and ``r/`` are skipped as link *sources*: quartodoc and pkgdown
emit cross-references to symbols by their own rules and a false positive there
would be noise in a check that has to stay believable. They are still valid
link *targets*.

Usage
-----
    python docs/check_links.py
"""

from __future__ import annotations

import html
import re
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote, urldefrag, urlsplit

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "docs" / "_site"

#: Pages whose links are checked. Generated reference output is excluded as a
#: source; see the module docstring.
SKIP_AS_SOURCE = ("reference/", "r/")

#: Link targets that a bare `quarto render` does not produce, because a later
#: step in the docs workflow writes them into `_site`: pkgdown builds `r/`, and
#: `build_downloads.py` writes `downloads/`. Reporting them as broken locally
#: would train a reader to ignore this check, which is the only failure mode
#: that would actually matter.
#:
#: Not a blanket exemption. The download *filenames* are pinned in
#: reference-groups.yml and `build_docs.py --check` asserts the navbar and the
#: build agree on them, so a wrong name is still caught, just not here.
BUILT_AFTER_RENDER = ("downloads/", "r/")

HREF = re.compile(r'<a\b[^>]*?href="([^"]+)"', re.I)
ID = re.compile(r'\bid="([^"]+)"')
NAME = re.compile(r'<a\b[^>]*?\bname="([^"]+)"', re.I)

#: The tab icon. Quarto emits one `<link rel="icon">` per page from the
#: `favicon:` key; pkgdown draws its own head and gets one from an
#: `in_header` include. Both are set in build_docs.py, and neither is
#: visible in anything else this repository checks -- the site rendered,
#: deployed and served for months with a blank tab and every test green,
#: because a missing favicon breaks nothing except recognising the tab.
LINK = re.compile(r"<link\b[^>]*>", re.I)
ATTR = re.compile(r'\b(rel|href)="([^"]*)"', re.I)

#: Where the site is served from, e.g. `/FALCONAge/`. pkgdown's pages sit at
#: two depths and share one include, so its icon href has to be
#: root-relative, which only resolves under this prefix.
def deploy_prefix() -> str:
    m = re.search(r"^\s*site-url:\s*(\S+)",
                  (ROOT / "docs" / "_quarto.yml").read_text(encoding="utf-8"),
                  re.M)
    return urlsplit(m.group(1)).path if m else "/"


def check_icons(pages: list[Path]) -> list[str]:
    """Every rendered page carries an icon link that resolves."""
    prefix = deploy_prefix()
    bad: list[str] = []
    for page in pages:
        rel = page.relative_to(SITE).as_posix()
        text = page.read_text(encoding="utf-8", errors="ignore")
        hrefs = []
        for tag in LINK.findall(text):
            attrs = {k.lower(): v for k, v in ATTR.findall(tag)}
            if "icon" in attrs.get("rel", "").lower().split() and attrs.get("href"):
                hrefs.append(html.unescape(attrs["href"]))
        if not hrefs:
            bad.append(f"{rel}: no <link rel=\"icon\">")
            continue
        for href in hrefs:
            if href.startswith(("http://", "https://", "data:")):
                continue
            if href.startswith("/"):
                if not href.startswith(prefix):
                    bad.append(f"{rel}: icon -> {href}   (outside {prefix})")
                    continue
                dest = SITE / unquote(href[len(prefix):])
            else:
                dest = (page.parent / unquote(href)).resolve()
            if not dest.exists():
                bad.append(f"{rel}: icon -> {href}   (no such file)")
    return bad


def ids_in(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return {html.unescape(m) for m in ID.findall(text)} | \
           {html.unescape(m) for m in NAME.findall(text)}


def main() -> int:
    if not SITE.exists():
        print(f"{SITE.relative_to(ROOT)} does not exist; render the site first:\n"
              "  docker run --rm -v \"$PWD:/work\" -w /work/docs "
              "falconage-quarto:nojupyter quarto render")
        return 1

    pages = sorted(SITE.rglob("*.html"))
    anchors = {p: ids_in(p) for p in pages}

    broken: list[str] = []
    counts: dict[str, int] = defaultdict(int)
    for page in pages:
        rel = page.relative_to(SITE).as_posix()
        if rel.startswith(SKIP_AS_SOURCE):
            continue
        text = page.read_text(encoding="utf-8", errors="ignore")
        for raw in HREF.findall(text):
            href = html.unescape(raw).strip()
            if not href or href.startswith(("http://", "https://", "mailto:",
                                            "javascript:", "data:", "tel:")):
                continue
            target, frag = urldefrag(href)
            frag = unquote(frag)
            counts["checked"] += 1

            dest = page.parent if not target else (page.parent / unquote(target))
            if target:
                dest = dest.resolve()
                try:
                    within = dest.relative_to(SITE).as_posix()
                except ValueError:
                    within = ""
                if within.startswith(BUILT_AFTER_RENDER):
                    counts["deferred"] += 1
                    continue
                if dest.is_dir():
                    dest = dest / "index.html"
                if not dest.exists():
                    broken.append(f"{rel}: -> {href}   (no such file)")
                    continue
            else:
                dest = page

            if frag and dest.suffix.lower() == ".html":
                if dest not in anchors:
                    anchors[dest] = ids_in(dest)
                if frag not in anchors[dest]:
                    where = "this page" if dest == page else \
                        dest.relative_to(SITE).as_posix()
                    broken.append(
                        f"{rel}: -> {href}\n      no id {frag!r} in {where}")

    icons = check_icons(pages)
    if icons:
        print(f"{len(icons)} page(s) with a missing or broken tab icon:")
        print()
        for b in icons[:20]:
            print(f"  {b}")
        if len(icons) > 20:
            print(f"  ... and {len(icons) - 20} more")
        print("The icon comes from `favicon:` in _quarto.yml and the pkgdown "
              "`in_header` include, both written by docs/build_docs.py.")
        return 1

    if broken:
        print(f"{len(broken)} broken internal link(s) of {counts['checked']} "
              f"checked across {len(pages)} page(s):\n")
        for b in sorted(broken)[:60]:
            print(f"  {b}")
        if len(broken) > 60:
            print(f"  ... and {len(broken) - 60} more")
        print("\nA renamed heading moves its anchor. Update the links that "
              "point at it, in the same change.")
        return 1

    print(f"{len(pages)} rendered page(s) all carry a tab icon that resolves")
    print(f"{counts['checked'] - counts['deferred']} internal link(s) across "
          f"{len(pages)} rendered page(s) all resolve, anchors included "
          f"({counts['deferred']} into downloads/ and r/ skipped: a later build "
          f"step writes those)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
