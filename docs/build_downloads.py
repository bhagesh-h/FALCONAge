#!/usr/bin/env python3
"""Build the offline copies of the documentation: one PDF, one markdown bundle.

WHY THESE EXIST. A documentation website is not something you can attach to a
thesis, read on a plane, cite a page of, or diff between releases. Every
documentation build produces these so they cannot fall behind the site.

    FALCONAge.pdf                    the whole site as one document
    FALCONAge-docs-markdown.zip      the same, as markdown, plus the R reference

WHY ONE COMBINED DOCUMENT RATHER THAN A QUARTO BOOK. Two reasons, both found by
trying the book first. A `website` project renders one PDF per page, which is
not a manual. A `book` project gives one document -- but books do not support
the typst format, and typst is the whole point: it ships inside Quarto, so the
PDF builds on a runner with no TeX distribution and in a fraction of the time.
Requiring LaTeX would mean apt-installing a texlive subset and then discovering
which further package `Rd2pdf` wanted, on someone else's CI minutes.

So the chapters are concatenated into one `.qmd` with their front matter
stripped and their headings demoted by one level, and that single document is
rendered twice: once to typst for the PDF, once to gfm for the markdown. No
project type, no LaTeX, no book.

WHY THERE IS NO SEPARATE R REFERENCE MANUAL. `R CMD Rd2pdf` produces the classic
CRAN manual and needs a working LaTeX install to do it, which is the dependency
this file exists to avoid. The R reference is in the markdown bundle instead,
converted from pkgdown's HTML with pandoc, and the rendered R site is on the
web already.

Usage
-----
    python docs/build_downloads.py --all
    python docs/build_downloads.py --pdf --out docs/_site/downloads
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GROUPS = HERE / "reference-groups.yml"
DEFAULT_OUT = HERE / "_site" / "downloads"

# The spine: the pages a person reads, in the order they read them.
#
# WHY THE GENERATED API REFERENCE IS NOT IN HERE. It was, and binding all 67
# pages produced a document nobody would read and a build nobody could
# maintain. Sixty quartodoc pages share heading names -- every one has
# "Parameters" and "Returns" -- and typst, unlike HTML, treats a duplicated or
# dangling label as a hard error rather than a dead link. Chasing anchor
# collisions across generated files is exactly the kind of upkeep that gets a
# feature switched off six months later.
#
# The API reference is a lookup table, and a lookup table wants search, which
# is what the website gives it. What is worth having on a plane is the
# narrative: what the thing is, how to use it, how to choose a clock, and the
# two long-form documents. Seven pages, all hand-written, all stable.
FRONT = ["index.qmd", "guide/FALCONAge.qmd", "guide/clocks.qmd", "clocks.qmd",
         "gpu.md", "science.qmd", "architecture.qmd"]

FRONTMATTER = re.compile(r"\A---\n.*?\n---\n", re.S)
# `[text](#anchor)` -> `text`. Every page carries a hand-written contents list
# whose anchors are generated from its own headings; demoting those headings
# into a combined document changes the anchors, and typst errors on a link to a
# label that does not exist. The combined document has its own table of
# contents, so the per-page ones are redundant anyway.
ANCHOR_LINK = re.compile(r"\[([^\]\[]*)\]\(#[^)]*\)")


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("  $", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def have(tool: str) -> bool:
    return shutil.which(tool) is not None


def demote(markdown: str) -> str:
    """Push every ATX heading down one level, so each chapter's `#` becomes `##`
    under the combined document's own title. Fenced code is left alone -- a `#`
    at the start of a line inside a shell block is a comment, not a heading."""
    out, fenced = [], False
    for line in markdown.split("\n"):
        if line.lstrip().startswith("```"):
            fenced = not fenced
        elif not fenced and re.match(r"#{1,5} ", line):
            line = "#" + line
        out.append(line)
    return "\n".join(out)


#: Typst renders a table at whatever width its content needs and lets the
#: columns collide when that is wider than the page, which is what the PDF's
#: tables were doing: a six-column table of clock metadata printed one column
#: on top of the next. Two things fix it together. Every table goes on its own
#: page turned sideways, which is 40% more width, and table text is set two
#: points smaller than body text, which buys the rest.
LANDSCAPE_OPEN = "```{=typst}\n#set page(flipped: true)\n```\n\n"
LANDSCAPE_CLOSE = "\n```{=typst}\n#set page(flipped: false)\n```\n"

#: Applied once, at the top of the document.
TYPST_PREAMBLE = (
    "```{=typst}\n"
    "#show table: set text(size: 8pt)\n"
    "#show table.cell: set par(justify: false)\n"
    "```\n"
)


def rotate_tables(markdown: str) -> tuple[str, int]:
    """Put every pipe table on a landscape page of its own.

    A table is a run of consecutive lines starting with `|`. Two kinds are
    left where they are. Fenced code, because a shell block can contain a line
    that starts with a pipe and turning the page sideways around a command
    would be strange. And anything inside a `:::` div, because Typst refuses
    page configuration inside a container: a table in a callout raised "page
    configuration is not allowed inside of containers" and took the whole
    build down with it. Those tables are small by nature -- a callout holding a
    six-column table is a different problem -- so leaving them inline costs
    nothing.
    """
    out: list[str] = []
    fenced = False
    depth = 0
    i = 0
    lines = markdown.split("\n")
    count = 0
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith("```"):
            fenced = not fenced
            out.append(line)
            i += 1
            continue
        if not fenced and line.lstrip().startswith(":::"):
            # An opening fence carries a class; a bare `:::` closes one.
            depth += 1 if line.strip().strip(":").strip() else -1
            depth = max(depth, 0)
            out.append(line)
            i += 1
            continue
        if not fenced and depth == 0 and line.startswith("|"):
            start = i
            while i < len(lines) and lines[i].startswith("|"):
                i += 1
            block = lines[start:i]
            # Two lines is a header and its rule with no rows: not a table.
            if len(block) > 2:
                count += 1
                out.append("")
                out.append(LANDSCAPE_OPEN.rstrip("\n"))
                out.append("")
                out.extend(block)
                out.append(LANDSCAPE_CLOSE.strip("\n"))
                out.append("")
            else:
                out.extend(block)
            continue
        out.append(line)
        i += 1
    return "\n".join(out), count


def combine() -> Path:
    """Concatenate the chapters into one .qmd and return its path."""
    site = yaml.safe_load(GROUPS.read_text(encoding="utf-8"))["site"]
    chapters = [c for c in FRONT if (HERE / c).exists()]
    if len(chapters) < 2:
        raise RuntimeError(
            "nothing to bind: none of " + ", ".join(FRONT) + " were found. "
            "Run this from the repository, not from a partial checkout.")

    parts = [
        "---\n"
        f'title: "{site["title"]}"\n'
        f'subtitle: "{" ".join(site["description"].split())}"\n'
        "author: Bhagesh Hunakunti\n"
        "toc: true\n"
        "toc-depth: 2\n"
        "number-sections: false\n"
        # Nothing here executes. clocks.qmd carries a live Python chunk that
        # queries the registry; it already ran when the site was rendered, and
        # running it again from a concatenated file with a different working
        # directory is a way to fail for no benefit.
        "execute:\n  enabled: false\n"
        "---\n",
        f"_Generated from the FALCONAge documentation site, {site['url']}_\n",
        TYPST_PREAMBLE,
    ]
    rotated = 0
    for n, c in enumerate(chapters, start=1):
        text = (HERE / c).read_text(encoding="utf-8")
        fm = FRONTMATTER.match(text)
        title = ""
        if fm:
            meta = yaml.safe_load(fm.group(0).strip("-\n")) or {}
            title = str(meta.get("title", "")).strip()
            text = text[fm.end():]
        if not title:
            m = re.search(r"^#\s+(.+)$", text, re.M)
            title = m.group(1).strip() if m else Path(c).stem
            text = re.sub(r"^#\s+.+$", "", text, count=1, flags=re.M)
        # Numbered, and each one starts a page. `number-sections` is not the
        # way to get there: two of these chapters number their own sections in
        # the heading text ("## 15. The layer that says what a score is
        # worth"), so turning Quarto's numbering on would print "1.15 15."
        # down the whole of the architecture chapter.
        body, k = rotate_tables(ANCHOR_LINK.sub(r"\1", demote(text)))
        rotated += k
        parts.append(f"\n\n{{{{< pagebreak >}}}}\n\n# Chapter {n}. {title}\n\n" + body)

    combined = HERE / "_combined.qmd"
    combined.write_text("\n".join(parts), encoding="utf-8", newline="\n")
    print(f"  combined {len(chapters)} chapters into {combined.name}, "
          f"{rotated} table(s) turned sideways")
    return combined


def build_pdf(out: Path) -> Path:
    if not have("quarto"):
        raise RuntimeError("quarto is not on PATH")
    src = combine()
    try:
        run(["quarto", "render", src.name, "--to", "typst",
             "--output", "FALCONAge.pdf"], cwd=HERE)
        built = HERE / "FALCONAge.pdf"
        target = out / "FALCONAge.pdf"
        shutil.move(str(built), target)
        return target
    finally:
        src.unlink(missing_ok=True)


def build_markdown(out: Path) -> Path:
    """The site as GitHub-flavoured markdown, both languages in one zip.

    The R half is converted from pkgdown's HTML rather than from the .Rd
    sources, because Rd is a TeX dialect and every direct converter for it
    loses the usage block -- the part of a help page anybody actually reads.
    """
    if not have("quarto"):
        raise RuntimeError("quarto is not on PATH")
    stage = HERE / "_md_out"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)

    src = combine()
    try:
        run(["quarto", "render", src.name, "--to", "gfm",
             "--output", "FALCONAge.md"], cwd=HERE)
        shutil.move(str(HERE / "FALCONAge.md"), stage / "FALCONAge.md")
    finally:
        src.unlink(missing_ok=True)

    r_html = sorted((HERE / "r" / "reference").glob("*.html"))
    if r_html and have("pandoc"):
        r_md = stage / "r-reference"
        r_md.mkdir(parents=True, exist_ok=True)
        for page in r_html:
            if page.name == "index.html":
                continue
            run(["pandoc", "-f", "html", "-t", "gfm", "--wrap=none",
                 "-o", str(r_md / f"{page.stem}.md"), str(page)])
        print(f"  converted {len(r_html) - 1} R help pages")
    else:
        print("  (no pkgdown output or no pandoc: skipping the R half)")

    for extra in ("README.md", "CHANGELOG.md", "CITATION.cff"):
        if (ROOT / extra).exists():
            shutil.copyfile(ROOT / extra, stage / extra)

    target = out / "FALCONAge-docs-markdown.zip"
    target.unlink(missing_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(stage).as_posix())
    shutil.rmtree(stage, ignore_errors=True)
    return target


BUILDERS = {"pdf": build_pdf, "markdown": build_markdown}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for name in BUILDERS:
        ap.add_argument(f"--{name}", action="store_true")
    ap.add_argument("--all", action="store_true", help="every artefact")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any requested artefact failed")
    args = ap.parse_args(argv)

    wanted = [n for n in BUILDERS if args.all or getattr(args, n)]
    if not wanted:
        ap.error("nothing requested; pass --all or one of "
                 + ", ".join(f"--{n}" for n in BUILDERS))

    args.out.mkdir(parents=True, exist_ok=True)
    failed = []
    for name in wanted:
        print(f"\n[{name}]")
        try:
            path = BUILDERS[name](args.out)
            print(f"  wrote {path.name}  ({path.stat().st_size / 1e6:.1f} MB)")
        except Exception as exc:
            print(f"  SKIPPED: {exc}")
            failed.append(name)

    print(f"\n{len(wanted) - len(failed)}/{len(wanted)} artefact(s) built into {args.out}")
    if failed:
        print("skipped: " + ", ".join(failed))
    return 1 if (failed and args.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
