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
         "gpu.md", "science.qmd", "architecture.qmd", "references.qmd"]

FRONTMATTER = re.compile(r"\A---\n.*?\n---\n", re.S)
# `[text](#anchor)` -> `text`. Every page carries a hand-written contents list
# whose anchors are generated from its own headings; demoting those headings
# into a combined document changes the anchors, and typst errors on a link to a
# label that does not exist. The combined document has its own table of
# contents, so the per-page ones are redundant anyway.
ANCHOR_LINK = re.compile(r"\[([^\]\[]*)\]\(#[^)]*\)")

#: `![caption](../images/x.png)` in a page that lives in docs/guide/.
#:
#: The combined document is written to docs/ and rendered from there, so every
#: chapter's relative paths are read from one directory up from where they were
#: written. A figure on a guide page points at `../images/`, which from docs/
#: resolves outside the project and silently drops out of the PDF: Typst prints
#: the alt text where the picture should be and the build still succeeds, so
#: nothing fails and the figure is simply missing.
GUIDE_IMAGE = re.compile(r"(!\[[^\]]*\]\()\.\./(images/)")

#: `[[7]](references.qmd#ref-7)` -> `[7]`.
#:
#: On the site a citation is a link to the references page. Bound into one
#: document that page is a chapter, so the link would point at a file that no
#: longer exists as a separate thing, and typst treats an unresolvable label as
#: a hard error rather than a dead link. The number is what carries the meaning
#: and the References chapter is in the same document with the same numbering,
#: so the bracket is kept and the link is dropped.
PDF_CITATION = re.compile(r"\[(\[\d+\])\]\((?:\.\./)?references\.qmd#ref-\d+\)")


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


#: A rotated page is 40% wider, and it costs the reader a page turn and a head
#: tilt. That trade is worth making for a fifteen-column table and is not worth
#: making for a three-column one, which was the bug: every table in the document
#: was being turned sideways, 75 of them, when only nine were wide enough to
#: need it.
#:
#: WHY NOT SIMPLY "FIVE COLUMNS OR A LONG CELL". That is the sibling project's
#: rule and it rotates two shapes of table that do not need it. A five-column
#: table of short numbers, like the GPU timings, is 69 characters wide and fits
#: portrait with room to spare. A two-column table whose second column holds a
#: paragraph also fits, because the paragraph WRAPS: a cell does not need its
#: natural width, it needs a column it can wrap inside.
#:
#: So the test is about what wrapping cannot fix. Wrapping trades width for
#: height, and it stops working when there are so many columns that none of them
#: is wide enough to hold a word, or when the table's natural width is so far
#: past the page that wrapping it produces a block taller than it is readable.
#:
#: PORTRAIT_BUDGET is the text block in characters: US Letter at 2.5cm margins
#: is 6.5 inches, which is about 118 characters of 9pt proportional text.
PORTRAIT_BUDGET = 118
#: Six columns on 6.5 inches is 1.08 inches each before padding, which is not
#: enough for prose in any of them. Five still works when the content is short.
LANDSCAPE_MIN_COLS = 6
#: How far past the budget a table's natural width may run before wrapping it
#: into portrait costs more in height than the rotation costs in page turns.
LANDSCAPE_WIDTH_FACTOR = 2.5

LANDSCAPE_OPEN = "```{=typst}\n#set page(flipped: true)\n```\n\n"
LANDSCAPE_CLOSE = "\n```{=typst}\n#set page(flipped: false)\n```\n"

#: A pipe that is part of a cell's text rather than a column separator. The
#: catalogue writes `\|coefficient\|` for absolute value, and counting those as
#: separators reports a three-column table as five and rotates it.
CELL_SPLIT = re.compile(r"(?<!\\)\|")

#: The `|---|:--:|` rule under a header. It is not data and its dashes would be
#: measured as though they were the widest cell in a narrow column.
ALIGN_ROW = re.compile(r"^[\s|:-]+$")


def table_cells(block: list[str]) -> list[list[str]]:
    """The rows of a pipe table, split into cells, with the alignment rule out."""
    rows = []
    for line in block:
        if ALIGN_ROW.match(line):
            continue
        # ZWSP is zero width on the page, so counting it would make a softened
        # identifier look wider than it prints and rotate tables that fit.
        cells = [c.strip().replace(ZWSP, "") for c in CELL_SPLIT.split(line)]
        # A row written `| a | b |` splits to an empty cell at each end.
        if cells and not cells[0]:
            cells = cells[1:]
        if cells and not cells[-1]:
            cells = cells[:-1]
        rows.append(cells)
    return rows


#: A token with no space in it that is longer than a narrow column. Typst breaks
#: lines between words and will not break inside one, so a cell holding
#: `quantile_normalize_and_scale_with_gold_standard` sets it on a single line,
#: overruns its column and prints on top of the neighbouring cell. That is the
#: overlap that survives rotation, because no amount of extra page width helps a
#: token that is never allowed to break.
LONG_TOKEN = re.compile(r"[^\s|]{25,}")
#: U+200B. Invisible, zero width, and a legal break point. Inserting it after the
#: separators inside a long identifier lets the cell wrap at a readable place
#: rather than at an arbitrary glyph.
ZWSP = "​"


def soften_long_tokens(row: str) -> str:
    """Let long identifiers in a table cell wrap, without changing what they say."""
    # The `|-----------|:---:|` rule is a run of dashes and would match the long
    # token pattern. Breaking it up stops it being an alignment row at all, and
    # the table then renders as ordinary paragraphs.
    if ALIGN_ROW.match(row):
        return row

    def fix(m: re.Match) -> str:
        token = m.group(0)
        # Never touch a URL: a zero-width space inside one is invisible here and
        # breaks the link if anybody copies it out of the PDF.
        if "://" in token:
            return token
        return re.sub(r"([_\-./])", r"\1" + ZWSP, token)

    return LONG_TOKEN.sub(fix, row)


def natural_width(rows: list[list[str]]) -> int:
    """Characters the table would occupy if nothing wrapped, padding included."""
    cols = max((len(r) for r in rows), default=0)
    if not cols:
        return 0
    per_column = [
        max((len(r[i]) for r in rows if i < len(r)), default=0)
        for i in range(cols)
    ]
    # Four characters an edge is the 8pt horizontal inset either side of a cell.
    return sum(per_column) + 4 * cols


def needs_landscape(block: list[str]) -> bool:
    """Whether this table earns a rotated page of its own."""
    rows = table_cells(block)
    if not rows:
        return False
    width = natural_width(rows)
    # A table that fits the text block as written is never rotated, whatever its
    # column count. The probe-masking table has six columns and is 83 characters
    # wide; turning it sideways would be a page turn to read something that was
    # already going to fit.
    if width <= PORTRAIT_BUDGET:
        return False
    cols = max(len(r) for r in rows)
    if cols >= LANDSCAPE_MIN_COLS:
        return True
    return width > PORTRAIT_BUDGET * LANDSCAPE_WIDTH_FACTOR


#: Applied once, at the top of the document. Everything here is layout that
#: Quarto's typst template does not give us and that the sibling project's
#: manual gets from its LaTeX preamble: a chapter opening that reads as a
#: chapter, section headings that are distinguishable from body text at a
#: glance, and tables with enough structure that a wrapped cell cannot be
#: mistaken for its neighbour.
#:
#: WHY THE TABLE STROKE MATTERS MORE THAN THE ROTATION. Quarto emits tables with
#: `stroke: none`, so a cell whose text wraps to three lines sits beside another
#: doing the same with nothing between them. That reads as overlap even when the
#: columns are correctly laid out and the text is wrapping properly. A hairline
#: under every row and a rule under the header is what separates them; the extra
#: horizontal inset is what keeps the text off the column boundary.
#:
#: The orange is `$falcon-orange` from the stylesheet, which was sampled from the
#: logo rather than chosen, so the PDF and the site carry the same accent. The
#: darker variant is the one used for links, because the brand orange does not
#: reach 4.5:1 on white.
BRAND = "#e06000"
ACCENT = "#a84800"
INK = "#1a1a1a"

TYPST_PREAMBLE = f"""```{{=typst}}
#let brand = rgb("{BRAND}")
#let accent = rgb("{ACCENT}")
#let ink = rgb("{INK}")

#set par(justify: true, leading: 0.62em)
#show link: set text(fill: accent)

// A running header naming the chapter, so a page torn out of a printed copy
// still says which chapter it came from. Queried rather than tracked by hand:
// `before(here())` finds the most recent chapter opening, which is the one this
// page belongs to. The title page and the contents have no chapter before them
// and correctly get no header.
#set page(header: context {{
  let seen = query(selector(heading.where(level: 1)).before(here()))
  if seen.len() > 0 {{
    set text(size: 8pt, fill: luma(95))
    grid(columns: (1fr, auto),
      align(left, seen.last().body),
      align(right)[FALCONAge])
    v(-7pt)
    line(length: 100%, stroke: 0.4pt + luma(185))
  }}
}})

// A chapter opens on its own page, names itself in the brand colour, and is
// separated from its first paragraph by a rule and real space. The pagebreak
// is emitted by the assembler rather than here, so a chapter that is already
// at the top of a page does not gain a blank one.
#show heading.where(level: 1): it => block(width: 100%, above: 0pt, below: 20pt)[
  #set text(size: 21pt, weight: "bold", fill: ink)
  #it.body
  #v(7pt)
  #line(length: 100%, stroke: 1.2pt + brand)
]

// WHICH LEVEL IS A SECTION. `demote()` pushes every heading in a page down one
// level so the chapter title can be the only level 1, and the source pages start
// their own sections at `##`. So in this document a section is level THREE, not
// two, and level two is empty. Styling two as the section and three as the
// subsection is why sections were setting at 11.5pt and reading as bold
// paragraphs rather than as headings. Two and three are both given the section
// treatment: two is unused today and would be a section if it ever appeared.
//
// The gap above each heading is larger than the gap below, so a heading sits
// with the text it introduces rather than floating between two paragraphs.
#show heading.where(level: 2): it => block(width: 100%, above: 24pt, below: 10pt)[
  #set text(size: 14.5pt, weight: "bold", fill: accent)
  #it.body
]
#show heading.where(level: 3): it => block(width: 100%, above: 24pt, below: 10pt)[
  #set text(size: 14.5pt, weight: "bold", fill: accent)
  #it.body
]
#show heading.where(level: 4): it => block(width: 100%, above: 17pt, below: 7pt)[
  #set text(size: 11.5pt, weight: "bold", fill: ink)
  #it.body
]
#show heading.where(level: 5): it => block(width: 100%, above: 13pt, below: 5pt)[
  #set text(size: 10.5pt, weight: "bold", style: "italic", fill: ink)
  #it.body
]

#set table(
  inset: (x: 8pt, y: 5pt),
  stroke: (_, y) => (bottom: if y == 0 {{ 0.7pt + ink }} else {{ 0.3pt + luma(190) }}),
)
#show table: set text(size: 9pt)
#show table.cell: set par(justify: false, leading: 0.5em)
#show table.cell.where(y: 0): set text(weight: "bold")

// Long commands and long clock ids are the two things that run past a column.
#show raw: set text(size: 8.5pt)
```
"""


def rotate_tables(markdown: str, soften: bool = True) -> tuple[str, int]:
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
            if soften:
                block = [soften_long_tokens(r) for r in block]
            # Two lines is a header and its rule with no rows: not a table.
            if len(block) > 2 and needs_landscape(block):
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


def combine(*, soften: bool = True) -> Path:
    """Concatenate the chapters into one .qmd and return its path.

    ``soften`` inserts zero-width break points inside long identifiers so typst
    can wrap them. It is wanted for the PDF and not for the markdown bundle: in
    a .md file the characters are still invisible, but anyone who copies a clock
    id out of the appendix table would carry them into their code, where the
    name silently stops matching.
    """
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
        # Three, not two. `demote()` makes a section level 3, so a depth of 2
        # listed the eight chapters and nothing else: a table of contents for a
        # two-hundred-page document with eight entries in it.
        "toc-depth: 3\n"
        "number-sections: false\n"
        # Quarto's typst default is 1.25in a side, which leaves six inches of
        # text on US Letter and is the single largest reason a table ran out of
        # room. 2.5cm is what the sibling project's manual uses and it returns
        # about three quarters of an inch of width to every table on every page,
        # rotated or not.
        "format:\n"
        "  typst:\n"
        "    papersize: us-letter\n"
        "    margin:\n"
        "      x: 2.5cm\n"
        "      y: 2.5cm\n"
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
        if Path(c).parent != Path("."):
            text = GUIDE_IMAGE.sub(r"\1\2", text)
        text = PDF_CITATION.sub(r"\1", text)
        body, k = rotate_tables(ANCHOR_LINK.sub(r"\1", demote(text)), soften)
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

    src = combine(soften=False)
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
