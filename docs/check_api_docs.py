#!/usr/bin/env python3
"""Every ``fa.<name>`` written in the documentation must exist in the package.

WHY THIS EXISTS. The README's quick start told readers to call
``fa.report(...)``, ``fa.cox(...)``, ``fa.probe(...)``,
``fa.preprocess_methylation(...)`` and ``fa.preprocess_clinical(...)``. None of
the five existed. The first example on the landing page raised AttributeError
on the last line, and nothing anywhere failed because of it -- prose is not
executed, and a name in a fenced block is just text.

That is the whole class of bug this catches: documentation that describes an
API the package does not have. It is cheap to check and it does not survive
being checked once, because the drift comes back the next time a function is
renamed.

Also checks the string arguments where a wrong value is equally silent --
``acceleration(method=)`` had ``"both"`` documented, which is not one of the
three the function accepts.

Usage
-----
    python docs/check_api_docs.py

Needs falconage importable. Exits non-zero listing every bad reference.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# Prose files a reader copies from. Generated pages are included too: they are
# generated from templates that can be just as wrong.
#
# `.claude/skills/` is in the list for a stronger reason than the rest. Those
# files are read by a model that then *runs* what they say, so a name that does
# not exist there does not confuse a reader into checking -- it becomes a
# command. The first draft of that skill named five readers that live under
# `fa.preprocess` as though they were top level, and invented a CLI verb
# outright; this is what caught it.
SOURCES = (
    [ROOT / "README.md", ROOT / "r" / "README.md"]
    + sorted((ROOT / "docs").rglob("*.qmd"))
    + sorted((ROOT / "docs").glob("*.md"))
    + sorted((ROOT / ".claude" / "skills").rglob("*.md"))
)

# Things that look like `fa.x` in prose but are not API references.
IGNORE = {"py", "R"}

# Keyword arguments whose accepted values are a closed set. A wrong one is as
# broken as a wrong function name and just as invisible in a code fence.
#
# Matched per call, not per keyword. A first attempt checked every `method=`
# anywhere in the file and duly flagged `method="pca"` on plot.clock_pca and
# `method="umap"` on the atlas -- different functions with their own vocabulary.
# A checker that cries wolf gets switched off, so the pattern anchors on the
# function name and only reads the arguments belonging to that call.
ENUMS = [
    ("acceleration", "method", {"absolute", "residual", "both", "within_group"}),
    ("agreement", "method", {"pearson", "spearman", "kendall"}),
    ("score", "imputation", {"reference", "mean", "none"}),
    ("score", "device", {"auto", "cpu", "cuda", "mps"}),
]


def main() -> int:
    try:
        import falconage as fa
    except ImportError:
        print("falconage is not importable; install it first "
              "(pip install ./python)")
        return 1

    public = {n for n in dir(fa) if not n.startswith("_")}
    problems: list[str] = []

    for path in SOURCES:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(ROOT)

        seen: set[str] = set()
        for m in re.finditer(r"\bfa\.([A-Za-z_][A-Za-z0-9_]*)", text):
            name = m.group(1)
            if name in IGNORE or name in public or name in seen:
                continue
            seen.add(name)
            line = text[:m.start()].count("\n") + 1
            near = [p for p in sorted(public)
                    if p.startswith(name[:4]) or name in p]
            hint = f"  did you mean {', '.join(near[:3])}?" if near else ""
            problems.append(f"{rel}:{line}: fa.{name} does not exist.{hint}")

        for func, kw, allowed in ENUMS:
            # The call and its arguments, up to the closing paren on that line.
            for call in re.finditer(rf"\b{func}\s*\(([^)\n]*)\)", text):
                for m in re.finditer(rf'\b{kw}\s*=\s*["\']([a-z_]+)["\']',
                                     call.group(1)):
                    val = m.group(1)
                    if val in allowed:
                        continue
                    line = text[:call.start()].count("\n") + 1
                    problems.append(
                        f"{rel}:{line}: {func}({kw}={val!r}) is not accepted. "
                        f"One of: {', '.join(sorted(allowed))}.")

    # CLI verbs, from the parser rather than from a list kept in step by hand.
    # A documented verb that does not exist is worse than a wrong function name:
    # it is a whole command that fails, and the skill files are read by
    # something that runs them.
    verbs = _cli_verbs()
    if verbs:
        for path in SOURCES:
            if not path.exists():
                continue
            rel = path.relative_to(ROOT)
            text = path.read_text(encoding="utf-8")
            for m in re.finditer(r"(?m)^\s*(?:\$\s*)?falconage ([a-z][a-z-]*)", text):
                if m.group(1) not in verbs:
                    line = text[:m.start()].count("\n") + 1
                    problems.append(
                        f"{rel}:{line}: `falconage {m.group(1)}` is not a verb. "
                        f"One of: {', '.join(sorted(verbs))}.")

    if problems:
        print(f"{len(problems)} documentation reference(s) do not match the API:")
        for p in problems:
            print(f"  {p}")
        return 1

    print(f"every fa.* reference in {len(SOURCES)} document(s) resolves, "
          f"every enumerated argument value is accepted, and every documented "
          f"CLI verb is one of the {len(verbs)} the parser defines")
    return 0


def _cli_verbs() -> set[str]:
    """The subcommand names the CLI parser actually registers."""
    try:
        from falconage.cli.app import build_parser
    except Exception:
        return set()
    for action in build_parser()._subparsers._group_actions:  # noqa: SLF001
        if hasattr(action, "choices") and action.choices:
            return set(action.choices)
    return set()


if __name__ == "__main__":
    sys.exit(main())
