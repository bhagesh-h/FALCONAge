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
#: docker/DOCKERHUB.md is here because it is the least correctable page in the
#: project. It is pushed to a registry, read by people who have not cloned
#: anything, and its example is often the first command they run; a function
#: name that does not exist fails on somebody else's machine with no way for
#: them to see that the page is wrong rather than their install.
SOURCES = (
    [ROOT / "README.md", ROOT / "r" / "README.md",
     ROOT / "docker" / "DOCKERHUB.md"]
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

    # The whole command line, not only the verb.
    #
    # The verb check above passed for years while six of the eight commands in
    # the skill's own reference could not run: `clocks --tier A` omitted the
    # required action, `score data.h5ad` and `bench results/` passed an input
    # positionally to verbs that want `--input`, `preprocess --out` was really
    # `--output`, and `power horvath2013` was really `--clock`. Every one of
    # them names a real verb, so nothing objected. Which arguments are
    # positional differs per verb and is not guessable, which is exactly why
    # this has to be checked against the parser rather than reviewed.
    problems += _check_cli_commands()

    if problems:
        print(f"{len(problems)} documentation reference(s) do not match the API:")
        for p in problems:
            print(f"  {p}")
        return 1

    print(f"every fa.* reference in {len(SOURCES)} document(s) resolves, "
          f"every enumerated argument value is accepted, every documented CLI "
          f"verb is one of the {len(verbs)} the parser defines, and every "
          f"documented command line parses")
    return 0


#: Placeholders a reader is meant to substitute. A command containing one is
#: a template, so it is checked for shape by filling them with a dummy rather
#: than skipped: `score --input <file>` should still fail if `--input` is wrong.
PLACEHOLDER = re.compile(r"<[^>]+>|\.\.\.|\$\{?\w+\}?")


def _check_cli_commands() -> list[str]:
    """Parse every documented ``falconage ...`` command with the real parser.

    Uses ``parse_args`` on the token list and catches the ``SystemExit``
    argparse raises, so nothing is executed: no file is read, no network is
    touched, and a command naming an input that does not exist still validates.
    Only the interface is under test, which is the part prose gets wrong.
    """
    try:
        from falconage.cli.app import build_parser
    except Exception:                                  # pragma: no cover
        return []

    import contextlib
    import io
    import shlex

    problems: list[str] = []
    for path in SOURCES:
        if not path.exists():
            continue
        rel = path.relative_to(ROOT)
        # architecture.qmd is the design record, written before the code and
        # deliberately wider than it. Its command lines are the specified
        # interface, several of which the shipped parser does not implement,
        # and §7.8 says so in the page. Checking them would fail on the gap the
        # page exists to document.
        if rel.as_posix() == "docs/architecture.qmd":
            continue
        text = path.read_text(encoding="utf-8")
        for m in re.finditer(r"(?m)^\s*(?:\$\s*)?falconage ((?:[^\n#]|\\\n)+)", text):
            raw = m.group(1).replace("\\\n", " ").strip()
            if not raw or raw.startswith("-"):
                continue
            filled = PLACEHOLDER.sub("PLACEHOLDER", raw)
            try:
                argv = shlex.split(filled)
            except ValueError:
                continue
            err = io.StringIO()
            try:
                with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                    build_parser().parse_args(argv)
            except SystemExit as exc:
                if exc.code:
                    line = text[:m.start()].count("\n") + 1
                    why = (err.getvalue().strip().splitlines() or ["rejected"])[-1]
                    problems.append(
                        f"{rel}:{line}: `falconage {raw[:60]}` does not parse. "
                        f"{why.split('error: ')[-1]}")
            except Exception:                          # pragma: no cover
                continue
    return problems


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
