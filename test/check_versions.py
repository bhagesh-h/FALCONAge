#!/usr/bin/env python3
"""Every place the version is written must agree, and with the tag if there is one.

WHY THIS EXISTS, AND WHY IT IS A SCRIPT RATHER THAN THREE INLINE CI STEPS.

Pushing v1.0.0 broke the release workflow on its very first step:

    v = tomllib.loads(...)["project"]["version"]
    KeyError: 'version'

`python/pyproject.toml` declares `dynamic = ["version"]` and points hatchling at
`src/falconage/_version.py`, so there is no `[project].version` key to read. The
check was gated on `startsWith(github.ref, 'refs/tags/v')`, so it had never run
before -- it was wrong from the day it was written and nothing noticed until a
tag existed.

The three checks it replaces lived in two jobs and two languages: pyproject in
Python, DESCRIPTION and CITATION.cff in R, each with its own parsing. One of the
three was broken and the other two were fine, which is what three copies of one
idea buys you. This is the single copy, and it runs locally -- which is the part
that would have caught the original.

Usage
-----
    python test/check_versions.py                        # the files agree
    python test/check_versions.py --tag v1.0.0           # ...and the tag agrees
    python test/check_versions.py --tag "$GITHUB_REF"    # refs/tags/v1.0.0 is fine
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def python_version() -> tuple[str, str]:
    """The version hatchling will build, found the way hatchling finds it.

    Reads `[tool.hatch.version].path` rather than assuming
    `src/falconage/_version.py`: if the file moves, hatchling follows the config
    and so should this.
    """
    cfg = tomllib.loads(
        (ROOT / "python" / "pyproject.toml").read_text(encoding="utf-8"))
    project = cfg.get("project", {})

    if "version" in project:
        return str(project["version"]), "python/pyproject.toml [project].version"

    if "version" not in (project.get("dynamic") or []):
        raise SystemExit(
            "python/pyproject.toml declares neither a static [project].version "
            "nor version in [project].dynamic")

    rel = cfg.get("tool", {}).get("hatch", {}).get("version", {}).get("path")
    if not rel:
        raise SystemExit(
            "python/pyproject.toml sets a dynamic version but no "
            "[tool.hatch.version].path to read it from")

    src = (ROOT / "python" / rel).read_text(encoding="utf-8")
    m = re.search(r"""^__version__\s*=\s*["']([^"']+)["']""", src, re.M)
    if not m:
        raise SystemExit(f"no __version__ in python/{rel}")
    return m.group(1), f"python/{rel}"


def description_version() -> tuple[str, str]:
    for line in (ROOT / "r" / "DESCRIPTION").read_text(encoding="utf-8").splitlines():
        if line.startswith("Version:"):
            return line.split(":", 1)[1].strip(), "r/DESCRIPTION"
    raise SystemExit("no Version: field in r/DESCRIPTION")


def citation_version() -> tuple[str, str]:
    for line in (ROOT / "CITATION.cff").read_text(encoding="utf-8").splitlines():
        if line.startswith("version:"):
            return line.split(":", 1)[1].strip().strip("\"'"), "CITATION.cff"
    raise SystemExit("no version: field in CITATION.cff")


def image_label_versions() -> list[tuple[str, str]]:
    """``org.opencontainers.image.version`` from each Dockerfile.

    A published image carries its labels wherever it is pulled from, and on a
    registry they are often the only provenance a reader gets. Both files said
    ``1.0.0`` for the whole of the 1.0.0 cycle, so pushing them would have put
    an image labelled as the previous release on Docker Hub. It went unnoticed
    because a label is not executed and nothing read it; reading it here costs
    nothing and makes it one more thing that has to agree before a tag.
    """
    out = []
    for rel in ("docker/Dockerfile.cpu", "docker/Dockerfile.cuda"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        m = re.search(r"""org\.opencontainers\.image\.version=["']([^"']+)["']""",
                      text)
        if not m:
            raise SystemExit(
                f"{rel} sets no org.opencontainers.image.version label. An "
                "image on a registry is provenance-free without it.")
        out.append((m.group(1), f"{rel} image.version label"))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", default="",
                    help="a tag or a full git ref; 'refs/tags/' and a leading "
                         "'v' are stripped. Omit to check file agreement only.")
    args = ap.parse_args(argv)

    found = [python_version(), description_version(), citation_version()]
    found += image_label_versions()

    tag = re.sub(r"^refs/tags/", "", args.tag).lstrip("v").strip()
    if tag:
        found.append((tag, f"git tag v{tag}"))

    for value, where in found:
        print(f"  {value:<12} {where}")

    distinct = {v for v, _ in found}
    if len(distinct) > 1:
        print(f"\nversion disagreement: {', '.join(sorted(distinct))}")
        print("  A release whose tag, wheel and citation file name different "
              "versions cannot be repaired afterwards -- the tag is what a "
              "citation points at.")
        return 1

    print(f"\nall {len(found)} agree on {distinct.pop()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
