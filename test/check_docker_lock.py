#!/usr/bin/env python3
"""The uv pinned in the Dockerfiles can read the lock file they install from.

WHY THIS EXISTS. ``uv.lock`` carries a ``revision`` field, and a uv older than
the one that wrote it refuses the whole file: ``Failed to parse `uv.lock` ``,
a TOML parse error, and a build that dies at the dependency layer. Both images
pinned ``uv==0.5.18`` while the lock moved to revision 3, so from some
regeneration onward neither ``docker/Dockerfile.cpu`` nor
``docker/Dockerfile.cuda`` built at all. Six minutes of build time to find out,
and the first instruction in ``README.md`` is one of those builds.

Nothing caught it. There is no workflow that builds an image, and the two
images that existed locally had been built when the lock was older. A stale
artefact hides the breakage that would have produced it.

WHAT THIS CHECKS, AND WHAT IT CANNOT. It compares the pinned uv against a
recorded floor per lock revision, and asserts both Dockerfiles agree on the
pin. It does not run uv, so it cannot discover the floor for a revision nobody
has tested: a new revision fails here with instructions rather than passing
silently, which is the right way round. Finding the floor is one command, and
it is in the failure message.

Usage
-----
    python test/check_docker_lock.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCK = ROOT / "python" / "uv.lock"
DOCKERFILES = ("docker/Dockerfile.cpu", "docker/Dockerfile.cuda")

#: lock ``revision`` -> the oldest uv that can read it, measured rather than
#: read off a changelog. Revision 3: 0.5.18 fails, 0.6.17 succeeds, and 0.6.17
#: through 0.8.17 export byte-identical requirement sets.
FLOOR = {3: (0, 6, 17)}


def _parts(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3])


def main() -> int:
    problems: list[str] = []

    m = re.search(r"^revision\s*=\s*(\d+)", LOCK.read_text(encoding="utf-8"), re.M)
    if m is None:
        print(f"{LOCK.relative_to(ROOT)} declares no revision; nothing to check")
        return 0
    revision = int(m.group(1))

    pins: dict[str, str] = {}
    for rel in DOCKERFILES:
        text = (ROOT / rel).read_text(encoding="utf-8")
        found = re.findall(r"uv==([0-9][0-9A-Za-z.\-]*)", text)
        if not found:
            problems.append(f"{rel} installs no pinned uv; it exports the lock "
                            "and needs one")
            continue
        if len(set(found)) > 1:
            problems.append(f"{rel} pins uv more than once and disagrees with "
                            f"itself: {sorted(set(found))}")
        pins[rel] = found[0]

    if len(set(pins.values())) > 1:
        problems.append(
            "the two images pin different uv versions "
            + ", ".join(f"{k}={v}" for k, v in pins.items())
            + ". They export the same lock; a difference here is a difference "
              "in what the two images install.")

    floor = FLOOR.get(revision)
    if floor is None:
        problems.append(
            f"{LOCK.relative_to(ROOT)} is revision {revision} and no floor is "
            f"recorded for it. Find it and add it to FLOOR:\n"
            f"      docker run --rm -v \"$PWD:/work\" -w /work/python "
            f"python:3.12-slim sh -c \\\n"
            f"        'pip install -q uv==X.Y.Z && uv export --frozen "
            f"--no-dev --no-emit-project \\\n"
            f"         --extra methylation --extra plot --extra anndata "
            f"--extra cli --format requirements-txt >/dev/null'")
    else:
        for rel, pin in pins.items():
            if _parts(pin) < floor:
                problems.append(
                    f"{rel} pins uv=={pin}, which cannot read a revision "
                    f"{revision} lock (floor {'.'.join(map(str, floor))}). "
                    f"The image will fail to build with \"Failed to parse "
                    f"`uv.lock`\".")

    if problems:
        print(f"{len(problems)} problem(s) with the Docker dependency layer:\n")
        for p in problems:
            print(f"  - {p}")
        return 1

    pin = next(iter(set(pins.values())))
    print(f"uv=={pin} in both images can read a revision {revision} lock "
          f"(floor {'.'.join(map(str, FLOOR[revision]))})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
