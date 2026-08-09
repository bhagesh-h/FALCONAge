"""Single source of truth for the package version.

r/DESCRIPTION and the git tag must agree with this string; the release workflow
refuses to publish when they do not (see PUBLISHING.md).
"""

__version__ = "1.1.0"

#: Version of the clock registry schema and contents. Pinned separately from
#: the package: a bug fix in the scoring loop must not silently change which
#: coefficients a result was computed from, and a coefficient correction must
#: be visible even when the code is untouched.
REGISTRY_VERSION = "1.1.0"
