#!/bin/sh
# =============================================================================
# FALCONAge container entrypoint
# =============================================================================
#
# Three jobs, and nothing else. Anything more here becomes behaviour that only
# exists inside the container, which is the fastest way to make "it works in
# Docker" stop meaning anything.
#
#   1. Let a mounted checkout take precedence over the installed copy, so an
#      edit-run cycle does not need a rebuild.
#   2. Treat a bare invocation as a help request rather than an error.
#   3. Step out of the way when the first argument is an interpreter, so the
#      test suite and the GPU check can be run without --entrypoint. Both need
#      the image's environment and neither is a `falconage` subcommand, and
#      `docker run --entrypoint python img -m pytest` puts the image name
#      between the entrypoint and its arguments, which nobody remembers.
# -----------------------------------------------------------------------------
set -eu

if [ -n "${FALCONAGE_SOURCE:-}" ]; then
  if [ ! -d "${FALCONAGE_SOURCE}/python/src/falconage" ]; then
    echo "FALCONAGE_SOURCE=${FALCONAGE_SOURCE} does not look like a FALCONAge checkout" >&2
    echo "  (expected ${FALCONAGE_SOURCE}/python/src/falconage to exist)" >&2
    exit 2
  fi

  # Prepending to PYTHONPATH rather than `pip install -e`: it is instant, it
  # leaves the image's site-packages untouched, and it works when the mount is
  # read-only -- which it usually is, because the whole point of mounting a
  # checkout is to edit it on the host.
  PYTHONPATH="${FALCONAGE_SOURCE}/python/src${PYTHONPATH:+:${PYTHONPATH}}"
  export PYTHONPATH

  # reticulate imports falconage through the same interpreter, so the R side
  # picks up the mounted source too and the two cannot drift apart mid-session.
  echo "FALCONAge: running mounted source at ${FALCONAGE_SOURCE}" >&2
fi

# `docker run falconage:1.0.0-cpu` with no arguments is somebody finding out
# what the image does, not a malformed command.
case "${1:-}" in
  ""|-h|--help) exec falconage --help ;;
  python|python3|pytest|R|Rscript|sh|bash) exec "$@" ;;
esac

exec falconage "$@"
