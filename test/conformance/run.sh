#!/usr/bin/env sh
# The conformance suite, in the falconage-conformance image (docker/Dockerfile.conformance).
# Needs the test corpus (test/data/README.md). Exits non-zero on an unexplained difference.
set -e
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/python/src:${PYTHONPATH:-}"
python test/conformance/prepare.py
Rscript test/conformance/run_r.R
/opt/biolearn/bin/python test/conformance/run_biolearn.py
python test/conformance/compare.py
