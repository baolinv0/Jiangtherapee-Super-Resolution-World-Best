#!/usr/bin/env sh
set -eu
python -m pytest -q
python -m jsr_repro.validate --output "${1:-runs/verification}"
