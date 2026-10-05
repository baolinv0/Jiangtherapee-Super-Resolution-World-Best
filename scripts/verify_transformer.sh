#!/usr/bin/env sh
set -eu
python -m pytest -q
python -m jsr_repro.validate_transformer --output "${1:-runs/transformer-verification}"
