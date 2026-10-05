#!/usr/bin/env bash
set -euo pipefail
python -m pytest tests/test_physical_components.py tests/test_spectral_data.py tests/test_calibrate_stacks.py -q
python -m jsr_repro.validate_spectral --output runs/spectral-validation
