$ErrorActionPreference = 'Stop'
python -m pytest tests/test_physical_components.py tests/test_spectral_data.py tests/test_calibrate_stacks.py -q
if ($LASTEXITCODE -ne 0) { throw 'Spectral data tests failed' }
python -m jsr_repro.validate_spectral --output runs/spectral-validation
if ($LASTEXITCODE -ne 0) { throw 'Spectral pipeline validation failed' }
