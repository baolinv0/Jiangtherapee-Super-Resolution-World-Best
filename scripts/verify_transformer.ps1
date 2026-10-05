param([string]$Output = "runs/transformer-verification")
$ErrorActionPreference = "Stop"
python -m pytest -q
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m jsr_repro.validate_transformer --output $Output
exit $LASTEXITCODE
