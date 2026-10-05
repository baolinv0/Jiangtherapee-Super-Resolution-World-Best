param([string]$Output = "runs/verification")
$ErrorActionPreference = "Stop"
python -m pytest -q
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m jsr_repro.validate --output $Output
exit $LASTEXITCODE
