# Local bootstrap. Superseded by scripts/dev.ps1 once Compose landed.
param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"

if ($Force -or -not (Test-Path ".venv")) {
    uv venv
}
uv sync
npm install
Write-Host "Environment ready."
