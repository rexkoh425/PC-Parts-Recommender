[CmdletBinding()]
param(
    [ValidateSet("all", "python", "web")]
    [string]$Suite = "all",
    [switch]$IncludeIntegration,
    [switch]$IncludeSlow,
    [switch]$Sync
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepositoryRoot = Split-Path -Parent $PSScriptRoot
$PythonPath = Join-Path $RepositoryRoot ".venv\Scripts\python.exe"

function Invoke-Checked {
    param(
        [Parameter(Mandatory)]
        [string]$Command,
        [Parameter(ValueFromRemainingArguments)]
        [string[]]$CommandArguments
    )

    & $Command @CommandArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Command $CommandArguments"
    }
}

Push-Location $RepositoryRoot
try {
    if ($Suite -in @("all", "python")) {
        if ($Sync) {
            Write-Warning (
                "uv sync restores the locked CPU-only PyTorch wheel. " +
                "Run scripts/setup-gpu.ps1 again before host GPU training."
            )
            Invoke-Checked "uv" "sync" "--locked"
        }
        if (-not (Test-Path -LiteralPath $PythonPath)) {
            throw "Missing .venv. Run 'uv sync --locked', then optionally setup-gpu.ps1."
        }

        Invoke-Checked $PythonPath "-m" "ruff" "check" `
            "packages/core/src" "services" "pipelines" "tests" "db"
        Invoke-Checked $PythonPath "-m" "mypy" "packages/core/src" "services"

        $MarkerParts = @()
        if (-not $IncludeIntegration) {
            $MarkerParts += "not integration"
        }
        if (-not $IncludeSlow) {
            $MarkerParts += "not slow"
        }
        $PytestArguments = @("-m", "pytest", "tests")
        if ($MarkerParts.Count -gt 0) {
            $PytestArguments += @("-m", ($MarkerParts -join " and "))
        }
        Invoke-Checked $PythonPath @PytestArguments
    }

    if ($Suite -in @("all", "web")) {
        if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
            throw "Node.js and npm are required for web checks."
        }
        if (-not (Test-Path -LiteralPath (Join-Path $RepositoryRoot "node_modules"))) {
            Invoke-Checked "npm" "ci"
        }
        Invoke-Checked "npm" "run" "lint:web"
        Invoke-Checked "npm" "--workspace" "apps/web" "run" "typecheck"
        Invoke-Checked "npm" "--workspace" "apps/web" "run" "test"
        Invoke-Checked "npm" "run" "build:web"
    }
}
finally {
    Pop-Location
}
