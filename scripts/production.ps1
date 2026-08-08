[CmdletBinding()]
param(
    [ValidateSet(
        "Validate",
        "Config",
        "DeployCore",
        "DeployPipeline",
        "DeployMlops",
        "DeployObservability",
        "Status",
        "Logs",
        "Backup",
        "Stop"
    )]
    [string]$Action = "Validate",
    [string]$EnvFile = ".env.production",
    [string]$Service,
    [ValidateRange(1, 5000)]
    [int]$Tail = 200
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepositoryRoot = Split-Path -Parent $PSScriptRoot
$ComposeFile = Join-Path $RepositoryRoot "docker-compose.production.yml"
$ResolvedEnvFile = if ([System.IO.Path]::IsPathRooted($EnvFile)) {
    $EnvFile
}
else {
    Join-Path $RepositoryRoot $EnvFile
}
$PythonPath = Join-Path $RepositoryRoot ".venv\Scripts\python.exe"
$PythonCommand = if (Test-Path -LiteralPath $PythonPath) { $PythonPath } else { "python" }

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

function Invoke-Preflight {
    if (-not (Test-Path -LiteralPath $ResolvedEnvFile -PathType Leaf)) {
        throw "Production environment file not found: $ResolvedEnvFile"
    }
    Invoke-Checked $PythonCommand `
        (Join-Path $RepositoryRoot "scripts\validate_production_env.py") `
        "--env-file" $ResolvedEnvFile
}

function Invoke-Compose {
    param([Parameter(ValueFromRemainingArguments)][string[]]$Arguments)

    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw "Docker is required for production action '$Action'."
    }
    $BaseArguments = @(
        "compose",
        "--env-file", $ResolvedEnvFile,
        "-f", $ComposeFile
    )
    $AllArguments = $BaseArguments + $Arguments
    Invoke-Checked "docker" @AllArguments
}

Push-Location $RepositoryRoot
try {
    Invoke-Preflight
    if ($Action -eq "Validate") {
        return
    }

    switch ($Action) {
        "Config" {
            Invoke-Compose "--profile" "pipeline" "--profile" "mlops" `
                "--profile" "observability" "--profile" "operations" `
                "--profile" "restore" "config" "--quiet"
        }
        "DeployCore" {
            Invoke-Compose "up" "-d" "postgres"
            Invoke-Compose "up" "migrate"
            Invoke-Compose "up" "catalog-release"
            Invoke-Compose "up" "-d" "api" "web"
            Invoke-Compose "ps"
        }
        "DeployPipeline" {
            Invoke-Compose "--profile" "pipeline" "up" "-d" `
                "dagster-code" "dagster-webserver" "dagster-daemon"
        }
        "DeployMlops" {
            Invoke-Compose "--profile" "mlops" "up" "mlflow-migrate"
            Invoke-Compose "--profile" "mlops" "up" "-d" "mlflow"
        }
        "DeployObservability" {
            Invoke-Compose "--profile" "observability" "up" "-d" `
                "postgres-exporter" "blackbox-exporter" "prometheus"
        }
        "Status" {
            Invoke-Compose "--profile" "pipeline" "--profile" "mlops" `
                "--profile" "observability" "ps" "-a"
        }
        "Logs" {
            if ([string]::IsNullOrWhiteSpace($Service)) {
                throw "-Service is required for the Logs action."
            }
            Invoke-Compose "logs" "--tail" $Tail.ToString() $Service
        }
        "Backup" {
            Invoke-Compose "--profile" "operations" "run" "--rm" "backup-postgres"
            Invoke-Compose "--profile" "operations" "run" "--rm" "backup-artifacts"
        }
        "Stop" {
            Invoke-Compose "--profile" "pipeline" "--profile" "mlops" `
                "--profile" "observability" "stop"
        }
    }
}
finally {
    Pop-Location
}
