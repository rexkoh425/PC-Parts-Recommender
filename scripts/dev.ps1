[CmdletBinding()]
param(
    [switch]$Build,
    [switch]$Detach,
    [switch]$WithDagster,
    [switch]$WithMlflow,
    [switch]$Down
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepositoryRoot = Split-Path -Parent $PSScriptRoot
$EnvironmentFile = Join-Path $RepositoryRoot ".env"
$EnvironmentExample = Join-Path $RepositoryRoot ".env.example"

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker Desktop is required and 'docker' was not found on PATH."
}

Push-Location $RepositoryRoot
try {
    & docker compose version | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Docker Compose is unavailable. Start or update Docker Desktop."
    }

    if (-not (Test-Path -LiteralPath $EnvironmentFile)) {
        Copy-Item -LiteralPath $EnvironmentExample -Destination $EnvironmentFile
        Write-Host "Created .env from development defaults. Review it before shared deployment."
    }

    $ComposeArguments = @("compose", "--env-file", ".env")
    if ($WithDagster) {
        $ComposeArguments += @("--profile", "pipeline")
    }
    if ($WithMlflow) {
        $ComposeArguments += @("--profile", "mlops")
    }

    if ($Down) {
        $ComposeArguments += "down"
    }
    else {
        $ComposeArguments += "up"
        if ($Build) {
            $ComposeArguments += "--build"
        }
        if ($Detach) {
            $ComposeArguments += "--detach"
        }
    }

    & docker @ComposeArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Docker Compose exited with code $LASTEXITCODE."
    }

    if ($Detach -and -not $Down) {
        & docker compose --env-file .env ps
    }
}
finally {
    Pop-Location
}

