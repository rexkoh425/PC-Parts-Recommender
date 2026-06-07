[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepositoryRoot = Split-Path -Parent $PSScriptRoot
$PythonPath = Join-Path $RepositoryRoot ".venv\Scripts\python.exe"
$CudaIndex = "https://download.pytorch.org/whl/cu130"
$TorchRequirement = "torch==2.13.0+cu130"

Push-Location $RepositoryRoot
try {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "uv is required and was not found on PATH."
    }
    if (-not (Test-Path -LiteralPath $PythonPath)) {
        Write-Host "Creating the locked Python environment before applying the CUDA override."
        & uv sync --locked
        if ($LASTEXITCODE -ne 0) {
            throw "uv sync failed with exit code $LASTEXITCODE."
        }
    }

    Write-Host "Installing $TorchRequirement into .venv from the official CUDA 13.0 index."
    & uv pip install `
        --python $PythonPath `
        --index-url $CudaIndex `
        --reinstall `
        $TorchRequirement
    if ($LASTEXITCODE -ne 0) {
        throw "CUDA PyTorch installation failed with exit code $LASTEXITCODE."
    }

    $VerificationCode = @"
import sys
import torch

print(f"torch={torch.__version__}")
print(f"cuda_build={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable. Check the NVIDIA driver and restart the shell.")
print(f"device={torch.cuda.get_device_name(0)}")
print(f"capability={torch.cuda.get_device_capability(0)}")
sys.exit(0)
"@
    & $PythonPath -c $VerificationCode
    if ($LASTEXITCODE -ne 0) {
        throw "PyTorch installed, but CUDA verification failed with exit code $LASTEXITCODE."
    }

    Write-Host "GPU setup verified."
    Write-Warning (
        "A later 'uv sync' restores the CPU wheel recorded in uv.lock. " +
        "Rerun scripts/setup-gpu.ps1 before host GPU training after every sync."
    )
}
finally {
    Pop-Location
}
