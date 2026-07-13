from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# The guard is a PowerShell 7 script; Windows PowerShell 5.1 is not a substitute,
# so skip rather than fail where pwsh is not installed.
pytestmark = [
    pytest.mark.skipif(
        sys.platform != "win32", reason="PowerShell memory guard is Windows-only"
    ),
    pytest.mark.skipif(
        shutil.which("pwsh") is None, reason="PowerShell 7 (pwsh) is not installed"
    ),
]


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check-memory-cap.ps1"


def _run_memory_guard(max_used_gb: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "pwsh",
            "-NoLogo",
            "-NoProfile",
            "-File",
            str(SCRIPT),
            "-MaxUsedGb",
            str(max_used_gb),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_memory_guard_reports_real_nonzero_telemetry() -> None:
    result = _run_memory_guard(1024)

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["total_gb"] > 0
    assert 0 <= payload["free_gb"] <= payload["total_gb"]
    assert payload["used_gb"] == pytest.approx(payload["total_gb"] - payload["free_gb"], abs=0.02)
    assert payload["within_cap"] is True


def test_memory_guard_fails_when_usage_exceeds_cap() -> None:
    result = _run_memory_guard(1)

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["used_gb"] >= 1
    assert payload["within_cap"] is False
