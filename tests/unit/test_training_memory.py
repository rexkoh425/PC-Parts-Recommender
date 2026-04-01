from __future__ import annotations

from pathlib import Path

import pytest
from training._common import (
    HostMemorySnapshot,
    estimate_materialized_file_memory_mib,
    portable_path_reference,
    require_host_memory_headroom,
)


def _snapshot(*, total_gib: float, available_gib: float) -> HostMemorySnapshot:
    gibibyte = 1024**3
    return HostMemorySnapshot(
        total_bytes=round(total_gib * gibibyte),
        available_bytes=round(available_gib * gibibyte),
        source="test",
    )


def test_host_memory_preflight_records_projected_usage_without_reading_live_host() -> None:
    preflight = require_host_memory_headroom(
        max_used_gib=55.0,
        estimated_additional_mib=512.0,
        minimum_free_mib=1024.0,
        snapshot=_snapshot(total_gib=61.7, available_gib=12.0),
    )

    report = preflight.to_dict()

    assert report["used_gib"] == pytest.approx(49.7, abs=0.001)
    assert report["projected_used_gib"] == pytest.approx(50.2, abs=0.001)
    assert report["projected_available_gib"] == pytest.approx(11.5, abs=0.001)
    assert report["max_used_gib"] == 55.0


def test_host_memory_preflight_rejects_an_at_or_above_cap_projection() -> None:
    with pytest.raises(MemoryError, match="at or above the 55.00 GiB cap"):
        require_host_memory_headroom(
            max_used_gib=55.0,
            estimated_additional_mib=1024.0,
            minimum_free_mib=0.0,
            snapshot=_snapshot(total_gib=61.7, available_gib=7.7),
        )


def test_host_memory_preflight_rejects_insufficient_remaining_headroom() -> None:
    with pytest.raises(MemoryError, match="below the required 1024 MiB headroom"):
        require_host_memory_headroom(
            max_used_gib=64.0,
            estimated_additional_mib=1536.0,
            minimum_free_mib=1024.0,
            snapshot=_snapshot(total_gib=61.7, available_gib=2.0),
        )


def test_materialized_file_estimate_accounts_for_every_input_and_runtime(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.json"
    first.write_bytes(b"a" * 1024)
    second.write_bytes(b"b" * (2 * 1024))

    estimate = estimate_materialized_file_memory_mib(
        [first, second],
        expansion_factor=8.0,
        runtime_allowance_mib=16.0,
    )

    assert estimate == pytest.approx(16.0234375)


def test_portable_path_reference_is_workspace_relative_or_redacted(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    assert (
        portable_path_reference(
            workspace / "data" / "catalogue.csv",
            workspace_root=workspace,
        )
        == "data/catalogue.csv"
    )
    assert (
        portable_path_reference(
            tmp_path / "operator-private" / "catalogue.csv",
            workspace_root=workspace,
        )
        == "<external>/catalogue.csv"
    )


@pytest.mark.parametrize(
    ("paths", "expansion_factor", "runtime_allowance_mib"),
    (((), 8.0, 1.0), ((Path("missing.jsonl"),), 8.0, 1.0), ((Path("x"),), 0.5, 1.0)),
)
def test_materialized_file_estimate_rejects_invalid_inputs(
    paths,  # type: ignore[no-untyped-def]
    expansion_factor: float,
    runtime_allowance_mib: float,
) -> None:
    with pytest.raises((FileNotFoundError, ValueError)):
        estimate_materialized_file_memory_mib(
            paths,
            expansion_factor=expansion_factor,
            runtime_allowance_mib=runtime_allowance_mib,
        )


@pytest.mark.parametrize(
    ("max_used_gib", "estimated_additional_mib", "minimum_free_mib"),
    ((0.0, 0.0, 0.0), (55.0, -1.0, 0.0), (55.0, 0.0, -1.0)),
)
def test_host_memory_preflight_rejects_invalid_configuration(
    max_used_gib: float,
    estimated_additional_mib: float,
    minimum_free_mib: float,
) -> None:
    with pytest.raises(ValueError):
        require_host_memory_headroom(
            max_used_gib=max_used_gib,
            estimated_additional_mib=estimated_additional_mib,
            minimum_free_mib=minimum_free_mib,
            snapshot=_snapshot(total_gib=61.7, available_gib=12.0),
        )
