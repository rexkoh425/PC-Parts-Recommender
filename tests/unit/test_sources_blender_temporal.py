from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest
from pipelines.sources.base import RAW_SNAPSHOT_SCHEMA_VERSION
from pipelines.sources.blender import (
    BLENDER_LICENSE_NOTE,
    BLENDER_PARSER_VERSION,
    BLENDER_SNAPSHOT_URL,
)
from pipelines.sources.blender_temporal import (
    BlenderCohortContract,
    diff_blender_snapshots,
    load_blender_raw_snapshot,
)


def _observation(
    *,
    version: str = "4.0.0",
    scene: str = "junkshop",
    hardware: str = "AMD Ryzen 5 5600X 6-Core Processor",
    score: float = 50.0,
) -> dict[str, object]:
    return {
        "benchmark_script": {"label": "3.1.0"},
        "blender_version": {"version": version, "build_hash": "build-4"},
        "device_info": {
            "device_type": "CPU",
            "num_cpu_threads": 12,
            "compute_devices": [{"name": hardware, "type": "CPU"}],
        },
        "scene": {"label": scene, "checksum": "scene-4"},
        "stats": {"samples_per_minute": score},
        "system_info": {
            "system": "Windows",
            "num_cpu_cores": 6,
            "num_cpu_sockets": 1,
            "num_cpu_threads": 12,
        },
        "timestamp": "2026-07-22T12:00:00+00:00",
    }


def _submission(identifier: str, observations: list[dict[str, object]]) -> dict[str, object]:
    return {
        "id": identifier,
        "created_at": f"2026-07-2{identifier[-1]}T12:00:00+00:00",
        "data": observations,
    }


def _snapshot(
    root: Path,
    name: str,
    submissions: list[dict[str, object]],
    *,
    retrieved_at: str,
) -> Path:
    staging = root / f"{name}.zip"
    member = f"opendata-{name}.jsonl"
    payload = "".join(json.dumps(value, sort_keys=True) + "\n" for value in submissions)
    with zipfile.ZipFile(staging, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, payload)
    content_sha256 = hashlib.sha256(staging.read_bytes()).hexdigest()
    final = root / f"{content_sha256}.zip"
    staging.replace(final)
    metadata = {
        "schema_version": RAW_SNAPSHOT_SCHEMA_VERSION,
        "source_name": "blender_open_data",
        "source_url": BLENDER_SNAPSHOT_URL,
        "source_type": "benchmark",
        "retrieved_at": retrieved_at,
        "content_sha256": content_sha256,
        "byte_count": final.stat().st_size,
        "media_type": "application/zip",
        "parser_version": BLENDER_PARSER_VERSION,
        "licence_or_access_note": BLENDER_LICENSE_NOTE,
        "raw_file": final.name,
    }
    final.with_suffix(".zip.metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return final


def _contract() -> BlenderCohortContract:
    return BlenderCohortContract(
        benchmark_version="4.0.0",
        scene="junkshop",
        backend="CPU",
        operating_system="Windows",
        source_unit="samples/minute",
        source_higher_is_better=True,
        source_score_field="samples_per_minute",
        blender_build_hash="build-4",
        benchmark_script="3.1.0",
        scene_checksum="scene-4",
    )


def test_temporal_diff_is_exact_and_retains_only_contract_rows(tmp_path: Path) -> None:
    old_path = _snapshot(
        tmp_path,
        "old",
        [_submission("id-1", [_observation()]), _submission("id-2", [_observation()])],
        retrieved_at="2026-07-22T00:00:00+00:00",
    )
    new_path = _snapshot(
        tmp_path,
        "new",
        [
            _submission("id-1", [_observation()]),
            _submission("id-2", [_observation()]),
            _submission("id-3", [_observation(), _observation(version="5.2.0")]),
        ],
        retrieved_at="2026-07-23T00:00:00+00:00",
    )

    result = diff_blender_snapshots(
        load_blender_raw_snapshot(old_path),
        load_blender_raw_snapshot(new_path),
        contract=_contract(),
    )

    assert result.old_submission_count == 2
    assert result.new_submission_count == 3
    assert result.removed_submission_count == 0
    assert result.novel_submission_ids == ("id-3",)
    assert result.novel_raw_observation_count == 2
    assert result.novel_normalised_observation_count == 2
    assert len(result.retained_cohort_records) == 1
    assert result.retained_cohort_records[0]["source_record_id"] == "id-3:0"
    assert result.cohort_mismatch_counts == {"benchmark_version": 1}
    assert result.summary()["new_snapshot_is_strict_superset"] is True
    assert result.summary()["bounded_memory"]["old_submission_index"] == "sqlite_on_disk"  # type: ignore[index]


def test_temporal_diff_reports_removed_submission_ids(tmp_path: Path) -> None:
    old_path = _snapshot(
        tmp_path,
        "old",
        [_submission("id-1", [_observation()]), _submission("id-2", [_observation()])],
        retrieved_at="2026-07-22T00:00:00+00:00",
    )
    new_path = _snapshot(
        tmp_path,
        "new",
        [_submission("id-1", [_observation()]), _submission("id-3", [_observation()])],
        retrieved_at="2026-07-23T00:00:00+00:00",
    )

    result = diff_blender_snapshots(
        load_blender_raw_snapshot(old_path),
        load_blender_raw_snapshot(new_path),
        contract=_contract(),
    )

    assert result.removed_submission_count == 1
    assert result.summary()["new_snapshot_is_strict_superset"] is False


def test_temporal_diff_enforces_novel_submission_bound(tmp_path: Path) -> None:
    old_path = _snapshot(
        tmp_path,
        "old",
        [_submission("id-1", [_observation()])],
        retrieved_at="2026-07-22T00:00:00+00:00",
    )
    new_path = _snapshot(
        tmp_path,
        "new",
        [
            _submission("id-1", [_observation()]),
            _submission("id-2", [_observation()]),
            _submission("id-3", [_observation()]),
        ],
        retrieved_at="2026-07-23T00:00:00+00:00",
    )

    with pytest.raises(MemoryError, match="novel submission count"):
        diff_blender_snapshots(
            load_blender_raw_snapshot(old_path),
            load_blender_raw_snapshot(new_path),
            contract=_contract(),
            max_novel_submissions=1,
        )


def test_snapshot_loader_rejects_tampered_archive(tmp_path: Path) -> None:
    path = _snapshot(
        tmp_path,
        "old",
        [_submission("id-1", [_observation()])],
        retrieved_at="2026-07-22T00:00:00+00:00",
    )
    path.write_bytes(path.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="byte count"):
        load_blender_raw_snapshot(path)
