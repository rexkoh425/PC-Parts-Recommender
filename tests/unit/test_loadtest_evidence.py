from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from scripts.loadtest.evidence import (
    LOAD_EVIDENCE_SCHEMA_VERSION,
    LoadEvidenceError,
    build_load_evidence,
    read_endpoint_metrics,
    write_load_evidence,
)
from scripts.loadtest.profile import load_profile

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_STATS_FIELDS = [
    "Type",
    "Name",
    "Request Count",
    "Failure Count",
    "Min Response Time",
    "Max Response Time",
    "Requests/s",
    "Failures/s",
    "50%",
    "95%",
    "99%",
]


def _profile():
    return load_profile(REPOSITORY_ROOT / "scripts" / "loadtest" / "development-profile.json")


def _write_locust_outputs(root: Path, *, include_build: bool = True) -> Path:
    prefix = root / "locust"
    rows = [
        {
            "Type": "POST",
            "Name": "/v1/products/search",
            "Request Count": "40",
            "Failure Count": "0",
            "Min Response Time": "1",
            "Max Response Time": "20",
            "Requests/s": "4.0",
            "Failures/s": "0.0",
            "50%": "3",
            "95%": "12",
            "99%": "18",
        },
    ]
    if include_build:
        rows.append(
            {
                "Type": "POST",
                "Name": "/v1/builds/generate",
                "Request Count": "10",
                "Failure Count": "1",
                "Min Response Time": "100",
                "Max Response Time": "450",
                "Requests/s": "1.0",
                "Failures/s": "0.1",
                "50%": "150",
                "95%": "300",
                "99%": "420",
            }
        )
    with (root / "locust_stats.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_STATS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (root / "locust_stats_history.csv").write_text("Timestamp\n", encoding="utf-8")
    (root / "locust_failures.csv").write_text("Method,Name,Error,Occurrences\n", encoding="utf-8")
    (root / "locust_exceptions.csv").write_text("Count,Message,Traceback,Nodes\n", encoding="utf-8")
    return prefix


def _api_metadata() -> dict[str, object]:
    return {
        "target_origin": "http://127.0.0.1:8000",
        "ready": {
            "data_version": "demo-seed-v1",
            "ranking_model": "deterministic-baseline-v1",
            "rule_version": "compat_v2",
            "solver_version": "in-memory-baseline-v1",
        },
        "freshness": {
            "data_version": "demo-seed-v1",
            "product_count": 23,
            "listing_count": 23,
            "production_ready": False,
            "release_artifact_verification": "development_unverified",
        },
    }


def _host_metadata() -> dict[str, object]:
    return {
        "operating_system": {"system": "Windows", "release": "test", "machine": "x86_64"},
        "cpu_logical_count": 8,
        "total_memory_bytes": 32 * 1024**3,
        "container_memory_limit_bytes": None,
    }


def test_load_evidence_binds_profile_metrics_metadata_and_all_raw_outputs(tmp_path: Path) -> None:
    prefix = _write_locust_outputs(tmp_path)

    evidence = build_load_evidence(
        profile=_profile(),
        csv_prefix=prefix,
        api_metadata=_api_metadata(),
        host_metadata=_host_metadata(),
        users=2,
        spawn_rate_per_second=1.0,
        run_time_seconds=30.0,
        warmup_seconds=0.0,
        cache_state="cold",
        database_state="in_memory_demo",
        created_at=datetime(2026, 7, 23, tzinfo=UTC),
    )

    assert evidence["schema_version"] == LOAD_EVIDENCE_SCHEMA_VERSION
    assert evidence["content_sha256"]
    metrics = evidence["endpoint_metrics"]
    assert isinstance(metrics, dict)
    assert metrics["search"]["p95_response_time_ms"] == 12.0
    assert metrics["build"]["failure_count"] == 1
    assessment = evidence["threshold_assessment"]
    assert isinstance(assessment, dict)
    assert assessment["search_p95_within_target"] is True
    assert assessment["build_p95_within_target"] is True
    assert assessment["claim_status"] == "development_only_not_a_production_latency_claim"
    raw_outputs = evidence["raw_locust_outputs"]
    assert isinstance(raw_outputs, dict)
    assert set(raw_outputs) == {
        "locust_stats.csv",
        "locust_stats_history.csv",
        "locust_failures.csv",
        "locust_exceptions.csv",
    }

    output = tmp_path / "summary.json"
    assert write_load_evidence(evidence=evidence, output_path=output) == output
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted == evidence
    with pytest.raises(LoadEvidenceError, match="will not be overwritten"):
        write_load_evidence(evidence=evidence, output_path=output)


def test_load_evidence_rejects_missing_reviewed_endpoint_metrics(tmp_path: Path) -> None:
    prefix = _write_locust_outputs(tmp_path, include_build=False)

    with pytest.raises(LoadEvidenceError, match="lacks profile endpoint rows: build"):
        read_endpoint_metrics(profile=_profile(), stats_csv_path=prefix.parent / "locust_stats.csv")


def test_load_evidence_rejects_api_metadata_with_unbound_data_versions(tmp_path: Path) -> None:
    prefix = _write_locust_outputs(tmp_path)
    api_metadata = _api_metadata()
    freshness = api_metadata["freshness"]
    assert isinstance(freshness, dict)
    freshness["data_version"] = "different-data-version"

    with pytest.raises(LoadEvidenceError, match="disagree on data_version"):
        build_load_evidence(
            profile=_profile(),
            csv_prefix=prefix,
            api_metadata=api_metadata,
            host_metadata=_host_metadata(),
            users=2,
            spawn_rate_per_second=1.0,
            run_time_seconds=30.0,
            warmup_seconds=0.0,
            cache_state="cold",
            database_state="in_memory_demo",
        )


def test_load_evidence_rejects_tampered_content_digest(tmp_path: Path) -> None:
    prefix = _write_locust_outputs(tmp_path)
    evidence = build_load_evidence(
        profile=_profile(),
        csv_prefix=prefix,
        api_metadata=_api_metadata(),
        host_metadata=_host_metadata(),
        users=2,
        spawn_rate_per_second=1.0,
        run_time_seconds=30.0,
        warmup_seconds=0.0,
        cache_state="cold",
        database_state="in_memory_demo",
    )
    evidence["content_sha256"] = "0" * 64

    with pytest.raises(LoadEvidenceError, match="does not match"):
        write_load_evidence(evidence=evidence, output_path=tmp_path / "summary.json")
