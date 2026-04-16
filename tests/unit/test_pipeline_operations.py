from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from pc_build_recommender.pipeline_operations import (
    PIPELINE_OPERATION_EVENT_SCHEMA_VERSION,
    record_pipeline_operation,
    summarize_pipeline_operations,
    write_pipeline_operation_event,
)


def test_operation_receipts_are_aggregate_safe_and_summarized_within_window(tmp_path) -> None:
    now = datetime(2030, 1, 15, 12, tzinfo=UTC)
    write_pipeline_operation_event(
        tmp_path,
        operation_name="catalog_ingestion",
        status="succeeded",
        finished_at=now - timedelta(minutes=5),
    )
    path = write_pipeline_operation_event(
        tmp_path,
        operation_name="catalog_ingestion",
        status="failed",
        finished_at=now - timedelta(minutes=1),
        failure_class="builtins.ValueError",
    )
    write_pipeline_operation_event(
        tmp_path,
        operation_name="catalog_ingestion",
        status="failed",
        finished_at=now - timedelta(days=8),
        failure_class="builtins.RuntimeError",
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == PIPELINE_OPERATION_EVENT_SCHEMA_VERSION
    assert set(payload) == {
        "schema_version",
        "event_id",
        "operation_name",
        "status",
        "finished_at",
        "failure_class",
    }
    summary = summarize_pipeline_operations(tmp_path, now=now)

    assert summary is not None
    assert summary.available is True
    assert summary.event_count == 2
    assert summary.succeeded_count == 1
    assert summary.failed_count == 1
    assert summary.latest_failure_at == now - timedelta(minutes=1)
    assert summary.invalid_receipt_count == 0


def test_operation_context_records_failure_class_without_exception_message(tmp_path) -> None:
    with (
        pytest.raises(ValueError, match="secret source URL"),
        record_pipeline_operation(tmp_path, "governed_web_retention"),
    ):
        raise ValueError("secret source URL https://retailer.example/private")

    receipts = list((tmp_path / "events").glob("*.json"))
    assert len(receipts) == 1
    serialized = receipts[0].read_text(encoding="utf-8")
    assert "secret source URL" not in serialized
    assert "retailer.example" not in serialized
    assert "builtins.ValueError" in serialized


def test_summary_is_explicitly_unavailable_for_missing_or_invalid_mount(tmp_path) -> None:
    assert summarize_pipeline_operations(None) is None
    missing = summarize_pipeline_operations(tmp_path / "missing")
    assert missing is not None
    assert missing.available is False

    root_file = tmp_path / "not-a-directory"
    root_file.write_text("not a directory", encoding="utf-8")
    unavailable = summarize_pipeline_operations(root_file)
    assert unavailable is not None
    assert unavailable.available is False


def test_summary_counts_invalid_receipts_without_interpreting_them_as_success(tmp_path) -> None:
    events = tmp_path / "events"
    events.mkdir()
    (events / "invalid.json").write_text('{"secret":"do not parse"}', encoding="utf-8")
    now = datetime(2030, 1, 15, 12, tzinfo=UTC)
    write_pipeline_operation_event(
        tmp_path,
        operation_name="benchmark_ingestion",
        status="succeeded",
        finished_at=now,
    )

    summary = summarize_pipeline_operations(tmp_path, now=now)

    assert summary is not None
    assert summary.available is True
    assert summary.event_count == 1
    assert summary.succeeded_count == 1
    assert summary.failed_count == 0
    assert summary.invalid_receipt_count == 1
