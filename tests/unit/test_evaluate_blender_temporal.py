from __future__ import annotations

import json

import pytest
from training.evaluate_blender_temporal import (
    _external_gate,
    _write_content_addressed_report,
    verify_temporal_report,
)


def test_external_gate_excludes_development_families_and_fails_small_cohort() -> None:
    rows = [
        {"product_family": "cpu:overlap", "target_score": 1.0},
        {"product_family": "cpu:new", "target_score": 2.0},
    ]

    clean, gate, blockers = _external_gate(
        rows,
        development_families={"cpu:overlap"},
        minimum_rows=20,
        minimum_families=10,
        strict_superset=True,
    )

    assert [row["product_family"] for row in clean] == ["cpu:new"]
    assert gate["excluded_development_overlap_row_count"] == 1
    assert gate["remaining_development_overlap_count"] == 0
    assert gate["metric_evaluation_eligible"] is False
    assert len(blockers) == 2


def test_content_addressed_report_detects_payload_tampering(tmp_path) -> None:
    path, digest = _write_content_addressed_report(
        {"schema_version": "fixture.v1", "status": "insufficient"}, tmp_path
    )

    assert path.stem == digest
    assert verify_temporal_report(path)["status"] == "insufficient"
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["report"]["status"] = "metrics_computed"
    path.write_text(json.dumps(envelope), encoding="utf-8")

    with pytest.raises(ValueError, match="digest"):
        verify_temporal_report(path)
