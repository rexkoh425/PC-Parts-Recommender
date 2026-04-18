from __future__ import annotations

import json
from pathlib import Path

import pytest

from pc_build_recommender.compatibility.evaluation import (
    GENERATED_SCENARIO_LABEL,
    SCENARIO_KINDS,
    EvaluationConfig,
    load_evaluation_report,
    run_generated_evaluation,
    write_evaluation_report,
)


def test_generated_evaluation_is_deterministic_and_explicitly_non_market() -> None:
    config = EvaluationConfig(scenario_count=100, seed=77)

    first = run_generated_evaluation(config)
    second = run_generated_evaluation(config)

    assert first.to_dict() == second.to_dict()
    assert first.artifact_sha256 == second.artifact_sha256
    assert first.payload["evaluation_passed"] is True
    assert first.payload["scenario_provenance"] == GENERATED_SCENARIO_LABEL
    assert first.payload["market_builds_evaluated"] == 0
    assert first.payload["scenario_count"] == 100
    assert first.payload["scenario_counts"] == {kind: 10 for kind in SCENARIO_KINDS}
    assert first.payload["oracle_mismatch_count"] == 0
    assert first.payload["memory_strategy"]["retained_scenario_records"] == 0
    assert first.payload["assertions"]["failed"] == 0


def test_different_seed_changes_stream_and_content_address() -> None:
    first = run_generated_evaluation(EvaluationConfig(scenario_count=20, seed=1))
    second = run_generated_evaluation(EvaluationConfig(scenario_count=20, seed=2))

    assert first.payload["scenario_stream_sha256"] != second.payload["scenario_stream_sha256"]
    assert first.artifact_sha256 != second.artifact_sha256


def test_report_write_is_content_addressed_idempotent_and_verified(tmp_path: Path) -> None:
    report = run_generated_evaluation(EvaluationConfig(scenario_count=20, seed=9))

    first = write_evaluation_report(report, tmp_path)
    second = write_evaluation_report(report, tmp_path)
    restored = load_evaluation_report(first)

    assert first == second
    assert report.artifact_sha256[:16] in first.name
    assert restored.to_dict() == report.to_dict()


def test_tampered_report_is_rejected(tmp_path: Path) -> None:
    report = run_generated_evaluation(EvaluationConfig(scenario_count=10, seed=4))
    path = write_evaluation_report(report, tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["scenario_count"] = 11
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="artifact_sha256"):
        load_evaluation_report(path)


def test_evaluation_accepts_small_balanced_runs() -> None:
    config = EvaluationConfig(scenario_count=10)
    report = run_generated_evaluation(config)

    assert report.payload["scenario_count"] == 10


def test_evaluation_rejects_invalid_scope() -> None:
    with pytest.raises(ValueError, match="at least"):
        EvaluationConfig(scenario_count=0)
    with pytest.raises(ValueError, match="compat_v2"):
        EvaluationConfig(rule_version="compat_v1")
