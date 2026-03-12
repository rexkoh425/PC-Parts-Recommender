from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from pc_build_recommender.evaluation import (
    SyntheticDataError,
    build_evaluation_artifact,
    evaluate_entity_resolution,
    load_evaluation_artifact,
    verify_evaluation_artifact,
    write_evaluation_artifact,
)


def test_artifact_serialises_metrics_counts_intervals_and_synthetic_policy(
    tmp_path: Path,
) -> None:
    result = evaluate_entity_resolution(
        labels=[1, 0, 1, 0],
        match_scores=[0.99, 0.1, 0.8, 0.2],
        threshold=0.8,
        is_synthetic=[False, False, False, False],
        n_resamples=50,
    )
    artifact = build_evaluation_artifact(
        task="entity_resolution",
        run_id="pilot-001",
        dataset_manifest_sha256="a" * 64,
        result=result,
        metadata={"git_sha": "deadbeef", "seed": 20260722},
        created_at=datetime(2026, 7, 22, tzinfo=UTC),
    )

    assert artifact["eligible_for_reported_metrics"] is True
    assert verify_evaluation_artifact(artifact)
    metrics = artifact["metrics"]
    assert isinstance(metrics, list)
    assert metrics[0]["sample_count"] == 2
    assert metrics[0]["confidence_interval"] is not None

    output = write_evaluation_artifact(artifact, tmp_path / "metrics.json")
    assert load_evaluation_artifact(output) == artifact

    artifact["run_id"] = "tampered"
    assert not verify_evaluation_artifact(artifact)


def test_included_synthetic_rows_are_explicitly_not_reportable() -> None:
    result = evaluate_entity_resolution(
        labels=[1, 0],
        match_scores=[0.9, 0.9],
        threshold=0.8,
        is_synthetic=[False, True],
        include_synthetic=True,
        n_resamples=20,
    )

    with pytest.raises(SyntheticDataError, match="synthetic rows were included"):
        result.data_use.require_reportable()

    artifact = build_evaluation_artifact(
        task="entity_resolution_smoke",
        run_id="synthetic-smoke",
        dataset_manifest_sha256="b" * 64,
        result=result,
    )
    assert artifact["eligible_for_reported_metrics"] is False
    assert artifact["reporting_block_reason"] == (
        "synthetic rows were included in evaluation metrics"
    )
