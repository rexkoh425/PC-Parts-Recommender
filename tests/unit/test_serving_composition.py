from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from pc_build_recommender.application import (
    ActiveServingModels,
    EmbeddingReleaseExpectation,
    ServingConfigurationError,
)
from pc_build_recommender.application.serving import _load_passing_ranker_decision
from pc_build_recommender.evaluation.manifest import json_sha256
from pc_build_recommender.ranking import RankerPromotionDecision, RankerPromotionPolicy


def test_active_versions_expose_exact_performance_routes() -> None:
    models = ActiveServingModels(
        catalog_data_version="catalog-v1",
        retrieval_model="postgres-hybrid-v1",
        ranking_model="ltr-v4",
        performance_models={"cpu/compile": "c" * 64, "gpu/local_ai": "d" * 64},
        embedding_index_version="embedding-v3",
        retrieval_report_sha256="a" * 64,
        ranker_promotion_decision_sha256="b" * 64,
        ranker_model_sha256="c" * 64,
        ranker_metadata_sha256="d" * 64,
        ranker_manifest_sha256="e" * 64,
    )

    assert models.performance_model_label == (
        "promoted[cpu/compile=" + "c" * 64 + ",gpu/local_ai=" + "d" * 64 + "]"
    )


def test_embedding_expectation_requires_pinned_revision() -> None:
    with pytest.raises(ValueError, match="encoder_revision"):
        EmbeddingReleaseExpectation(
            data_version="data-v1",
            index_version="index-v1",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            encoder_revision="",
            encoder_fingerprint="a" * 64,
            dataset_content_hash="b" * 64,
        )


def test_ranker_promotion_decision_is_hash_verified(tmp_path: Path) -> None:
    model_sha256 = "b" * 64
    metadata_sha256 = "c" * 64
    manifest_sha256 = "d" * 64
    provisional = RankerPromotionDecision(
        comparison_report_sha256="a" * 64,
        challenger_model="lambdamart",
        passed=True,
        failures=(),
        measured_values={
            "test_query_count": 150,
            "test_query_group_count": 150,
            "label_source": "human",
            "split_name": "test",
            "challenger_recall_at_50": 0.97,
            "relative_ndcg_lift_percent_over_bm25": 18.0,
            "bm25_ndcg_delta_ci_lower": 0.01,
            "rrf_ndcg_delta_ci_lower": 0.0,
            "ranker_training_label_source": "human",
            "ranker_version": "ltr-v4",
            "ranker_model_sha256": model_sha256,
            "ranker_metadata_sha256": metadata_sha256,
            "ranker_manifest_sha256": manifest_sha256,
            "artifact_binding_sha256": "e" * 64,
            "metadata_payload_sha256": "f" * 64,
            "feature_snapshot_sha256": "1" * 64,
            "candidate_snapshot_sha256": "2" * 64,
            "score_snapshot_sha256": "3" * 64,
            "ranking_sha256": "4" * 64,
        },
        policy=RankerPromotionPolicy(),
        ranker_version="ltr-v4",
        ranker_model_sha256=model_sha256,
        ranker_metadata_sha256=metadata_sha256,
        ranker_manifest_sha256=manifest_sha256,
        decision_sha256="",
    )
    decision = replace(
        provisional,
        decision_sha256=json_sha256(provisional.content_payload()),
    )
    payload = decision.to_dict()
    path = tmp_path / "decision.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert _load_passing_ranker_decision(path).ranker_version == "ltr-v4"

    payload["ranker_version"] = "tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ServingConfigurationError, match="decision hash"):
        _load_passing_ranker_decision(path)

    payload = decision.to_dict()
    measured_values = payload["measured_values"]
    assert isinstance(measured_values, dict)
    measured_values["challenger_recall_at_50"] = float("nan")
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ServingConfigurationError, match="non-finite JSON"):
        _load_passing_ranker_decision(path)
