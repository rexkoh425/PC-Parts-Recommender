from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from pc_build_recommender.evaluation.manifest import json_sha256
from pc_build_recommender.ranking import (
    ProductRanker,
    RankerArtifactIdentity,
    RankerMetadata,
    RankerPromotionPolicy,
    evaluate_ranker_promotion,
)
from pc_build_recommender.retrieval import (
    ArtifactBoundRankingEvidence,
    FrozenCandidateQuery,
    PinnedCandidateSet,
    QueryGroupSplit,
    RelevanceLabelSource,
    compare_ranked_models,
    load_diagnostic_ranking_artifact,
    load_ranking_comparison_report,
    write_ranking_comparison_report,
)


def _human_dataset(
    *, source: RelevanceLabelSource = RelevanceLabelSource.HUMAN
) -> PinnedCandidateSet:
    queries = []
    for group_number in range(3):
        for paraphrase in range(2):
            query_id = f"q-{group_number}-{paraphrase}"
            queries.append(
                FrozenCandidateQuery(
                    query_id=query_id,
                    query_group_id=f"intent-{group_number}",
                    query_text=f"query {group_number} paraphrase {paraphrase}",
                    category="gpu",
                    candidate_ids=("best", "ok", "bad"),
                    relevance_labels={"best": 4, "ok": 2, "bad": 0},
                )
            )
    return PinnedCandidateSet.create(
        "human-ranking-v1",
        queries,
        label_source=source,
        adjudication_complete=source is RelevanceLabelSource.HUMAN,
        judgment_manifest_sha256="a" * 64 if source is RelevanceLabelSource.HUMAN else None,
    )


def _rankings(dataset: PinnedCandidateSet) -> dict[str, dict[str, list[str]]]:
    return {
        "bm25": {query.query_id: ["bad", "ok", "best"] for query in dataset.queries},
        "vector": {query.query_id: ["ok", "best", "bad"] for query in dataset.queries},
        "rrf_hybrid": {query.query_id: ["best", "ok", "bad"] for query in dataset.queries},
        "lambdamart": {query.query_id: ["best", "ok", "bad"] for query in dataset.queries},
    }


def _bound_ranker_evidence(
    dataset: PinnedCandidateSet,
    split: QueryGroupSplit,
    metadata: RankerMetadata,
    identity: RankerArtifactIdentity,
) -> ArtifactBoundRankingEvidence:
    evaluated = split.subset(dataset, "test")
    ranking = {
        query.query_id: _rankings(dataset)["lambdamart"][query.query_id]
        for query in evaluated.queries
    }
    return ArtifactBoundRankingEvidence.create(
        model_name="lambdamart",
        ranker_version=metadata.ranker_version,
        model_sha256=identity.model_sha256,
        metadata_sha256=identity.metadata_sha256,
        manifest_sha256=identity.manifest_sha256,
        ranker_metadata_payload=metadata.to_dict(),
        feature_version=metadata.feature_version,
        feature_names=metadata.feature_names,
        candidate_snapshot_sha256=evaluated.checksum,
        feature_snapshot_sha256="e" * 64,
        score_snapshot_sha256="f" * 64,
        ranking_sha256=json_sha256(ranking),
        split_name="test",
        split_checksum=split.checksum,
        query_count=len(evaluated.queries),
        row_count=sum(len(query.candidate_ids) for query in evaluated.queries),
    )


def _verified_ranker(
    metadata: RankerMetadata,
    identity: RankerArtifactIdentity,
) -> ProductRanker:
    return cast(
        ProductRanker,
        SimpleNamespace(
            metadata=metadata,
            artifact_identity=identity,
            verified_artifact_loaded=True,
        ),
    )


def test_frozen_query_group_split_keeps_paraphrases_together(tmp_path: Path) -> None:
    dataset = _human_dataset()
    split = QueryGroupSplit.create(dataset, version="split-v1", seed=11)

    for group_number in range(3):
        assert split.assignments[f"q-{group_number}-0"] == split.assignments[f"q-{group_number}-1"]

    output = split.save(tmp_path / "split.json")
    assert QueryGroupSplit.load(output) == split
    assert split.subset(dataset, "test").eligible_for_promotion


def test_paired_report_is_reproducible_and_can_pass_reduced_test_gate(
    tmp_path: Path,
) -> None:
    dataset = _human_dataset()
    split = QueryGroupSplit.create(dataset, version="split-v1", seed=11)
    metadata = RankerMetadata(
        ranker_version="ltr-human-v1",
        ranking_basis="lightgbm_lambdamart",
        feature_version="ranking-features-v1",
        model_type="LGBMRanker",
        feature_names=("bm25_score",),
        created_at_utc="2026-07-22T00:00:00Z",
        training_label_source="human",
        training_adjudication_complete=True,
        training_judgment_manifest_sha256="a" * 64,
        training_dataset_manifest_sha256="e" * 64,
        training_prelabel_snapshot_sha256="f" * 64,
        training_feature_contract_sha256="1" * 64,
        query_group_split_checksum=split.checksum,
        query_split_membership_verified=True,
        model_sha256="b" * 64,
        promotion_eligible=True,
    )
    artifact_identity = RankerArtifactIdentity(
        model_sha256="b" * 64,
        metadata_sha256="c" * 64,
        manifest_sha256="d" * 64,
    )
    report = compare_ranked_models(
        dataset,
        _rankings(dataset),
        artifact_bound_rankings={
            "lambdamart": _bound_ranker_evidence(
                dataset,
                split,
                metadata,
                artifact_identity,
            )
        },
        query_split=split,
        split_name="test",
        n_resamples=100,
        seed=19,
    )

    assert report.eligible_for_promotion
    assert report.query_group_count == 1
    assert (
        report.model_evaluations["lambdamart"].metric_estimates["ndcg_at_10"].ci_lower is not None
    )
    assert report.paired_comparisons["lambdamart_minus_bm25"]["ndcg_at_10"].ci_lower > 0

    output = write_ranking_comparison_report(report, tmp_path / "comparison.json")
    assert load_ranking_comparison_report(output)["report_sha256"] == report.report_sha256

    decision = evaluate_ranker_promotion(
        report,
        challenger_model="lambdamart",
        ranker=_verified_ranker(metadata, artifact_identity),
        policy=RankerPromotionPolicy(minimum_test_query_groups=1),
    )
    assert decision.passed


@pytest.mark.parametrize(
    "digest_field",
    ["model_sha256", "metadata_sha256", "manifest_sha256"],
)
def test_promotion_rejects_same_version_swapped_ranker_artifact(digest_field: str) -> None:
    dataset = _human_dataset()
    split = QueryGroupSplit.create(dataset, version="swap-split-v1", seed=11)
    metadata = RankerMetadata(
        ranker_version="ltr-same-version-v1",
        ranking_basis="lightgbm_lambdamart",
        feature_version="ranking-features-v1",
        model_type="LGBMRanker",
        feature_names=("bm25_score",),
        created_at_utc="2026-07-23T00:00:00Z",
        training_label_source="human",
        training_adjudication_complete=True,
        training_judgment_manifest_sha256="a" * 64,
        training_dataset_manifest_sha256="e" * 64,
        training_prelabel_snapshot_sha256="f" * 64,
        training_feature_contract_sha256="1" * 64,
        query_group_split_checksum=split.checksum,
        query_split_membership_verified=True,
        model_sha256="b" * 64,
        promotion_eligible=True,
    )
    evaluated_identity = RankerArtifactIdentity(
        model_sha256="b" * 64,
        metadata_sha256="c" * 64,
        manifest_sha256="d" * 64,
    )
    report = compare_ranked_models(
        dataset,
        _rankings(dataset),
        artifact_bound_rankings={
            "lambdamart": _bound_ranker_evidence(
                dataset,
                split,
                metadata,
                evaluated_identity,
            )
        },
        query_split=split,
        split_name="test",
        n_resamples=50,
    )
    loaded_identity = replace(evaluated_identity, **{digest_field: "9" * 64})
    loaded_metadata = (
        replace(metadata, model_sha256=loaded_identity.model_sha256)
        if digest_field == "model_sha256"
        else metadata
    )

    decision = evaluate_ranker_promotion(
        report,
        challenger_model="lambdamart",
        ranker=_verified_ranker(loaded_metadata, loaded_identity),
        policy=RankerPromotionPolicy(minimum_test_query_groups=1),
    )

    assert not decision.passed
    assert "evaluated ranking is bound to different artifact bytes" in decision.failures


def test_promotion_rejects_unbound_challenger_rankings() -> None:
    dataset = _human_dataset()
    split = QueryGroupSplit.create(dataset, version="unbound-split-v1", seed=11)
    metadata = RankerMetadata(
        ranker_version="ltr-unbound-v1",
        ranking_basis="lightgbm_lambdamart",
        feature_version="ranking-features-v1",
        model_type="LGBMRanker",
        feature_names=("bm25_score",),
        created_at_utc="2026-07-23T00:00:00Z",
        training_label_source="human",
        training_adjudication_complete=True,
        training_judgment_manifest_sha256="a" * 64,
        training_dataset_manifest_sha256="e" * 64,
        training_prelabel_snapshot_sha256="f" * 64,
        training_feature_contract_sha256="1" * 64,
        query_group_split_checksum=split.checksum,
        query_split_membership_verified=True,
        model_sha256="b" * 64,
        promotion_eligible=True,
    )
    identity = RankerArtifactIdentity(
        model_sha256="b" * 64,
        metadata_sha256="c" * 64,
        manifest_sha256="d" * 64,
    )
    report = compare_ranked_models(
        dataset,
        _rankings(dataset),
        query_split=split,
        split_name="test",
        n_resamples=50,
    )

    decision = evaluate_ranker_promotion(
        report,
        challenger_model="lambdamart",
        ranker=_verified_ranker(metadata, identity),
        policy=RankerPromotionPolicy(minimum_test_query_groups=1),
    )

    assert not decision.passed
    assert "challenger ranking is not bound to a verified model artifact" in decision.failures


def test_silver_labels_are_non_promotable_even_with_strong_diagnostic_scores() -> None:
    dataset = _human_dataset(source=RelevanceLabelSource.SILVER)
    split = QueryGroupSplit.create(dataset, version="silver-split-v1", seed=11)
    report = compare_ranked_models(
        dataset,
        _rankings(dataset),
        query_split=split,
        split_name="test",
        n_resamples=50,
    )

    decision = evaluate_ranker_promotion(
        report,
        challenger_model="lambdamart",
        ranker=None,
        policy=RankerPromotionPolicy(minimum_test_query_groups=1),
    )
    assert not report.eligible_for_promotion
    assert not decision.passed
    assert any("silver" in reason for reason in decision.failures)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_ranker_promotion_policy_rejects_non_finite_thresholds(value: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        RankerPromotionPolicy(minimum_recall_at_50=value)


def test_ranker_promotion_rejects_non_finite_report_metrics() -> None:
    dataset = _human_dataset()
    split = QueryGroupSplit.create(dataset, version="non-finite-split-v1", seed=11)
    report = compare_ranked_models(
        dataset,
        _rankings(dataset),
        query_split=split,
        split_name="test",
        n_resamples=50,
    )
    comparison = report.paired_comparisons["lambdamart_minus_bm25"]["ndcg_at_10"]
    report.paired_comparisons["lambdamart_minus_bm25"]["ndcg_at_10"] = replace(
        comparison,
        ci_lower=float("nan"),
    )

    with pytest.raises(ValueError, match="non-finite evidence"):
        evaluate_ranker_promotion(
            report,
            challenger_model="lambdamart",
            ranker=None,
            policy=RankerPromotionPolicy(minimum_test_query_groups=1),
        )


def test_existing_silver_report_shape_is_consumable_but_never_promotable(
    tmp_path: Path,
) -> None:
    payload = {
        "schema_version": "pc-build-recommender.retrieval-silver-pilot.v1",
        "eligible_for_production_or_resume_metric_claims": False,
        "human_relevance_judgments": False,
        "training_performed": False,
        "reporting_block_reason": "silver labels are diagnostic only",
        "dataset": {
            "version": "silver-full-v2",
            "candidate_checksum": "c" * 64,
            "query_count": 32,
        },
        "models": {
            "bm25": {"metrics": {"aggregate": {"ndcg_at_10": 0.1934}}},
            "vector_only": {"metrics": {"aggregate": {"ndcg_at_10": 0.2723}}},
            "rrf_hybrid": {"metrics": {"aggregate": {"ndcg_at_10": 0.3085}}},
        },
        "paired_baseline_comparisons": {
            "rrf_hybrid_minus_bm25": {"ndcg_at_10": {"relative_delta_percent": 59.52}}
        },
    }
    payload["artifact_sha256"] = json_sha256(payload)
    path = tmp_path / "silver-metrics.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    diagnostic = load_diagnostic_ranking_artifact(path)

    assert diagnostic.query_count == 32
    assert diagnostic.aggregate_model_metrics["rrf_hybrid"]["ndcg_at_10"] == 0.3085
    assert not diagnostic.eligible_for_promotion
