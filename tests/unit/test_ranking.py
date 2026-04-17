from __future__ import annotations

import pytest

from pc_build_recommender.ranking import (
    HeuristicRanker,
    RankingCandidate,
    RankingContext,
    RankingFeatureBuilder,
)


def _candidate(
    product_id: str,
    *,
    price: float,
    workload: float,
    vector: float,
    reliability: float,
) -> RankingCandidate:
    return RankingCandidate(
        product_id=product_id,
        category="gpu",
        price_sgd=price,
        brand="Example",
        retrieval_scores={
            "bm25_score": vector * 10,
            "vector_similarity": vector,
            "rrf_score": vector / 30,
        },
        workload_scores={"local_ai": workload},
        signals={
            "availability_score": 1,
            "reliability_score": reliability,
            "predicted_workload_score": workload,
            "median_price_30d_sgd": price * 1.1,
        },
        attributes={"vram_gb": 16, "model": product_id, "warranty_years": 5},
    )


def test_hard_compatibility_cannot_be_a_ranking_signal() -> None:
    with pytest.raises(ValueError, match="compatibility must be filtered"):
        RankingCandidate(
            product_id="bad",
            category="gpu",
            signals={"hard_compatible": 1.0},
        )


def test_feature_builder_records_observed_vs_predicted_basis() -> None:
    candidate = RankingCandidate(
        product_id="gpu-a",
        category="gpu",
        price_sgd=1000,
        retrieval_scores={"bm25_score": 3, "vector_similarity": 0.8, "rrf_score": 0.03},
        workload_scores={"local_ai": 92, "gaming_1440p": 80},
        signals={
            "observed_benchmark_score": 91,
            "predicted_workload_score": 88,
            "availability_score": 1,
        },
        attributes={"vram_gb": 24},
    )
    context = RankingContext(
        query_id="q1",
        budget_sgd=1500,
        workload_weights={"local_ai": 0.75, "gaming_1440p": 0.25},
        requirements={"minimum_gpu_vram_gb": 16},
    )

    batch = RankingFeatureBuilder().build(context, [candidate])
    features = dict(zip(batch.feature_names, batch.values[0], strict=True))

    assert features["weighted_workload_score"] == pytest.approx(89)
    assert features["observed_benchmark_score"] == 91
    assert features["predicted_workload_score"] == 88
    assert features["has_observed_benchmark"] == 1
    assert features["gpu_memory_fit"] == pytest.approx(1.5)
    assert not any("compat" in name for name in batch.feature_names)


def test_heuristic_ranker_is_deterministic_and_exposes_basis_metadata() -> None:
    candidates = [
        _candidate("value", price=800, workload=85, vector=0.8, reliability=0.9),
        _candidate("expensive", price=1400, workload=86, vector=0.7, reliability=0.8),
        _candidate("weak", price=900, workload=50, vector=0.3, reliability=0.5),
    ]
    context = RankingContext(
        query_id="q-value",
        budget_sgd=1500,
        workload_weights={"local_ai": 1.0},
    )
    ranker = HeuristicRanker()

    first = ranker.rank_query(context, candidates)
    second = ranker.rank_query(context, candidates)

    assert [item.product_id for item in first] == [item.product_id for item in second]
    assert first[0].product_id == "value"
    assert first[0].rank == 1
    assert first[0].feature_contributions
    assert ranker.metadata.ranker_version == "heuristic-v1"
    assert ranker.metadata.ranking_basis == "heuristic_baseline"

