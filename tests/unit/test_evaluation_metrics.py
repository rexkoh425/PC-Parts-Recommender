from __future__ import annotations

import pytest

from pc_build_recommender.evaluation import (
    evaluate_entity_resolution,
    evaluate_ranker_lift,
    evaluate_regression,
    evaluate_retrieval,
    wilson_confidence_interval,
)


def test_entity_metrics_exclude_synthetic_rows_and_preserve_counts() -> None:
    result = evaluate_entity_resolution(
        labels=[1, 0, 1, 0, 1],
        match_scores=[0.99, 0.95, 0.90, 0.10, 0.0],
        threshold=0.80,
        is_synthetic=[False, True, False, False, False],
        groups=["a", "b", "c", "d", "e"],
        n_resamples=100,
    )

    assert result.metric("entity.precision").value == 1.0
    assert result.metric("entity.precision").sample_count == 2
    assert result.metric("entity.recall").value == pytest.approx(2 / 3)
    assert result.metric("entity.f1").value == pytest.approx(0.8)
    assert result.data_use.synthetic_rows == 1
    assert result.data_use.evaluated_rows == 4
    assert result.data_use.eligible_for_reported_metrics


def test_entity_metrics_including_synthetic_rows_blocks_reporting() -> None:
    result = evaluate_entity_resolution(
        labels=[1, 0, 1],
        match_scores=[0.9, 0.9, 0.8],
        threshold=0.8,
        is_synthetic=[False, True, False],
        include_synthetic=True,
        n_resamples=50,
    )

    assert result.metric("entity.precision").value == pytest.approx(2 / 3)
    assert not result.data_use.eligible_for_reported_metrics
    assert result.data_use.reporting_block_reason is not None


def test_regression_metrics_are_group_bootstrapped() -> None:
    result = evaluate_regression(
        actual=[1.0, 2.0, 3.0, 100.0],
        predicted=[1.0, 2.0, 3.0, 0.0],
        is_synthetic=[False, False, False, True],
        groups=["family-a", "family-b", "family-c", "synthetic"],
        n_resamples=100,
    )

    assert result.metric("regression.mae").value == 0.0
    assert result.metric("regression.rmse").value == 0.0
    assert result.metric("regression.r_squared").value == 1.0
    assert result.metric("regression.spearman").value == 1.0
    assert result.metric("regression.mape").value == 0.0
    assert result.metadata["bootstrap_unit"] == "group"


def test_retrieval_metrics_use_query_level_evidence() -> None:
    result = evaluate_retrieval(
        query_ids=["q1", "q1", "q1", "q2", "q2", "q2"],
        relevance=[4, 0, 2, 0, 3, 4],
        scores=[0.9, 0.1, 0.8, 0.1, 0.8, 10.0],
        is_synthetic=[False, False, False, False, False, True],
        recall_cutoffs=(1, 2),
        ndcg_cutoff=2,
        reciprocal_rank_cutoff=2,
        n_resamples=100,
    )

    assert result.metric("retrieval.ndcg_at_2").value == 1.0
    assert result.metric("retrieval.recall_at_2").value == 1.0
    assert result.metric("retrieval.mrr_at_2").value == 1.0
    assert result.metric("retrieval.ndcg_at_2").sample_count == 2
    assert result.metadata["recall_scope"] == "judged_pool"


def test_ranker_lift_is_paired_on_identical_queries() -> None:
    result = evaluate_ranker_lift(
        query_ids=["q1", "q1", "q2", "q2"],
        relevance=[3, 0, 3, 0],
        baseline_scores=[0.0, 1.0, 0.0, 1.0],
        candidate_scores=[1.0, 0.0, 1.0, 0.0],
        is_synthetic=[False, False, False, False],
        ndcg_cutoff=2,
        n_resamples=100,
    )

    baseline = result.metric("ranker.baseline_ndcg_at_2").value
    candidate = result.metric("ranker.candidate_ndcg_at_2").value
    relative = result.metric("ranker.relative_ndcg_at_2_lift_percent").value
    assert baseline == pytest.approx(1 / 1.584962500721156)
    assert candidate == 1.0
    assert relative == pytest.approx((candidate - baseline) / baseline * 100)
    assert result.metric("ranker.query_win_rate").value == 1.0


def test_wilson_interval_handles_empty_denominator_explicitly() -> None:
    assert wilson_confidence_interval(0, 0) is None
    lower, upper = wilson_confidence_interval(99, 100) or (0.0, 0.0)
    assert lower < 0.99 < upper
