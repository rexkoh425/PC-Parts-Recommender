from __future__ import annotations

import pytest

from pc_build_recommender.ranking import (
    evaluate_frozen_rankings,
    rankings_from_scores,
    relative_ndcg_improvement,
)
from pc_build_recommender.retrieval import FrozenCandidateQuery, PinnedCandidateSet


def _dataset() -> PinnedCandidateSet:
    return PinnedCandidateSet.create(
        "ranking-frozen-v1",
        [
            FrozenCandidateQuery(
                query_id="q1",
                candidate_ids=("best", "ok", "bad"),
                relevance_labels={"best": 4, "ok": 2, "bad": 0},
            )
        ],
    )


def test_frozen_ranking_evaluation_compares_same_candidate_set() -> None:
    dataset = _dataset()
    evaluation = evaluate_frozen_rankings(
        dataset,
        {"q1": ["best", "ok", "bad"]},
        baseline_ranked_product_ids={"q1": ["bad", "ok", "best"]},
        ranker_version="ltr-v1",
        ranking_basis="lightgbm_lambdamart",
    )

    assert evaluation.ndcg_at_10 == 1.0
    assert evaluation.baseline_ndcg_at_10 is not None
    assert evaluation.relative_improvement_percent is not None
    assert evaluation.relative_improvement_percent > 0
    assert evaluation.candidate_checksum == dataset.checksum


def test_rankings_from_scores_is_stable_on_ties() -> None:
    ranking = rankings_from_scores(
        _dataset(),
        {"q1": {"best": 1.0, "ok": 1.0, "bad": 0.0}},
    )
    assert ranking == {"q1": ["best", "ok", "bad"]}


def test_ranking_evaluation_rejects_missing_candidate() -> None:
    with pytest.raises(ValueError, match="does not match the frozen candidate set"):
        evaluate_frozen_rankings(_dataset(), {"q1": ["best", "ok"]})


def test_relative_ndcg_claim_formula() -> None:
    assert relative_ndcg_improvement(0.70, 0.60) == pytest.approx(16.6666667)
    with pytest.raises(ValueError, match="positive"):
        relative_ndcg_improvement(0.5, 0.0)
