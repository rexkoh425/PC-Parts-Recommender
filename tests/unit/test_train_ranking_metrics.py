from __future__ import annotations

from collections.abc import Sequence

import pytest
from training.train_ranking import _mean_ndcg

from pc_build_recommender.ranking import (
    LabeledRankingQuery,
    RankingCandidate,
    RankingContext,
)


class _FixedScoreRanker:
    def __init__(self, scores: tuple[float, ...]) -> None:
        self._scores = scores

    def predict(
        self,
        _context: RankingContext,
        _candidates: Sequence[RankingCandidate],
    ) -> tuple[float, ...]:
        return self._scores


def test_training_report_ndcg_matches_exponential_gain_evaluator() -> None:
    query = LabeledRankingQuery.create(
        RankingContext(query_id="q1"),
        (
            RankingCandidate(
                product_id="best",
                category="gpu",
                retrieval_scores={"bm25_score": 3.0},
            ),
            RankingCandidate(
                product_id="strong",
                category="gpu",
                retrieval_scores={"bm25_score": 2.0},
            ),
            RankingCandidate(
                product_id="irrelevant",
                category="gpu",
                retrieval_scores={"bm25_score": 1.0},
            ),
        ),
        (4, 3, 0),
    )

    learned, bm25 = _mean_ndcg(_FixedScoreRanker((2.0, 3.0, 1.0)), (query,))

    expected = (7.0 + 15.0 / 1.584962500721156) / (
        15.0 + 7.0 / 1.584962500721156
    )
    assert learned == pytest.approx(expected)
    assert learned < 0.9
    assert bm25 == 1.0


def test_training_report_breaks_score_ties_by_product_id() -> None:
    query = LabeledRankingQuery.create(
        RankingContext(query_id="q-tie"),
        (
            RankingCandidate(product_id="z-best", category="gpu"),
            RankingCandidate(product_id="a-strong", category="gpu"),
            RankingCandidate(product_id="m-irrelevant", category="gpu"),
        ),
        (4, 3, 0),
    )

    learned, bm25 = _mean_ndcg(_FixedScoreRanker((0.0, 0.0, 0.0)), (query,))

    expected = (7.0 + 15.0 / 2.0) / (
        15.0 + 7.0 / 1.584962500721156
    )
    assert learned == pytest.approx(expected)
    assert bm25 == pytest.approx(expected)
