from __future__ import annotations

import json

import pytest

from pc_build_recommender.retrieval import (
    FrozenCandidateQuery,
    PinnedCandidateSet,
    evaluate_ranked_candidates,
    ndcg_at_k,
)


def _dataset() -> PinnedCandidateSet:
    return PinnedCandidateSet.create(
        "retrieval-test-v1",
        [
            FrozenCandidateQuery(
                query_id="q-ai",
                query_text="16 GB local AI GPU",
                category="gpu",
                candidate_ids=("a", "b", "c"),
                relevance_labels={"a": 4, "b": 2, "c": 0},
            ),
            FrozenCandidateQuery(
                query_id="q-dev",
                query_text="compilation CPU",
                category="cpu",
                candidate_ids=("d", "e", "f"),
                relevance_labels={"d": 0, "e": 4, "f": 1},
            ),
        ],
    )


def test_frozen_candidate_set_round_trips_with_checksum(tmp_path) -> None:
    dataset = _dataset()
    path = tmp_path / "candidates.json"

    dataset.save(path)
    loaded = PinnedCandidateSet.load(path)

    assert loaded == dataset
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["queries"][0]["candidate_ids"].append("tampered")
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        PinnedCandidateSet.load(path)


def test_retrieval_metrics_use_same_frozen_candidate_universe() -> None:
    dataset = _dataset()
    evaluation = evaluate_ranked_candidates(
        dataset,
        {"q-ai": ["a", "b", "c"], "q-dev": ["e", "f", "d"]},
        recall_ks=(1, 2),
    )

    assert evaluation.recall_at[1] == pytest.approx(0.5)
    assert evaluation.recall_at[2] == pytest.approx(1.0)
    assert evaluation.mean_reciprocal_rank == 1.0
    assert evaluation.ndcg_at_10 == 1.0
    assert evaluation.candidate_checksum == dataset.checksum


def test_retrieval_metrics_treat_missing_sparse_qrels_as_zero_grade() -> None:
    dataset = PinnedCandidateSet.create(
        "sparse-qrels-v1",
        [
            FrozenCandidateQuery(
                query_id="q-sparse",
                candidate_ids=("best", "strong", "irrelevant"),
                relevance_labels={"best": 4, "strong": 3},
            )
        ],
    )

    evaluation = evaluate_ranked_candidates(
        dataset,
        {"q-sparse": ["strong", "best", "irrelevant"]},
    )

    expected = (7.0 + 15.0 / 1.584962500721156) / (
        15.0 + 7.0 / 1.584962500721156
    )
    assert evaluation.ndcg_at_10 == pytest.approx(expected)


def test_retrieval_metrics_reject_candidate_drift() -> None:
    with pytest.raises(ValueError, match="changed the frozen candidate set"):
        evaluate_ranked_candidates(
            _dataset(),
            {"q-ai": ["outside"], "q-dev": ["e", "f", "d"]},
        )


def test_ndcg_uses_exponential_gains_and_exact_candidate_universe() -> None:
    score = ndcg_at_k(
        {"best": 4, "strong": 3, "irrelevant": 0},
        ("strong", "best", "irrelevant"),
        k=10,
    )

    expected = (7.0 + 15.0 / 1.584962500721156) / (
        15.0 + 7.0 / 1.584962500721156
    )
    assert score == pytest.approx(expected)
    assert score < 0.9

    with pytest.raises(ValueError, match="labeled candidate universe"):
        ndcg_at_k({"best": 4, "strong": 3}, ("best",), k=10)
