from __future__ import annotations

from dataclasses import replace

import pytest

from pc_build_recommender.retrieval import (
    AdjudicationDecision,
    HumanJudgmentSet,
    LabelingQuery,
    ReviewerJudgment,
)


def _review(
    product_id: str,
    reviewer_id: str,
    grade: int,
    *,
    synthetic: bool = False,
) -> ReviewerJudgment:
    return ReviewerJudgment(
        query_id="q-ai",
        product_id=product_id,
        reviewer_id=reviewer_id,
        grade=grade,
        rationale=f"grade {grade} based on the frozen evidence",
        reviewed_at_utc="2026-07-22T01:00:00Z",
        is_synthetic=synthetic,
    )


def _dataset(*, include_adjudication: bool = True) -> HumanJudgmentSet:
    decisions = (
        (
            AdjudicationDecision(
                query_id="q-ai",
                product_id="gpu-b",
                adjudicator_id="reviewer-3",
                grade=1,
                rationale="The product is usable but misses the preferred VRAM tier.",
                adjudicated_at_utc="2026-07-22T02:00:00Z",
            ),
        )
        if include_adjudication
        else ()
    )
    return HumanJudgmentSet(
        dataset_name="human-retrieval-pilot",
        dataset_version="human-v1",
        queries=(
            LabelingQuery(
                query_id="q-ai",
                query_group_id="intent-local-ai-gpu",
                query_text="GPU for local AI",
                category="GPU",
                candidate_ids=("gpu-a", "gpu-b"),
            ),
        ),
        judgments=(
            _review("gpu-a", "reviewer-1", 4),
            _review("gpu-a", "reviewer-2", 4),
            _review("gpu-b", "reviewer-1", 0),
            _review("gpu-b", "reviewer-2", 1),
        ),
        adjudications=decisions,
    )


def test_human_labels_require_two_reviewers_and_resolve_disagreement() -> None:
    result = _dataset().adjudicate()

    assert result.summary.candidate_pair_count == 2
    assert result.summary.unanimous_pair_count == 1
    assert result.summary.adjudicated_pair_count == 1
    assert result.summary.exact_agreement_rate == 0.5
    query = result.frozen_candidates.queries[0]
    assert query.relevance_labels == {"gpu-a": 4, "gpu-b": 1}
    assert query.query_group_id == "intent-local-ai-gpu"
    assert result.frozen_candidates.eligible_for_promotion
    assert result.frozen_candidates.judgment_manifest_sha256 == _dataset().content_sha256


def test_disagreement_without_adjudication_is_rejected() -> None:
    with pytest.raises(ValueError, match="needs adjudication"):
        _dataset(include_adjudication=False).adjudicate()


def test_boolean_grade_is_not_silently_coerced_to_relevance_one() -> None:
    with pytest.raises(ValueError, match="integer from 0 to 4"):
        ReviewerJudgment(
            query_id="q",
            product_id="p",
            reviewer_id="r",
            grade=True,
            rationale="invalid boolean grade",
            reviewed_at_utc="2026-07-22T01:00:00Z",
        )


def test_synthetic_human_fixture_remains_non_promotable() -> None:
    dataset = _dataset()
    synthetic_judgment = replace(dataset.judgments[0], is_synthetic=True)
    result = replace(
        dataset,
        judgments=(synthetic_judgment, *dataset.judgments[1:]),
    ).adjudicate()

    assert result.frozen_candidates.contains_synthetic_labels
    assert not result.frozen_candidates.eligible_for_promotion
    assert "synthetic relevance labels are present" in (
        result.frozen_candidates.promotion_block_reasons
    )
