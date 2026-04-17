"""Evaluation hooks that enforce a frozen candidate universe."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pc_build_recommender.evaluation.manifest import json_sha256
from pc_build_recommender.retrieval import (
    ArtifactBoundRankingEvidence,
    FrozenCandidateSet,
    QueryGroupSplit,
    evaluate_ranked_candidates,
)

from .lambdamart import LambdaMARTRanker, relative_ndcg_improvement
from .models import ProductRanker, RankingCandidate, RankingContext


@dataclass(frozen=True, slots=True)
class FrozenRankingEvaluation:
    dataset_version: str
    candidate_checksum: str
    query_count: int
    ndcg_at_10: float
    baseline_ndcg_at_10: float | None = None
    relative_improvement_percent: float | None = None
    ranker_version: str | None = None
    ranking_basis: str | None = None


@dataclass(frozen=True, slots=True)
class ArtifactBoundRankerOutput:
    """Exact ranking output and its artifact/feature/score provenance binding."""

    rankings: Mapping[str, tuple[str, ...]]
    evidence: ArtifactBoundRankingEvidence


def generate_artifact_bound_rankings(
    ranker: LambdaMARTRanker,
    dataset: FrozenCandidateSet,
    *,
    model_name: str,
    contexts: Mapping[str, RankingContext],
    candidates: Mapping[str, Sequence[RankingCandidate]],
    query_split: QueryGroupSplit | None = None,
    split_name: str | None = None,
) -> ArtifactBoundRankerOutput:
    """Score a frozen snapshot with exact verified artifact bytes and bind every input/output."""

    if not ranker.verified_artifact_loaded:
        raise ValueError("artifact-bound evaluation requires a manifest-verified loaded ranker")
    if not model_name:
        raise ValueError("model_name must not be empty")
    if query_split is None:
        if split_name is not None:
            raise ValueError("split_name requires a frozen query split")
        evaluation_dataset = dataset
    else:
        if split_name is None:
            raise ValueError("split_name is required with a frozen query split")
        evaluation_dataset = query_split.subset(dataset, split_name)

    identity = ranker.artifact_identity
    metadata = ranker.metadata
    if metadata.model_sha256 != identity.model_sha256:
        raise ValueError("loaded ranker metadata and model artifact hashes do not match")

    feature_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    rankings: dict[str, tuple[str, ...]] = {}
    row_count = 0
    for query in evaluation_dataset.queries:
        context = contexts.get(query.query_id)
        query_candidates = candidates.get(query.query_id)
        if context is None or query_candidates is None:
            raise ValueError(f"missing context or candidates for query {query.query_id!r}")
        if context.query_id != query.query_id:
            raise ValueError(f"context query ID does not match {query.query_id!r}")
        candidate_rows = tuple(query_candidates)
        if {candidate.product_id for candidate in candidate_rows} != set(query.candidate_ids):
            raise ValueError(
                f"candidate objects for {query.query_id!r} do not match the frozen set"
            )
        batch = ranker.feature_builder.build(context, candidate_rows)
        scores = ranker.predict_feature_batch(batch)
        feature_rows.append(
            {
                "query_id": query.query_id,
                "candidates": [
                    {
                        "product_id": candidate.product_id,
                        "values_hex": [float(value).hex() for value in batch.values[index]],
                    }
                    for index, candidate in enumerate(candidate_rows)
                ],
            }
        )
        score_rows.append(
            {
                "query_id": query.query_id,
                "scores": [
                    {
                        "product_id": candidate.product_id,
                        "score_hex": float(scores[index]).hex(),
                    }
                    for index, candidate in enumerate(candidate_rows)
                ],
            }
        )
        order = sorted(
            range(len(candidate_rows)),
            key=lambda index: (-float(scores[index]), candidate_rows[index].product_id),
        )
        rankings[query.query_id] = tuple(candidate_rows[index].product_id for index in order)
        row_count += len(candidate_rows)

    canonical_rankings = {query_id: list(product_ids) for query_id, product_ids in rankings.items()}
    feature_snapshot = {
        "feature_version": metadata.feature_version,
        "feature_names": list(metadata.feature_names),
        "candidate_snapshot_sha256": evaluation_dataset.checksum,
        "rows": feature_rows,
    }
    score_snapshot = {
        "model_name": model_name,
        "ranker_version": metadata.ranker_version,
        "candidate_snapshot_sha256": evaluation_dataset.checksum,
        "rows": score_rows,
    }
    evidence = ArtifactBoundRankingEvidence.create(
        model_name=model_name,
        ranker_version=metadata.ranker_version,
        model_sha256=identity.model_sha256,
        metadata_sha256=identity.metadata_sha256,
        manifest_sha256=identity.manifest_sha256,
        ranker_metadata_payload=metadata.to_dict(),
        feature_version=metadata.feature_version,
        feature_names=metadata.feature_names,
        candidate_snapshot_sha256=evaluation_dataset.checksum,
        feature_snapshot_sha256=json_sha256(feature_snapshot),
        score_snapshot_sha256=json_sha256(score_snapshot),
        ranking_sha256=json_sha256(canonical_rankings),
        split_name=split_name,
        split_checksum=query_split.checksum if query_split is not None else None,
        query_count=len(evaluation_dataset.queries),
        row_count=row_count,
    )
    return ArtifactBoundRankerOutput(rankings=rankings, evidence=evidence)


def assert_complete_frozen_rankings(
    dataset: FrozenCandidateSet,
    ranked_product_ids: Mapping[str, Sequence[str]],
) -> None:
    """Require each model to rank exactly the same products for every query."""

    expected_query_ids = {query.query_id for query in dataset.queries}
    if set(ranked_product_ids) != expected_query_ids:
        raise ValueError("ranking query IDs do not match the frozen dataset")
    for query in dataset.queries:
        if query.query_id not in ranked_product_ids:
            raise ValueError(f"missing ranking for query {query.query_id!r}")
        ranked = list(ranked_product_ids[query.query_id])
        if len(ranked) != len(set(ranked)):
            raise ValueError(f"ranking for query {query.query_id!r} contains duplicates")
        if set(ranked) != set(query.candidate_ids):
            raise ValueError(
                f"ranking for query {query.query_id!r} does not match the frozen candidate set"
            )


def rankings_from_scores(
    dataset: FrozenCandidateSet,
    scores: Mapping[str, Mapping[str, float]],
) -> dict[str, list[str]]:
    """Create stable rankings from externally-produced model scores."""

    result: dict[str, list[str]] = {}
    for query in dataset.queries:
        query_scores = scores.get(query.query_id)
        if query_scores is None or set(query_scores) != set(query.candidate_ids):
            raise ValueError(
                f"scores for query {query.query_id!r} do not match the frozen candidate set"
            )
        if any(not math.isfinite(float(score)) for score in query_scores.values()):
            raise ValueError(f"scores for query {query.query_id!r} must be finite")
        result[query.query_id] = sorted(
            query.candidate_ids,
            key=lambda product_id: (-float(query_scores[product_id]), product_id),
        )
    return result


def evaluate_frozen_rankings(
    dataset: FrozenCandidateSet,
    ranked_product_ids: Mapping[str, Sequence[str]],
    *,
    baseline_ranked_product_ids: Mapping[str, Sequence[str]] | None = None,
    ranker_version: str | None = None,
    ranking_basis: str | None = None,
) -> FrozenRankingEvaluation:
    """Evaluate NDCG@10 and optional relative gain on one checksummed set."""

    assert_complete_frozen_rankings(dataset, ranked_product_ids)
    result = evaluate_ranked_candidates(dataset, ranked_product_ids)
    baseline_ndcg: float | None = None
    improvement: float | None = None
    if baseline_ranked_product_ids is not None:
        assert_complete_frozen_rankings(dataset, baseline_ranked_product_ids)
        baseline = evaluate_ranked_candidates(dataset, baseline_ranked_product_ids)
        baseline_ndcg = baseline.ndcg_at_10
        if baseline_ndcg > 0:
            improvement = relative_ndcg_improvement(result.ndcg_at_10, baseline_ndcg)
    return FrozenRankingEvaluation(
        dataset_version=dataset.version,
        candidate_checksum=dataset.checksum,
        query_count=len(dataset.queries),
        ndcg_at_10=result.ndcg_at_10,
        baseline_ndcg_at_10=baseline_ndcg,
        relative_improvement_percent=improvement,
        ranker_version=ranker_version,
        ranking_basis=ranking_basis,
    )


def evaluate_product_ranker(
    ranker: ProductRanker,
    dataset: FrozenCandidateSet,
    *,
    contexts: Mapping[str, RankingContext],
    candidates: Mapping[str, Sequence[RankingCandidate]],
    baseline_ranked_product_ids: Mapping[str, Sequence[str]] | None = None,
) -> FrozenRankingEvaluation:
    """Run a ranker end-to-end while guarding candidate-set parity."""

    rankings: dict[str, list[str]] = {}
    for query in dataset.queries:
        context = contexts.get(query.query_id)
        query_candidates = candidates.get(query.query_id)
        if context is None or query_candidates is None:
            raise ValueError(f"missing context or candidates for query {query.query_id!r}")
        if context.query_id != query.query_id:
            raise ValueError(f"context query ID does not match {query.query_id!r}")
        if {item.product_id for item in query_candidates} != set(query.candidate_ids):
            raise ValueError(
                f"candidate objects for {query.query_id!r} do not match the frozen set"
            )
        rankings[query.query_id] = [
            item.product_id for item in ranker.rank_query(context, query_candidates)
        ]
    return evaluate_frozen_rankings(
        dataset,
        rankings,
        baseline_ranked_product_ids=baseline_ranked_product_ids,
        ranker_version=ranker.metadata.ranker_version,
        ranking_basis=ranker.metadata.ranking_basis,
    )
