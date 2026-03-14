"""Transparent heuristic component ranker and BM25-era baseline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

import numpy as np

from .features import RankingFeatureBuilder
from .models import (
    RankedCandidate,
    RankerArtifactIdentity,
    RankerMetadata,
    ScoredCandidate,
    RankingContext,
)

DEFAULT_HEURISTIC_WEIGHTS: dict[str, float] = {
    "bm25_score": 0.05,
    "vector_similarity": 0.05,
    "rrf_score": 0.08,
    "exact_model_match": 0.02,
    "specification_match": 0.05,
    "weighted_workload_score": 0.20,
    "observed_benchmark_score": 0.08,
    "predicted_workload_score": 0.04,
    "gpu_memory_fit": 0.04,
    "memory_capacity_fit": 0.03,
    "budget_fit": 0.07,
    "price_to_performance": 0.12,
    "discount_vs_30d_median": 0.03,
    "availability_score": 0.03,
    "reliability_score": 0.04,
    "warranty_score": 0.01,
    "review_score": 0.02,
    "freshness_score": 0.01,
    "upgradeability_score": 0.02,
    "power_efficiency_score": 0.02,
    "preference_match_score": 0.04,
    "budget_share": -0.02,
    "price_percentile": -0.01,
}


class HeuristicRanker:
    """A deterministic baseline with per-feature contribution evidence."""

    def __init__(
        self,
        *,
        feature_builder: RankingFeatureBuilder | None = None,
        weights: Mapping[str, float] | None = None,
        ranker_version: str = "heuristic-v1",
    ) -> None:
        self.feature_builder = feature_builder or RankingFeatureBuilder()
        self.weights = dict(weights or DEFAULT_HEURISTIC_WEIGHTS)
        unknown = set(self.weights) - set(self.feature_builder.feature_names)
        if unknown:
            raise ValueError(f"weights reference unknown features: {sorted(unknown)}")
        self._metadata = RankerMetadata(
            ranker_version=ranker_version,
            ranking_basis="heuristic_baseline",
            feature_version=self.feature_builder.feature_version,
            model_type="weighted_normalised_features",
            feature_names=self.feature_builder.feature_names,
            created_at_utc=datetime.now(UTC).isoformat(),
            parameters={"weights": self.weights},
        )

    @property
    def metadata(self) -> RankerMetadata:
        return self._metadata

    @property
    def artifact_identity(self) -> RankerArtifactIdentity:
        raise RuntimeError("heuristic rankers do not have persisted ML artifact identity")

    @property
    def verified_artifact_loaded(self) -> bool:
        return False

    def rank_query(
        self, context: RankingContext, candidates: Sequence[ScoredCandidate]
    ) -> list[RankedCandidate]:
        if not candidates:
            return []
        batch = self.feature_builder.build(context, candidates)
        values = batch.values
        minima = values.min(axis=0)
        ranges = values.max(axis=0) - minima
        normalised = np.divide(
            values - minima,
            ranges,
            out=np.zeros_like(values),
            where=ranges > 0,
        )
        name_to_column = {
            name: position for position, name in enumerate(self.feature_builder.feature_names)
        }
        contributions: list[dict[str, float]] = []
        scores: list[float] = []
        for row in normalised:
            row_contributions = {
                name: weight * float(row[name_to_column[name]])
                for name, weight in self.weights.items()
            }
            contributions.append(row_contributions)
            scores.append(sum(row_contributions.values()))

        order = sorted(
            range(len(candidates)),
            key=lambda index: (-scores[index], candidates[index].product_id),
        )
        return [
            RankedCandidate(
                candidate=candidates[index],
                score=scores[index],
                rank=rank,
                ranker_version=self.metadata.ranker_version,
                ranking_basis=self.metadata.ranking_basis,
                feature_contributions=contributions[index],
            )
            for rank, index in enumerate(order, start=1)
        ]
