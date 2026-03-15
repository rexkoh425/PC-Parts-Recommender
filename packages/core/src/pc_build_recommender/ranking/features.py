"""Deterministic feature construction for heuristic and learned rankers."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pc_build_recommender.retrieval.text import tokenize

from .models import ScoredCandidate, RankingContext

FloatFeatureMatrix = NDArray[np.float64]


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _unit(value: Any, default: float = 0.0) -> float:
    number = _number(value, default)
    if number > 1.0:
        number /= 100.0
    return min(1.0, max(0.0, number))


def _first(candidate: ScoredCandidate, *names: str) -> Any:
    for name in names:
        value = candidate.get(name)
        if value is not None:
            return value
    return None


@dataclass(frozen=True, slots=True)
class FeatureBatch:
    query_id: str
    product_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    values: FloatFeatureMatrix


class RankingFeatureBuilder:
    """Build the same ordered feature matrix for training and serving."""

    feature_version = "ranking-features-v1"
    feature_names = (
        "bm25_score",
        "vector_similarity",
        "rrf_score",
        "exact_model_match",
        "specification_match",
        "weighted_workload_score",
        "observed_benchmark_score",
        "predicted_workload_score",
        "has_observed_benchmark",
        "gpu_memory_fit",
        "memory_capacity_fit",
        "price_log1p",
        "budget_share",
        "budget_fit",
        "price_to_performance",
        "price_percentile",
        "discount_vs_30d_median",
        "availability_score",
        "reliability_score",
        "warranty_score",
        "review_score",
        "freshness_score",
        "upgradeability_score",
        "power_efficiency_score",
        "preference_match_score",
    )

    def build(
        self, context: RankingContext, candidates: Sequence[ScoredCandidate]
    ) -> FeatureBatch:
        if not candidates:
            return FeatureBatch(
                query_id=context.query_id,
                product_ids=(),
                feature_names=self.feature_names,
                values=np.empty((0, len(self.feature_names)), dtype=np.float64),
            )
        prices = np.asarray(
            [
                candidate.price_sgd if candidate.price_sgd is not None else np.nan
                for candidate in candidates
            ],
            dtype=np.float64,
        )
        known_prices = prices[np.isfinite(prices)]
        rows = [
            self._build_row(context, candidate, known_prices)
            for candidate in candidates
        ]
        values = np.asarray(rows, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("feature builder produced a non-finite value")
        return FeatureBatch(
            query_id=context.query_id,
            product_ids=tuple(candidate.product_id for candidate in candidates),
            feature_names=self.feature_names,
            values=values,
        )

    def _build_row(
        self,
        context: RankingContext,
        candidate: ScoredCandidate,
        known_prices: NDArray[np.float64],
    ) -> list[float]:
        weighted_workload = self._weighted_workload(context, candidate)
        price = candidate.price_sgd
        if price is None:
            price_log = 0.0
            budget_share = 1.5 if context.budget_sgd else 0.0
            budget_fit = 0.0
            price_to_performance = 0.0
            price_percentile = 1.0
        else:
            price_log = math.log1p(price)
            budget_share = price / context.budget_sgd if context.budget_sgd else 0.0
            budget_fit = max(0.0, 1.0 - budget_share) if context.budget_sgd else 0.5
            price_to_performance = weighted_workload * 1000.0 / max(price, 1.0)
            price_percentile = (
                float(np.count_nonzero(known_prices <= price)) / len(known_prices)
                if len(known_prices)
                else 1.0
            )

        median_30d = _number(
            candidate.signals.get(
                "median_price_30d_sgd", candidate.attributes.get("median_price_30d_sgd")
            )
        )
        discount = (
            (median_30d - price) / median_30d
            if price is not None and median_30d > 0
            else 0.0
        )
        observed = _number(candidate.signals.get("observed_benchmark_score"))
        predicted = _number(candidate.signals.get("predicted_workload_score"))

        values: Mapping[str, float] = {
            "bm25_score": _number(candidate.retrieval_scores.get("bm25_score")),
            "vector_similarity": _number(candidate.retrieval_scores.get("vector_similarity")),
            "rrf_score": _number(candidate.retrieval_scores.get("rrf_score")),
            "exact_model_match": self._exact_model_match(context, candidate),
            "specification_match": self._specification_match(context, candidate),
            "weighted_workload_score": weighted_workload,
            "observed_benchmark_score": observed,
            "predicted_workload_score": predicted,
            "has_observed_benchmark": float("observed_benchmark_score" in candidate.signals),
            "gpu_memory_fit": self._minimum_fit(
                context,
                candidate,
                "minimum_gpu_vram_gb",
                ("vram_gb", "gpu_vram_gb", "vram_capacity_gb"),
            ),
            "memory_capacity_fit": self._minimum_fit(
                context,
                candidate,
                "minimum_memory_gb",
                ("capacity_gb", "memory_gb", "total_capacity_gb"),
            ),
            "price_log1p": price_log,
            "budget_share": budget_share,
            "budget_fit": budget_fit,
            "price_to_performance": price_to_performance,
            "price_percentile": price_percentile,
            "discount_vs_30d_median": discount,
            "availability_score": _unit(candidate.signals.get("availability_score")),
            "reliability_score": _unit(candidate.signals.get("reliability_score")),
            "warranty_score": min(
                1.0,
                max(
                    0.0,
                    _number(
                        candidate.signals.get(
                            "warranty_years", candidate.attributes.get("warranty_years")
                        )
                    )
                    / 10.0,
                ),
            ),
            "review_score": _unit(candidate.signals.get("review_score")),
            "freshness_score": _unit(candidate.signals.get("freshness_score")),
            "upgradeability_score": _unit(candidate.signals.get("upgradeability_score")),
            "power_efficiency_score": _unit(candidate.signals.get("power_efficiency_score")),
            "preference_match_score": self._preference_match(context, candidate),
        }
        return [values[name] for name in self.feature_names]

    @staticmethod
    def _weighted_workload(context: RankingContext, candidate: ScoredCandidate) -> float:
        if not candidate.workload_scores:
            return _number(candidate.signals.get("workload_score"))
        if not context.workload_weights:
            return sum(candidate.workload_scores.values()) / len(candidate.workload_scores)
        total_weight = sum(context.workload_weights.values())
        return sum(
            context.workload_weights.get(name, 0.0) * score
            for name, score in candidate.workload_scores.items()
        ) / total_weight

    @staticmethod
    def _exact_model_match(context: RankingContext, candidate: ScoredCandidate) -> float:
        if "exact_model_match" in candidate.signals:
            return _unit(candidate.signals["exact_model_match"])
        model = candidate.get("model")
        if not model or not context.query_text:
            return 0.0
        model_tokens = set(tokenize(str(model)))
        query_tokens = set(tokenize(context.query_text))
        return float(bool(model_tokens) and model_tokens.issubset(query_tokens))

    @staticmethod
    def _minimum_fit(
        context: RankingContext,
        candidate: ScoredCandidate,
        requirement_name: str,
        field_names: tuple[str, ...],
    ) -> float:
        required = _number(context.requirements.get(requirement_name))
        if required <= 0:
            return 0.0
        actual = _number(_first(candidate, *field_names))
        return min(2.0, max(0.0, actual / required))

    def _specification_match(
        self, context: RankingContext, candidate: ScoredCandidate
    ) -> float:
        if "specification_match" in candidate.signals:
            return _unit(candidate.signals["specification_match"])
        checks: list[bool] = []
        for name, required in context.requirements.items():
            if not name.startswith("minimum_"):
                actual = candidate.get(name)
                if actual is not None:
                    checks.append(str(actual).casefold() == str(required).casefold())
        return sum(checks) / len(checks) if checks else 0.0

    @staticmethod
    def _preference_match(context: RankingContext, candidate: ScoredCandidate) -> float:
        if "preference_match_score" in candidate.signals:
            return _unit(candidate.signals["preference_match_score"])
        preferred = context.preferences.get("preferred_brands", ())
        if isinstance(preferred, str):
            preferred = (preferred,)
        if preferred and candidate.brand:
            return float(candidate.brand.casefold() in {str(item).casefold() for item in preferred})
        return 0.0
