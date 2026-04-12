"""Artifact-backed, exact-route workload inference for online serving."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
from math import isfinite
from types import MappingProxyType

from pc_build_recommender.domain import MasterProduct, WorkloadLabel, WorkloadPerformanceSignal
from pc_build_recommender.performance_models import (
    ARTIFACT_SCHEMA_VERSION,
    PerformanceModelArtifact,
    estimate_performance,
)

from .catalog import ApplicationCatalog


class ArtifactPerformanceProvider:
    """Route immutable performance artifacts by exact category and workload."""

    def __init__(self, artifacts: Sequence[PerformanceModelArtifact]) -> None:
        routes: dict[tuple[str, str], PerformanceModelArtifact] = {}
        for artifact in artifacts:
            route = (artifact.config.category, artifact.config.workload)
            if route in routes:
                raise ValueError(f"duplicate performance model route: {route}")
            if artifact.schema_version != ARTIFACT_SCHEMA_VERSION:
                raise ValueError(f"performance model route {route} has an unsupported schema")
            if len(artifact.model_version) != 64 or any(
                character not in "0123456789abcdef" for character in artifact.model_version
            ):
                raise ValueError(f"performance model route {route} has an invalid version digest")
            routes[route] = artifact
        if not routes:
            raise ValueError("at least one performance artifact is required")
        self._artifacts = MappingProxyType(routes)

    @property
    def model_versions(self) -> Mapping[str, str]:
        return MappingProxyType(
            {
                f"{category}/{workload}": artifact.model_version
                for (category, workload), artifact in sorted(self._artifacts.items())
            }
        )

    @property
    def artifacts(self) -> tuple[PerformanceModelArtifact, ...]:
        return tuple(self._artifacts[route] for route in sorted(self._artifacts))

    @property
    def all_promotable(self) -> bool:
        return all(
            artifact.promotable and artifact.precise_predictions_enabled
            for artifact in self._artifacts.values()
        )

    @property
    def promotion_block_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        for route, artifact in sorted(self._artifacts.items()):
            if artifact.promotable and artifact.precise_predictions_enabled:
                continue
            route_name = f"{route[0]}/{route[1]}"
            blockers = artifact.promotion_block_reasons or ("not promotion-eligible",)
            reasons.extend(f"{route_name}: {reason}" for reason in blockers)
            if artifact.promotable and not artifact.precise_predictions_enabled:
                reasons.append(f"{route_name}: precise predictions are disabled")
        return tuple(dict.fromkeys(reasons))

    def validate_catalog(self, catalog: ApplicationCatalog) -> None:
        """Reject artifacts whose online feature contract cannot be supplied."""

        supported_workloads = {workload.value for workload in WorkloadLabel}
        for (category, workload), artifact in sorted(self._artifacts.items()):
            if workload not in supported_workloads:
                raise ValueError(
                    f"performance model route {category}/{workload} is not a serving workload"
                )
            products = [
                item.product for item in catalog.items if item.product.category.value == category
            ]
            if not products:
                raise ValueError(
                    f"performance model route {category}/{workload} has no catalog category"
                )
            available_fields = set(self._raw_features(products[0]))
            unknown = sorted(set(artifact.config.feature_columns).difference(available_fields))
            if unknown:
                raise ValueError(
                    f"performance model route {category}/{workload} uses unavailable product "
                    f"features: {unknown}"
                )
            for product in products:
                self._features_for(product, artifact)

    def estimate(
        self,
        product: MasterProduct,
        workload: str,
    ) -> WorkloadPerformanceSignal | None:
        """Estimate one exact route; absence means the route is unsupported."""

        route = (product.category.value, workload)
        artifact = self._artifacts.get(route)
        if artifact is None:
            return None
        estimate = estimate_performance(
            artifact,
            self._features_for(product, artifact),
        )
        return WorkloadPerformanceSignal(
            workload=WorkloadLabel(workload),
            metric=artifact.config.target_column,
            unit=None,
            score=estimate.score,
            relative_score=estimate.relative_score,
            basis="relative" if estimate.basis == "relative_only" else estimate.basis,
            confidence=estimate.confidence,
            decision=estimate.decision,
            model_version=estimate.model_version,
            lower_score=estimate.lower_score,
            upper_score=estimate.upper_score,
        )

    @staticmethod
    def _raw_features(product: MasterProduct) -> dict[str, object]:
        return {
            **product.common_attributes.model_dump(mode="python"),
            **product.category_attributes.model_dump(mode="python"),
        }

    @classmethod
    def _features_for(
        cls,
        product: MasterProduct,
        artifact: PerformanceModelArtifact,
    ) -> dict[str, float | int | None]:
        raw = cls._raw_features(product)
        features: dict[str, float | int | None] = {}
        for name in artifact.config.feature_columns:
            if name not in raw:
                raise ValueError(f"catalog product is missing model feature {name!r}")
            value = raw[name]
            if value is None:
                features[name] = None
            elif isinstance(value, bool):
                features[name] = int(value)
            elif isinstance(value, (int, float, Decimal)):
                numeric = float(value)
                if not isfinite(numeric):
                    raise ValueError(f"catalog feature {name!r} must be finite")
                features[name] = numeric
            else:
                raise TypeError(
                    f"catalog feature {name!r} for product {product.product_id!r} is not numeric"
                )
        return features
