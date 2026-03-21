"""Versioned workload contracts and exact model routing for performance inference."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import isfinite
from typing import Any

from .contracts import PerformanceEstimate, PerformanceModelArtifact
from .inference import estimate_performance, observed_performance_estimate


@dataclass(frozen=True, slots=True)
class WorkloadModelSpec:
    """Exact target and benchmark cohort supported by one model route."""

    category: str
    workload: str
    feature_columns: tuple[str, ...]
    metric: str
    unit: str
    higher_is_better: bool
    cohort: tuple[tuple[str, str], ...] = ()
    schema_version: str = "pc-build-recommender.workload-model-spec.v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature_columns", tuple(self.feature_columns))
        object.__setattr__(
            self,
            "cohort",
            tuple(sorted((str(name), str(value)) for name, value in self.cohort)),
        )
        if not all((self.category, self.workload, self.metric, self.unit)):
            raise ValueError("workload spec identifiers, metric, and unit must not be empty")
        if not self.feature_columns or len(self.feature_columns) != len(set(self.feature_columns)):
            raise ValueError("workload spec features must be non-empty and unique")
        cohort_names = [name for name, _ in self.cohort]
        if len(cohort_names) != len(set(cohort_names)) or any(not name for name in cohort_names):
            raise ValueError("workload cohort fields must have unique non-empty names")

    @property
    def key(self) -> tuple[str, str]:
        return (self.category, self.workload)

    @property
    def spec_sha256(self) -> str:
        payload = json.dumps(
            {
                "schema_version": self.schema_version,
                "category": self.category,
                "workload": self.workload,
                "feature_columns": self.feature_columns,
                "metric": self.metric,
                "unit": self.unit,
                "higher_is_better": self.higher_is_better,
                "cohort": self.cohort,
            },
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class ObservedPerformanceObservation:
    """Measured evidence that must exactly match a registered target contract."""

    product_id: str
    category: str
    workload: str
    score: float
    metric: str
    unit: str
    higher_is_better: bool
    source: str
    cohort: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "cohort",
            tuple(sorted((str(name), str(value)) for name, value in self.cohort)),
        )
        if not self.product_id or not self.source.strip():
            raise ValueError("observed performance requires a product ID and source")
        if not isfinite(self.score) or self.score <= 0:
            raise ValueError("observed performance score must be finite and positive")
        cohort_names = [name for name, _ in self.cohort]
        if len(cohort_names) != len(set(cohort_names)) or any(not name for name in cohort_names):
            raise ValueError("observed cohort fields must have unique non-empty names")

    def mismatch_reasons(self, spec: WorkloadModelSpec) -> tuple[str, ...]:
        reasons: list[str] = []
        for name, actual, expected in (
            ("category", self.category, spec.category),
            ("workload", self.workload, spec.workload),
            ("metric", self.metric, spec.metric),
            ("unit", self.unit, spec.unit),
        ):
            if actual != expected:
                reasons.append(f"{name} {actual!r} does not match {expected!r}")
        if self.higher_is_better != spec.higher_is_better:
            reasons.append("target direction does not match")
        if self.cohort != spec.cohort:
            reasons.append("benchmark cohort does not match")
        return tuple(reasons)


class PerformanceModelRegistry:
    """Route exact category/workload contracts without cross-target fallbacks."""

    def __init__(self) -> None:
        self._specs: dict[tuple[str, str], WorkloadModelSpec] = {}
        self._artifacts: dict[tuple[str, str], PerformanceModelArtifact] = {}

    def register_spec(self, spec: WorkloadModelSpec) -> None:
        existing = self._specs.get(spec.key)
        if existing is not None and existing != spec:
            raise ValueError(f"a different workload spec is already registered for {spec.key}")
        self._specs[spec.key] = spec

    def register_artifact(
        self,
        artifact: PerformanceModelArtifact,
        *,
        replace: bool = False,
    ) -> None:
        key = (artifact.config.category, artifact.config.workload)
        spec = self._specs.get(key)
        if spec is None:
            raise KeyError(f"no workload spec is registered for {key}")
        if tuple(artifact.config.feature_columns) != spec.feature_columns:
            raise ValueError("artifact feature contract does not match the workload spec")
        existing = self._artifacts.get(key)
        if (
            existing is not None
            and existing.model_version != artifact.model_version
            and not replace
        ):
            raise ValueError(f"a different artifact is already registered for {key}")
        self._artifacts[key] = artifact

    def resolve(self, category: str, workload: str) -> PerformanceModelArtifact:
        key = (category, workload)
        if key not in self._specs:
            raise KeyError(f"unknown performance workload route: {key}")
        try:
            return self._artifacts[key]
        except KeyError as exc:
            raise LookupError(f"no performance artifact is loaded for {key}") from exc

    def estimate(
        self,
        *,
        category: str,
        workload: str,
        product_id: str,
        features: dict[str, float | int | None] | None = None,
        observed: ObservedPerformanceObservation | None = None,
    ) -> PerformanceEstimate:
        """Prefer an exactly comparable observation before resolving any model."""

        key = (category, workload)
        try:
            spec = self._specs[key]
        except KeyError as exc:
            raise KeyError(f"unknown performance workload route: {key}") from exc
        if observed is not None:
            if observed.product_id != product_id:
                raise ValueError("observed product ID does not match the inference request")
            mismatches = observed.mismatch_reasons(spec)
            if mismatches:
                raise ValueError("observed benchmark is not comparable: " + "; ".join(mismatches))
            return observed_performance_estimate(observed.score, observed.source)
        artifact = self.resolve(category, workload)
        return estimate_performance(artifact, features)

    def snapshot(self) -> dict[str, Any]:
        """Return a deterministic deployment inventory without model objects."""

        models: list[dict[str, Any]] = []
        for key in sorted(self._specs):
            spec = self._specs[key]
            artifact = self._artifacts.get(key)
            models.append(
                {
                    "category": spec.category,
                    "workload": spec.workload,
                    "workload_spec_sha256": spec.spec_sha256,
                    "model_version": artifact.model_version if artifact is not None else None,
                    "promotable": artifact.promotable if artifact is not None else None,
                }
            )
        return {
            "schema_version": "pc-build-recommender.performance-model-registry.v1",
            "models": models,
        }
