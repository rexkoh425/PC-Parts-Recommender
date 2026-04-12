"""Shared, serialisable contracts for reproducible model evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any


class SyntheticDataError(ValueError):
    """Raised when an evaluation containing synthetic rows is reported as measured evidence."""


@dataclass(frozen=True, slots=True)
class DataUseDeclaration:
    """Describe how synthetic rows were handled for one evaluation.

    ``synthetic_flags_declared`` is intentionally explicit. Metrics are eligible for
    external claims only when callers supplied row-level provenance and every synthetic
    row was excluded.
    """

    total_rows: int
    evaluated_rows: int
    synthetic_rows: int
    synthetic_rows_excluded: bool
    synthetic_flags_declared: bool = True

    def __post_init__(self) -> None:
        if min(self.total_rows, self.evaluated_rows, self.synthetic_rows) < 0:
            raise ValueError("row counts must be non-negative")
        if self.evaluated_rows > self.total_rows:
            raise ValueError("evaluated_rows cannot exceed total_rows")
        if self.synthetic_rows > self.total_rows:
            raise ValueError("synthetic_rows cannot exceed total_rows")
        if self.synthetic_rows_excluded:
            maximum_evaluated = self.total_rows - self.synthetic_rows
            if self.evaluated_rows > maximum_evaluated:
                raise ValueError(
                    "evaluated_rows cannot include synthetic rows when exclusion is declared"
                )

    @classmethod
    def from_flags(
        cls,
        is_synthetic: list[bool] | tuple[bool, ...],
        *,
        include_synthetic: bool = False,
    ) -> DataUseDeclaration:
        synthetic_rows = sum(is_synthetic)
        total_rows = len(is_synthetic)
        return cls(
            total_rows=total_rows,
            evaluated_rows=total_rows if include_synthetic else total_rows - synthetic_rows,
            synthetic_rows=synthetic_rows,
            synthetic_rows_excluded=not include_synthetic,
            synthetic_flags_declared=True,
        )

    @property
    def eligible_for_reported_metrics(self) -> bool:
        return self.synthetic_flags_declared and (
            self.synthetic_rows == 0 or self.synthetic_rows_excluded
        )

    @property
    def reporting_block_reason(self) -> str | None:
        if not self.synthetic_flags_declared:
            return "synthetic provenance was not declared"
        if self.synthetic_rows and not self.synthetic_rows_excluded:
            return "synthetic rows were included in evaluation metrics"
        return None

    def require_reportable(self) -> None:
        if reason := self.reporting_block_reason:
            raise SyntheticDataError(reason)

    def to_dict(self) -> dict[str, object]:
        return {
            "total_rows": self.total_rows,
            "evaluated_rows": self.evaluated_rows,
            "synthetic_rows": self.synthetic_rows,
            "synthetic_rows_excluded": self.synthetic_rows_excluded,
            "synthetic_flags_declared": self.synthetic_flags_declared,
            "eligible_for_reported_metrics": self.eligible_for_reported_metrics,
            "reporting_block_reason": self.reporting_block_reason,
        }


@dataclass(frozen=True, slots=True)
class MetricEstimate:
    """A metric value with its evidence count and optional confidence interval."""

    name: str
    value: float
    sample_count: int
    ci_lower: float | None = None
    ci_upper: float | None = None
    confidence_level: float | None = None
    numerator: int | None = None
    denominator: int | None = None
    unit: str = "ratio"
    slice_name: str = "overall"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("metric name must not be empty")
        if not isfinite(self.value):
            raise ValueError(f"metric {self.name!r} must be finite")
        if self.sample_count < 0:
            raise ValueError("sample_count must be non-negative")
        interval_values = (self.ci_lower, self.ci_upper, self.confidence_level)
        if any(value is None for value in interval_values) and not all(
            value is None for value in interval_values
        ):
            raise ValueError("confidence interval fields must be all present or all absent")
        if self.ci_lower is not None and self.ci_upper is not None:
            if not (isfinite(self.ci_lower) and isfinite(self.ci_upper)):
                raise ValueError("confidence interval bounds must be finite")
            if self.ci_lower > self.ci_upper:
                raise ValueError("ci_lower cannot exceed ci_upper")
        if self.confidence_level is not None and not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be between zero and one")
        if (self.numerator is None) != (self.denominator is None):
            raise ValueError("numerator and denominator must be supplied together")
        if self.numerator is not None and self.denominator is not None:
            if self.numerator < 0 or self.denominator < 0:
                raise ValueError("count fields must be non-negative")
            if self.numerator > self.denominator:
                raise ValueError("numerator cannot exceed denominator")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "value": self.value,
            "sample_count": self.sample_count,
            "confidence_interval": (
                {
                    "lower": self.ci_lower,
                    "upper": self.ci_upper,
                    "confidence_level": self.confidence_level,
                }
                if self.ci_lower is not None
                else None
            ),
            "numerator": self.numerator,
            "denominator": self.denominator,
            "unit": self.unit,
            "slice": self.slice_name,
        }


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Metrics and provenance produced by one evaluation function."""

    metrics: tuple[MetricEstimate, ...]
    data_use: DataUseDeclaration
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names = [metric.name for metric in self.metrics]
        if len(names) != len(set(names)):
            raise ValueError("metric names must be unique within an evaluation result")

    def metric(self, name: str) -> MetricEstimate:
        for metric in self.metrics:
            if metric.name == name:
                return metric
        raise KeyError(name)

    def to_dict(self) -> dict[str, object]:
        return {
            "metrics": [metric.to_dict() for metric in self.metrics],
            "synthetic_data": self.data_use.to_dict(),
            "metadata": self.metadata,
        }
