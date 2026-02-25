"""Public result types for the compatibility engine.

The compatibility package intentionally does not depend on the persistence models.  Data
ingestion, the API, and the optimiser can therefore all validate dictionary-shaped product
records without constructing ORM objects.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any


class CompatVerdict(StrEnum):
    """Outcome of one compatibility rule or an aggregate report."""

    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    UNKNOWN = "UNKNOWN"


class MissingDataRiskLevel(StrEnum):
    """Operational severity of unresolved compatibility evidence."""

    NONE = "NONE"
    ELEVATED = "ELEVATED"
    CRITICAL = "CRITICAL"


_STATUS_PRIORITY = {
    CompatVerdict.PASS: 0,
    CompatVerdict.WARNING: 1,
    CompatVerdict.UNKNOWN: 2,
    CompatVerdict.FAIL: 3,
}


@dataclass(frozen=True, slots=True)
class CompatibilityResult:
    """Auditable outcome of a single, versioned compatibility rule."""

    rule_id: str
    rule_version: str
    status: CompatVerdict
    message: str
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Do not let a caller mutate the evidence after a build was validated.
        object.__setattr__(self, "evidence", MappingProxyType(dict(self.evidence)))

    @property
    def is_blocking(self) -> bool:
        """Whether the rule found a known hard incompatibility."""

        return self.status is CompatVerdict.FAIL

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation for API and persisted reports."""

        return {
            "rule_id": self.rule_id,
            "rule_version": self.rule_version,
            "status": self.status.value,
            "message": self.message,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True, slots=True)
class CompatibilityReport:
    """Aggregate compatibility result for a pair or a complete build."""

    rule_version: str
    results: tuple[CompatibilityResult, ...] = ()

    @property
    def status(self) -> CompatVerdict:
        """Return the most conservative status present in the report."""

        if not self.results:
            return CompatVerdict.PASS
        return max((result.status for result in self.results), key=_STATUS_PRIORITY.__getitem__)

    @property
    def overall_status(self) -> CompatVerdict:
        """Alias used by API-facing callers."""

        return self.status

    @property
    def has_failures(self) -> bool:
        return any(result.status is CompatVerdict.FAIL for result in self.results)

    @property
    def has_unknowns(self) -> bool:
        return any(result.status is CompatVerdict.UNKNOWN for result in self.results)

    @property
    def has_warnings(self) -> bool:
        return any(result.status is CompatVerdict.WARNING for result in self.results)

    @property
    def is_compatible(self) -> bool:
        """Whether every hard rule is known and non-failing.

        Warnings are feasible (for example, a documented BIOS update), but an UNKNOWN is not
        accepted as compatible.  This keeps missing dimensions or connector data out of the
        optimiser's feasible set.
        """

        return not self.has_failures and not self.has_unknowns

    @property
    def is_feasible(self) -> bool:
        """Optimiser-oriented alias for :attr:`is_compatible`."""

        return self.is_compatible

    def by_rule(self, rule_id: str) -> tuple[CompatibilityResult, ...]:
        return tuple(result for result in self.results if result.rule_id == rule_id)

    @property
    def status_counts(self) -> Mapping[str, int]:
        """Stable status counts for logs, dashboards, and evaluation artefacts."""

        return MappingProxyType(
            {
                status.value: sum(result.status is status for result in self.results)
                for status in CompatVerdict
            }
        )

    @property
    def missing_data_risk(self) -> MissingDataRiskSummary:
        """Aggregate unresolved fields without discarding their originating rules."""

        unknowns = tuple(
            result for result in self.results if result.status is CompatVerdict.UNKNOWN
        )
        missing_fields: set[str] = set()
        affected_components: set[str] = set()
        for result in unknowns:
            fields = result.evidence.get("missing_fields", ())
            if isinstance(fields, (list, tuple, set, frozenset)):
                missing_fields.update(str(field) for field in fields)
            components = result.evidence.get("components", {})
            if isinstance(components, Mapping):
                affected_components.update(str(component) for component in components)

        if not unknowns:
            risk_level = MissingDataRiskLevel.NONE
        elif len(unknowns) == 1:
            risk_level = MissingDataRiskLevel.ELEVATED
        else:
            risk_level = MissingDataRiskLevel.CRITICAL
        return MissingDataRiskSummary(
            level=risk_level,
            blocks_feasibility=bool(unknowns),
            unknown_rule_count=len(unknowns),
            unknown_rule_ids=tuple(sorted({result.rule_id for result in unknowns})),
            missing_fields=tuple(sorted(missing_fields)),
            affected_components=tuple(sorted(affected_components)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_version": self.rule_version,
            "status": self.status.value,
            "is_compatible": self.is_compatible,
            "status_counts": dict(self.status_counts),
            "missing_data_risk": self.missing_data_risk.to_dict(),
            "results": [result.to_dict() for result in self.results],
        }


@dataclass(frozen=True, slots=True)
class MissingDataRiskSummary:
    """Machine-readable missing-evidence diagnostic for one compatibility report."""

    level: MissingDataRiskLevel
    blocks_feasibility: bool
    unknown_rule_count: int
    unknown_rule_ids: tuple[str, ...]
    missing_fields: tuple[str, ...]
    affected_components: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "blocks_feasibility": self.blocks_feasibility,
            "unknown_rule_count": self.unknown_rule_count,
            "unknown_rule_ids": list(self.unknown_rule_ids),
            "missing_fields": list(self.missing_fields),
            "affected_components": list(self.affected_components),
        }


@dataclass(frozen=True, slots=True)
class PowerPolicy:
    """Configurable full-build power safety policy."""

    headroom_ratio: float = 0.25
    accessory_allowance_w: float = 100.0
    gpu_transient_multiplier: float = 1.25

    def __post_init__(self) -> None:
        if not 0.0 <= self.headroom_ratio <= 1.0:
            raise ValueError("headroom_ratio must be between 0 and 1")
        if self.accessory_allowance_w < 0.0:
            raise ValueError("accessory_allowance_w must be non-negative")
        if self.gpu_transient_multiplier < 1.0:
            raise ValueError("gpu_transient_multiplier must be at least 1")
