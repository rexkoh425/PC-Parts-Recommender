"""Versioned deterministic PC component compatibility engine."""

from .engine import (
    AUTHORITATIVE_COMPATIBILITY_POLICY,
    COMPATIBILITY_AUTHORITY_KEY,
    CONTROLLED_NON_PRODUCTION_POLICY,
    DEFAULT_REQUIRED_CATEGORIES,
    DEFAULT_RULE_VERSION,
    CompatibilityEngine,
    check_build_compatibility,
)
from .models import (
    CompatibilityReport,
    CompatibilityResult,
    CompatVerdict,
    MissingDataRiskLevel,
    MissingDataRiskSummary,
    PowerPolicy,
)

__all__ = [
    "AUTHORITATIVE_COMPATIBILITY_POLICY",
    "COMPATIBILITY_AUTHORITY_KEY",
    "CONTROLLED_NON_PRODUCTION_POLICY",
    "DEFAULT_REQUIRED_CATEGORIES",
    "DEFAULT_RULE_VERSION",
    "CompatibilityEngine",
    "CompatibilityReport",
    "CompatibilityResult",
    "CompatVerdict",
    "MissingDataRiskLevel",
    "MissingDataRiskSummary",
    "PowerPolicy",
    "check_build_compatibility",
]
