"""Strict, content-addressed contracts for entity-resolution release authority.

These records deliberately separate measured model evidence, operational policy, and
data-rights approval.  A caller cannot turn a diagnostic model into a production model
by passing an eligibility boolean: authority is derived only after the exact persisted
artifacts have been loaded, hashed, and cross-checked by :mod:`.serving`.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn, TypedDict, cast

ER_EVALUATION_SCHEMA_VERSION = "pc-build-recommender.er-production-evaluation.v1"
ER_EVALUATION_SCHEMA_VERSION_V2 = "pc-build-recommender.er-production-evaluation.v2"
ER_POLICY_SCHEMA_VERSION = "pc-build-recommender.er-production-policy.v1"
ER_RIGHTS_APPROVAL_SCHEMA_VERSION = "pc-build-recommender.er-rights-approval.v1"
ER_RELEASE_IDENTITY_SCHEMA_VERSION = "pc-build-recommender.er-release-identity.v1"

ER_REQUIRED_APPROVED_USES = frozenset(
    {"derive_features", "publish_metrics", "serve_derived_model", "train_model"}
)
ER_PRODUCTION_TERRITORY = "SG"
ER_MINIMUM_PRODUCTION_PRECISION = 0.99
ER_MINIMUM_PRODUCTION_LABELLED_PAIRS = 1000
ER_MINIMUM_PRODUCTION_AUTO_MATCHES = 100
ER_MINIMUM_PRODUCTION_RECALL = 0.94
ER_MINIMUM_PRODUCTION_F1 = 0.96
ER_MINIMUM_PRODUCTION_AUTO_MATCH_THRESHOLD = 0.98
ER_MINIMUM_PRODUCTION_PRODUCTS = 750
ER_MINIMUM_PRODUCTION_MAPPING_RATE = 0.80
ER_MINIMUM_PRODUCTION_CRITICAL_FIELD_RATE = 0.90
_MAX_CONTRACT_BYTES = 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}")


class EntityResolutionMatcherKwargs(TypedDict):
    max_candidates: int
    minimum_text_score: float
    minimum_auto_margin: float
    evidence_candidate_limit: int


class ProductionCatalogPolicyKwargs(TypedDict):
    minimum_products: int
    minimum_products_per_category: int
    minimum_mapping_rate: float
    minimum_critical_field_rate: float
    require_complete_priced_coverage: bool
    require_complete_in_stock_coverage: bool
    require_complete_product_provenance: bool
    require_complete_offer_provenance: bool
    require_explicit_offer_rights: bool
    require_production_offer_rights: bool
    require_complete_listing_provenance: bool
    minimum_er_precision: float
    minimum_er_labelled_pairs: int
    require_promoted_entity_resolution_model: bool


class EntityResolutionContractError(ValueError):
    """Raised when persisted ER authority evidence is malformed or untrusted."""


def _reject_constant(value: str) -> NoReturn:
    raise EntityResolutionContractError(f"non-finite JSON number is forbidden: {value}")


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EntityResolutionContractError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _load_json_object(path: str | Path) -> tuple[Path, dict[str, Any], bytes]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    size = resolved.stat().st_size
    if size > _MAX_CONTRACT_BYTES:
        raise EntityResolutionContractError(
            f"{resolved.name} exceeds the {_MAX_CONTRACT_BYTES}-byte contract limit"
        )
    raw = resolved.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise EntityResolutionContractError(f"{resolved.name} must be UTF-8 JSON") from error
    try:
        payload: Any = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as error:
        raise EntityResolutionContractError(f"invalid JSON in {resolved.name}: {error}") from error
    if not isinstance(payload, dict):
        raise EntityResolutionContractError(f"{resolved.name} must contain a JSON object")
    return resolved, cast(dict[str, Any], payload), raw


def _canonical_json_sha256(payload: Mapping[str, Any], *, omit: str | None = None) -> str:
    material = dict(payload)
    if omit is not None:
        material.pop(omit, None)
    try:
        encoded = json.dumps(
            material,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise EntityResolutionContractError("contract is not canonical JSON") from error
    return hashlib.sha256(encoded).hexdigest()


def entity_resolution_file_sha256(path: str | Path) -> str:
    """Return the SHA-256 of exact persisted bytes for an external release manifest."""

    resolved = Path(path).resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_exact_fields(
    payload: Mapping[str, Any],
    required: frozenset[str],
    *,
    contract: str,
) -> None:
    missing = sorted(required - set(payload))
    extra = sorted(set(payload) - required)
    if missing:
        raise EntityResolutionContractError(f"{contract} missing fields: {missing}")
    if extra:
        raise EntityResolutionContractError(f"{contract} contains unknown fields: {extra}")


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EntityResolutionContractError(f"{name} must be a non-empty string")
    return value.strip()


def _sha256(value: object, name: str) -> str:
    result = _text(value, name)
    if _SHA256.fullmatch(result) is None:
        raise EntityResolutionContractError(f"{name} must be a lowercase SHA-256")
    return result


def _bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise EntityResolutionContractError(f"{name} must be a boolean")
    return value


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise EntityResolutionContractError(f"{name} must be an integer >= {minimum}")
    return value


def _number(
    value: object,
    name: str,
    *,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise EntityResolutionContractError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise EntityResolutionContractError(
            f"{name} must be finite and between {minimum} and {maximum}"
        )
    return result


def _timestamp(value: object, name: str) -> datetime:
    raw = _text(value, name)
    try:
        result = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as error:
        raise EntityResolutionContractError(f"{name} must be an ISO-8601 timestamp") from error
    if result.tzinfo is None or result.utcoffset() is None:
        raise EntityResolutionContractError(f"{name} must include a timezone")
    return result.astimezone(UTC)


def _optional_timestamp(value: object, name: str) -> datetime | None:
    return None if value is None else _timestamp(value, name)


def _sorted_unique_text_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise EntityResolutionContractError(f"{name} must be an array")
    result = tuple(_text(item, f"{name} item") for item in value)
    if not result:
        raise EntityResolutionContractError(f"{name} must not be empty")
    if result != tuple(sorted(set(result))):
        raise EntityResolutionContractError(f"{name} must be sorted and unique")
    return result


@dataclass(frozen=True, slots=True)
class EntityResolutionProductionEvaluation:
    """Frozen v2 human-labelled evaluation; eligibility comes from policy, not flags."""

    evaluation_id: str
    dataset_version: str
    model_version: str
    label_source: str
    synthetic: bool
    precision: float
    labelled_pair_count: int
    evaluated_at: datetime
    schema_version: str = ER_EVALUATION_SCHEMA_VERSION
    artifact_sha256: str | None = None
    review_queue_sha256: str | None = None
    frozen_test_groups_sha256: str | None = None
    auto_match_threshold: float | None = None
    precision_numerator: int | None = None
    precision_denominator: int | None = None
    precision_ci_lower: float | None = None
    precision_ci_upper: float | None = None
    recall: float | None = None
    f1: float | None = None
    reportable: bool | None = None
    deployment_eligible: bool | None = None

    def __post_init__(self) -> None:
        for name in ("evaluation_id", "dataset_version", "model_version", "label_source"):
            _text(getattr(self, name), name)
        _bool(self.synthetic, "synthetic")
        _number(self.precision, "precision")
        _integer(self.labelled_pair_count, "labelled_pair_count", minimum=1)
        if self.evaluated_at.tzinfo is None or self.evaluated_at.utcoffset() is None:
            raise EntityResolutionContractError("evaluated_at must include a timezone")
        if self.schema_version not in {
            ER_EVALUATION_SCHEMA_VERSION,
            ER_EVALUATION_SCHEMA_VERSION_V2,
        }:
            raise EntityResolutionContractError("unsupported entity-resolution evaluation schema")
        if self.schema_version == ER_EVALUATION_SCHEMA_VERSION_V2:
            for name in (
                "artifact_sha256",
                "review_queue_sha256",
                "frozen_test_groups_sha256",
            ):
                _sha256(getattr(self, name), name)
            _number(self.auto_match_threshold, "auto_match_threshold")
            numerator = _integer(self.precision_numerator, "precision_numerator")
            denominator = _integer(self.precision_denominator, "precision_denominator", minimum=1)
            if numerator > denominator:
                raise EntityResolutionContractError(
                    "precision_numerator cannot exceed precision_denominator"
                )
            lower = _number(self.precision_ci_lower, "precision_ci_lower")
            upper = _number(self.precision_ci_upper, "precision_ci_upper")
            if lower > upper:
                raise EntityResolutionContractError("precision confidence interval is inverted")
            _number(self.recall, "recall")
            _number(self.f1, "f1")
            _bool(self.reportable, "reportable")
            _bool(self.deployment_eligible, "deployment_eligible")

    def blockers(
        self,
        *,
        minimum_precision: float,
        minimum_labelled_pairs: int,
    ) -> tuple[str, ...]:
        blockers: list[str] = []
        if self.schema_version != ER_EVALUATION_SCHEMA_VERSION_V2:
            blockers.append("entity-resolution evaluation schema is not production v2")
        if self.synthetic:
            blockers.append("entity-resolution evaluation is synthetic")
        if self.label_source != "human_reviewed":
            blockers.append("entity-resolution evaluation is not human-labelled")
        if self.precision < minimum_precision:
            blockers.append(
                f"entity-resolution precision={self.precision:.4f} below "
                f"minimum={minimum_precision:.4f}"
            )
        if self.labelled_pair_count < minimum_labelled_pairs:
            blockers.append(
                f"entity-resolution labelled_pair_count={self.labelled_pair_count} below "
                f"minimum={minimum_labelled_pairs}"
            )
        if self.schema_version == ER_EVALUATION_SCHEMA_VERSION_V2:
            if self.precision_ci_lower is None or self.precision_ci_lower < minimum_precision:
                blockers.append(
                    "entity-resolution precision confidence lower bound is below minimum"
                )
            if self.precision_denominator is None or self.precision_numerator is None:
                blockers.append("entity-resolution precision evidence counts are missing")
            elif abs(self.precision_numerator / self.precision_denominator - self.precision) > 1e-9:
                blockers.append("entity-resolution precision does not match evidence counts")
        return tuple(blockers)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "evaluation_id": self.evaluation_id,
            "dataset_version": self.dataset_version,
            "model_version": self.model_version,
            "label_source": self.label_source,
            "synthetic": self.synthetic,
            "precision": self.precision,
            "labelled_pair_count": self.labelled_pair_count,
            "evaluated_at": self.evaluated_at.isoformat(),
        }
        if self.schema_version == ER_EVALUATION_SCHEMA_VERSION_V2:
            payload.update(
                {
                    "artifact_sha256": self.artifact_sha256,
                    "review_queue_sha256": self.review_queue_sha256,
                    "frozen_test_groups_sha256": self.frozen_test_groups_sha256,
                    "auto_match_threshold": self.auto_match_threshold,
                    "precision_numerator": self.precision_numerator,
                    "precision_denominator": self.precision_denominator,
                    "precision_ci_lower": self.precision_ci_lower,
                    "precision_ci_upper": self.precision_ci_upper,
                    "recall": self.recall,
                    "f1": self.f1,
                    "reportable": self.reportable,
                    "deployment_eligible": self.deployment_eligible,
                }
            )
        return payload


_EVALUATION_BASE_FIELDS = frozenset(
    {
        "schema_version",
        "evaluation_id",
        "dataset_version",
        "model_version",
        "label_source",
        "synthetic",
        "precision",
        "labelled_pair_count",
        "evaluated_at",
    }
)
_EVALUATION_V2_FIELDS = _EVALUATION_BASE_FIELDS | {
    "artifact_sha256",
    "review_queue_sha256",
    "frozen_test_groups_sha256",
    "auto_match_threshold",
    "precision_numerator",
    "precision_denominator",
    "precision_ci_lower",
    "precision_ci_upper",
    "recall",
    "f1",
    "reportable",
    "deployment_eligible",
}


def load_entity_resolution_evaluation(
    path: str | Path | None,
) -> EntityResolutionProductionEvaluation | None:
    if path is None:
        return None
    _, payload, _ = _load_json_object(path)
    schema = payload.get("schema_version")
    if schema not in {ER_EVALUATION_SCHEMA_VERSION, ER_EVALUATION_SCHEMA_VERSION_V2}:
        raise EntityResolutionContractError("unsupported entity-resolution evaluation schema")
    _require_exact_fields(
        payload,
        _EVALUATION_V2_FIELDS
        if schema == ER_EVALUATION_SCHEMA_VERSION_V2
        else _EVALUATION_BASE_FIELDS,
        contract="entity-resolution evaluation",
    )
    return EntityResolutionProductionEvaluation(
        evaluation_id=_text(payload["evaluation_id"], "evaluation_id"),
        dataset_version=_text(payload["dataset_version"], "dataset_version"),
        model_version=_text(payload["model_version"], "model_version"),
        label_source=_text(payload["label_source"], "label_source"),
        synthetic=_bool(payload["synthetic"], "synthetic"),
        precision=_number(payload["precision"], "precision"),
        labelled_pair_count=_integer(
            payload["labelled_pair_count"], "labelled_pair_count", minimum=1
        ),
        evaluated_at=_timestamp(payload["evaluated_at"], "evaluated_at"),
        schema_version=cast(str, schema),
        artifact_sha256=(
            _sha256(payload["artifact_sha256"], "artifact_sha256")
            if schema == ER_EVALUATION_SCHEMA_VERSION_V2
            else None
        ),
        review_queue_sha256=(
            _sha256(payload["review_queue_sha256"], "review_queue_sha256")
            if schema == ER_EVALUATION_SCHEMA_VERSION_V2
            else None
        ),
        frozen_test_groups_sha256=(
            _sha256(payload["frozen_test_groups_sha256"], "frozen_test_groups_sha256")
            if schema == ER_EVALUATION_SCHEMA_VERSION_V2
            else None
        ),
        auto_match_threshold=(
            _number(payload["auto_match_threshold"], "auto_match_threshold")
            if schema == ER_EVALUATION_SCHEMA_VERSION_V2
            else None
        ),
        precision_numerator=(
            _integer(payload["precision_numerator"], "precision_numerator")
            if schema == ER_EVALUATION_SCHEMA_VERSION_V2
            else None
        ),
        precision_denominator=(
            _integer(payload["precision_denominator"], "precision_denominator", minimum=1)
            if schema == ER_EVALUATION_SCHEMA_VERSION_V2
            else None
        ),
        precision_ci_lower=(
            _number(payload["precision_ci_lower"], "precision_ci_lower")
            if schema == ER_EVALUATION_SCHEMA_VERSION_V2
            else None
        ),
        precision_ci_upper=(
            _number(payload["precision_ci_upper"], "precision_ci_upper")
            if schema == ER_EVALUATION_SCHEMA_VERSION_V2
            else None
        ),
        recall=(
            _number(payload["recall"], "recall")
            if schema == ER_EVALUATION_SCHEMA_VERSION_V2
            else None
        ),
        f1=(_number(payload["f1"], "f1") if schema == ER_EVALUATION_SCHEMA_VERSION_V2 else None),
        reportable=(
            _bool(payload["reportable"], "reportable")
            if schema == ER_EVALUATION_SCHEMA_VERSION_V2
            else None
        ),
        deployment_eligible=(
            _bool(payload["deployment_eligible"], "deployment_eligible")
            if schema == ER_EVALUATION_SCHEMA_VERSION_V2
            else None
        ),
    )


_POLICY_FIELDS = frozenset(
    {
        "schema_version",
        "policy_sha256",
        "policy_id",
        "claim_scope",
        "territory",
        "required_label_source",
        "required_model_type",
        "required_matcher_decision_version",
        "required_serving_projection_version",
        "minimum_precision",
        "minimum_labelled_pairs",
        "minimum_auto_matches",
        "minimum_recall",
        "minimum_f1",
        "auto_match_threshold",
        "manual_review_threshold",
        "max_candidates",
        "minimum_text_score",
        "minimum_auto_margin",
        "evidence_candidate_limit",
        "minimum_products",
        "minimum_products_per_category",
        "minimum_mapping_rate",
        "minimum_critical_field_rate",
        "require_complete_priced_coverage",
        "require_complete_in_stock_coverage",
        "require_complete_product_provenance",
        "require_complete_offer_provenance",
        "require_explicit_offer_rights",
        "require_production_offer_rights",
        "require_complete_listing_provenance",
        "require_promoted_entity_resolution_model",
    }
)


@dataclass(frozen=True, slots=True)
class EntityResolutionPolicy:
    """Versioned production thresholds used by matching and catalogue readiness."""

    policy_sha256: str
    policy_id: str
    claim_scope: str
    territory: str
    required_label_source: str
    required_model_type: str
    required_matcher_decision_version: str
    required_serving_projection_version: str
    minimum_precision: float
    minimum_labelled_pairs: int
    minimum_auto_matches: int
    minimum_recall: float
    minimum_f1: float
    auto_match_threshold: float
    manual_review_threshold: float
    max_candidates: int
    minimum_text_score: float
    minimum_auto_margin: float
    evidence_candidate_limit: int
    minimum_products: int
    minimum_products_per_category: int
    minimum_mapping_rate: float
    minimum_critical_field_rate: float
    require_complete_priced_coverage: bool
    require_complete_in_stock_coverage: bool
    require_complete_product_provenance: bool
    require_complete_offer_provenance: bool
    require_explicit_offer_rights: bool
    require_production_offer_rights: bool
    require_complete_listing_provenance: bool
    require_promoted_entity_resolution_model: bool

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EntityResolutionPolicy:
        _require_exact_fields(payload, _POLICY_FIELDS, contract="entity-resolution policy")
        if payload.get("schema_version") != ER_POLICY_SCHEMA_VERSION:
            raise EntityResolutionContractError("unsupported entity-resolution policy schema")
        expected = _canonical_json_sha256(payload, omit="policy_sha256")
        actual = _sha256(payload["policy_sha256"], "policy_sha256")
        if actual != expected:
            raise EntityResolutionContractError(
                "entity-resolution policy SHA-256 does not match its content"
            )
        auto = _number(payload["auto_match_threshold"], "auto_match_threshold")
        manual = _number(payload["manual_review_threshold"], "manual_review_threshold")
        if manual > auto:
            raise EntityResolutionContractError(
                "manual_review_threshold cannot exceed auto_match_threshold"
            )
        territory = _text(payload["territory"], "territory")
        if territory != territory.upper():
            raise EntityResolutionContractError("territory must use canonical uppercase form")
        result = cls(
            policy_sha256=actual,
            policy_id=_text(payload["policy_id"], "policy_id"),
            claim_scope=_text(payload["claim_scope"], "claim_scope"),
            territory=territory,
            required_label_source=_text(payload["required_label_source"], "required_label_source"),
            required_model_type=_text(payload["required_model_type"], "required_model_type"),
            required_matcher_decision_version=_text(
                payload["required_matcher_decision_version"],
                "required_matcher_decision_version",
            ),
            required_serving_projection_version=_text(
                payload["required_serving_projection_version"],
                "required_serving_projection_version",
            ),
            minimum_precision=_number(payload["minimum_precision"], "minimum_precision"),
            minimum_labelled_pairs=_integer(
                payload["minimum_labelled_pairs"], "minimum_labelled_pairs", minimum=1
            ),
            minimum_auto_matches=_integer(
                payload["minimum_auto_matches"], "minimum_auto_matches", minimum=1
            ),
            minimum_recall=_number(payload["minimum_recall"], "minimum_recall"),
            minimum_f1=_number(payload["minimum_f1"], "minimum_f1"),
            auto_match_threshold=auto,
            manual_review_threshold=manual,
            max_candidates=_integer(payload["max_candidates"], "max_candidates", minimum=1),
            minimum_text_score=_number(payload["minimum_text_score"], "minimum_text_score"),
            minimum_auto_margin=_number(payload["minimum_auto_margin"], "minimum_auto_margin"),
            evidence_candidate_limit=_integer(
                payload["evidence_candidate_limit"], "evidence_candidate_limit", minimum=1
            ),
            minimum_products=_integer(payload["minimum_products"], "minimum_products"),
            minimum_products_per_category=_integer(
                payload["minimum_products_per_category"], "minimum_products_per_category"
            ),
            minimum_mapping_rate=_number(payload["minimum_mapping_rate"], "minimum_mapping_rate"),
            minimum_critical_field_rate=_number(
                payload["minimum_critical_field_rate"], "minimum_critical_field_rate"
            ),
            require_complete_priced_coverage=_bool(
                payload["require_complete_priced_coverage"],
                "require_complete_priced_coverage",
            ),
            require_complete_in_stock_coverage=_bool(
                payload["require_complete_in_stock_coverage"],
                "require_complete_in_stock_coverage",
            ),
            require_complete_product_provenance=_bool(
                payload["require_complete_product_provenance"],
                "require_complete_product_provenance",
            ),
            require_complete_offer_provenance=_bool(
                payload["require_complete_offer_provenance"],
                "require_complete_offer_provenance",
            ),
            require_explicit_offer_rights=_bool(
                payload["require_explicit_offer_rights"], "require_explicit_offer_rights"
            ),
            require_production_offer_rights=_bool(
                payload["require_production_offer_rights"],
                "require_production_offer_rights",
            ),
            require_complete_listing_provenance=_bool(
                payload["require_complete_listing_provenance"],
                "require_complete_listing_provenance",
            ),
            require_promoted_entity_resolution_model=_bool(
                payload["require_promoted_entity_resolution_model"],
                "require_promoted_entity_resolution_model",
            ),
        )
        result.assert_production_floors()
        return result

    def assert_production_floors(self) -> None:
        """Reject a signed-looking policy that weakens non-negotiable release gates."""

        blockers: list[str] = []
        numeric_floors: tuple[tuple[str, int | float, int | float], ...] = (
            ("minimum_precision", self.minimum_precision, ER_MINIMUM_PRODUCTION_PRECISION),
            (
                "minimum_labelled_pairs",
                self.minimum_labelled_pairs,
                ER_MINIMUM_PRODUCTION_LABELLED_PAIRS,
            ),
            (
                "minimum_auto_matches",
                self.minimum_auto_matches,
                ER_MINIMUM_PRODUCTION_AUTO_MATCHES,
            ),
            ("minimum_recall", self.minimum_recall, ER_MINIMUM_PRODUCTION_RECALL),
            ("minimum_f1", self.minimum_f1, ER_MINIMUM_PRODUCTION_F1),
            (
                "auto_match_threshold",
                self.auto_match_threshold,
                ER_MINIMUM_PRODUCTION_AUTO_MATCH_THRESHOLD,
            ),
            ("minimum_products", self.minimum_products, ER_MINIMUM_PRODUCTION_PRODUCTS),
            ("minimum_products_per_category", self.minimum_products_per_category, 1),
            (
                "minimum_mapping_rate",
                self.minimum_mapping_rate,
                ER_MINIMUM_PRODUCTION_MAPPING_RATE,
            ),
            (
                "minimum_critical_field_rate",
                self.minimum_critical_field_rate,
                ER_MINIMUM_PRODUCTION_CRITICAL_FIELD_RATE,
            ),
        )
        for name, value, floor in numeric_floors:
            if value < floor:
                blockers.append(f"{name}={value} is below {floor}")
        required_true = (
            "require_complete_priced_coverage",
            "require_complete_in_stock_coverage",
            "require_complete_product_provenance",
            "require_complete_offer_provenance",
            "require_explicit_offer_rights",
            "require_production_offer_rights",
            "require_complete_listing_provenance",
            "require_promoted_entity_resolution_model",
        )
        blockers.extend(name for name in required_true if getattr(self, name) is not True)
        if self.territory != ER_PRODUCTION_TERRITORY:
            blockers.append(f"territory={self.territory!r} is not {ER_PRODUCTION_TERRITORY!r}")
        if blockers:
            raise EntityResolutionContractError(
                "entity-resolution production policy weakens non-negotiable floors: "
                + "; ".join(blockers)
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": ER_POLICY_SCHEMA_VERSION,
            "policy_sha256": self.policy_sha256,
            "policy_id": self.policy_id,
            "claim_scope": self.claim_scope,
            "territory": self.territory,
            "required_label_source": self.required_label_source,
            "required_model_type": self.required_model_type,
            "required_matcher_decision_version": self.required_matcher_decision_version,
            "required_serving_projection_version": self.required_serving_projection_version,
            "minimum_precision": self.minimum_precision,
            "minimum_labelled_pairs": self.minimum_labelled_pairs,
            "minimum_auto_matches": self.minimum_auto_matches,
            "minimum_recall": self.minimum_recall,
            "minimum_f1": self.minimum_f1,
            "auto_match_threshold": self.auto_match_threshold,
            "manual_review_threshold": self.manual_review_threshold,
            "max_candidates": self.max_candidates,
            "minimum_text_score": self.minimum_text_score,
            "minimum_auto_margin": self.minimum_auto_margin,
            "evidence_candidate_limit": self.evidence_candidate_limit,
            "minimum_products": self.minimum_products,
            "minimum_products_per_category": self.minimum_products_per_category,
            "minimum_mapping_rate": self.minimum_mapping_rate,
            "minimum_critical_field_rate": self.minimum_critical_field_rate,
            "require_complete_priced_coverage": self.require_complete_priced_coverage,
            "require_complete_in_stock_coverage": self.require_complete_in_stock_coverage,
            "require_complete_product_provenance": self.require_complete_product_provenance,
            "require_complete_offer_provenance": self.require_complete_offer_provenance,
            "require_explicit_offer_rights": self.require_explicit_offer_rights,
            "require_production_offer_rights": self.require_production_offer_rights,
            "require_complete_listing_provenance": self.require_complete_listing_provenance,
            "require_promoted_entity_resolution_model": (
                self.require_promoted_entity_resolution_model
            ),
        }

    def matcher_kwargs(self) -> EntityResolutionMatcherKwargs:
        return {
            "max_candidates": self.max_candidates,
            "minimum_text_score": self.minimum_text_score,
            "minimum_auto_margin": self.minimum_auto_margin,
            "evidence_candidate_limit": self.evidence_candidate_limit,
        }

    def production_catalog_policy_kwargs(self) -> ProductionCatalogPolicyKwargs:
        return {
            "minimum_products": self.minimum_products,
            "minimum_products_per_category": self.minimum_products_per_category,
            "minimum_mapping_rate": self.minimum_mapping_rate,
            "minimum_critical_field_rate": self.minimum_critical_field_rate,
            "require_complete_priced_coverage": self.require_complete_priced_coverage,
            "require_complete_in_stock_coverage": self.require_complete_in_stock_coverage,
            "require_complete_product_provenance": self.require_complete_product_provenance,
            "require_complete_offer_provenance": self.require_complete_offer_provenance,
            "require_explicit_offer_rights": self.require_explicit_offer_rights,
            "require_production_offer_rights": self.require_production_offer_rights,
            "require_complete_listing_provenance": self.require_complete_listing_provenance,
            "minimum_er_precision": self.minimum_precision,
            "minimum_er_labelled_pairs": self.minimum_labelled_pairs,
            "require_promoted_entity_resolution_model": (
                self.require_promoted_entity_resolution_model
            ),
        }


def seal_entity_resolution_policy(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy with its canonical content digest populated."""

    result = dict(payload)
    result["policy_sha256"] = _canonical_json_sha256(result, omit="policy_sha256")
    EntityResolutionPolicy.from_dict(result)
    return result


def load_entity_resolution_policy(path: str | Path) -> EntityResolutionPolicy:
    _, payload, _ = _load_json_object(path)
    return EntityResolutionPolicy.from_dict(payload)


_RIGHTS_FIELDS = frozenset(
    {
        "schema_version",
        "rights_sha256",
        "approval_id",
        "decision",
        "dataset_version",
        "model_version",
        "model_release_sha256",
        "evaluation_sha256",
        "policy_sha256",
        "review_queue_sha256",
        "frozen_test_groups_sha256",
        "approved_uses",
        "territories",
        "approved_by",
        "approved_at",
        "expires_at",
        "source_contract_sha256s",
        "evidence_references",
    }
)


@dataclass(frozen=True, slots=True)
class EntityResolutionRightsApproval:
    """Rights/compliance approval over one exact model, evaluation, and policy."""

    rights_sha256: str
    approval_id: str
    decision: str
    dataset_version: str
    model_version: str
    model_release_sha256: str
    evaluation_sha256: str
    policy_sha256: str
    review_queue_sha256: str
    frozen_test_groups_sha256: str
    approved_uses: tuple[str, ...]
    territories: tuple[str, ...]
    approved_by: str
    approved_at: datetime
    expires_at: datetime | None
    source_contract_sha256s: tuple[str, ...]
    evidence_references: tuple[str, ...]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EntityResolutionRightsApproval:
        _require_exact_fields(payload, _RIGHTS_FIELDS, contract="ER rights approval")
        if payload.get("schema_version") != ER_RIGHTS_APPROVAL_SCHEMA_VERSION:
            raise EntityResolutionContractError("unsupported ER rights approval schema")
        expected = _canonical_json_sha256(payload, omit="rights_sha256")
        actual = _sha256(payload["rights_sha256"], "rights_sha256")
        if actual != expected:
            raise EntityResolutionContractError(
                "ER rights approval SHA-256 does not match its content"
            )
        contracts = _sorted_unique_text_tuple(
            payload["source_contract_sha256s"], "source_contract_sha256s"
        )
        for item in contracts:
            _sha256(item, "source_contract_sha256s item")
        territories = _sorted_unique_text_tuple(payload["territories"], "territories")
        if any(item != item.upper() for item in territories):
            raise EntityResolutionContractError("territories must use canonical uppercase form")
        approved_at = _timestamp(payload["approved_at"], "approved_at")
        expires_at = _optional_timestamp(payload["expires_at"], "expires_at")
        if expires_at is not None and expires_at <= approved_at:
            raise EntityResolutionContractError("rights approval expiry must follow approval")
        return cls(
            rights_sha256=actual,
            approval_id=_text(payload["approval_id"], "approval_id"),
            decision=_text(payload["decision"], "decision"),
            dataset_version=_text(payload["dataset_version"], "dataset_version"),
            model_version=_text(payload["model_version"], "model_version"),
            model_release_sha256=_sha256(payload["model_release_sha256"], "model_release_sha256"),
            evaluation_sha256=_sha256(payload["evaluation_sha256"], "evaluation_sha256"),
            policy_sha256=_sha256(payload["policy_sha256"], "policy_sha256"),
            review_queue_sha256=_sha256(payload["review_queue_sha256"], "review_queue_sha256"),
            frozen_test_groups_sha256=_sha256(
                payload["frozen_test_groups_sha256"], "frozen_test_groups_sha256"
            ),
            approved_uses=_sorted_unique_text_tuple(payload["approved_uses"], "approved_uses"),
            territories=territories,
            approved_by=_text(payload["approved_by"], "approved_by"),
            approved_at=approved_at,
            expires_at=expires_at,
            source_contract_sha256s=contracts,
            evidence_references=_sorted_unique_text_tuple(
                payload["evidence_references"], "evidence_references"
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": ER_RIGHTS_APPROVAL_SCHEMA_VERSION,
            "rights_sha256": self.rights_sha256,
            "approval_id": self.approval_id,
            "decision": self.decision,
            "dataset_version": self.dataset_version,
            "model_version": self.model_version,
            "model_release_sha256": self.model_release_sha256,
            "evaluation_sha256": self.evaluation_sha256,
            "policy_sha256": self.policy_sha256,
            "review_queue_sha256": self.review_queue_sha256,
            "frozen_test_groups_sha256": self.frozen_test_groups_sha256,
            "approved_uses": list(self.approved_uses),
            "territories": list(self.territories),
            "approved_by": self.approved_by,
            "approved_at": self.approved_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at is not None else None,
            "source_contract_sha256s": list(self.source_contract_sha256s),
            "evidence_references": list(self.evidence_references),
        }

    def assert_active(self, *, territory: str, as_of: datetime | None = None) -> None:
        now = (as_of or datetime.now(UTC)).astimezone(UTC)
        if self.decision != "approved":
            raise EntityResolutionContractError("ER rights decision is not approved")
        if now < self.approved_at:
            raise EntityResolutionContractError("ER rights approval is not yet effective")
        if self.expires_at is not None and now >= self.expires_at:
            raise EntityResolutionContractError("ER rights approval has expired")
        if territory.strip().upper() not in self.territories:
            raise EntityResolutionContractError(
                "ER rights approval does not cover the production territory"
            )
        missing = sorted(ER_REQUIRED_APPROVED_USES - set(self.approved_uses))
        if missing:
            raise EntityResolutionContractError(
                f"ER rights approval is missing required uses: {missing}"
            )


def seal_entity_resolution_rights_approval(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["rights_sha256"] = _canonical_json_sha256(result, omit="rights_sha256")
    EntityResolutionRightsApproval.from_dict(result)
    return result


def load_entity_resolution_rights_approval(
    path: str | Path,
) -> EntityResolutionRightsApproval:
    _, payload, _ = _load_json_object(path)
    return EntityResolutionRightsApproval.from_dict(payload)


@dataclass(frozen=True, slots=True)
class EntityResolutionReleaseIdentity:
    """All immutable identities needed to reproduce one authorized ER runtime."""

    artifact_core_sha256: str
    model_file_sha256: str
    metadata_sha256: str
    calibrator_sha256: str
    serving_evidence_sha256: str
    model_release_sha256: str
    evaluation_sha256: str
    policy_sha256: str
    rights_sha256: str
    binding_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "artifact_core_sha256",
            "model_file_sha256",
            "metadata_sha256",
            "calibrator_sha256",
            "serving_evidence_sha256",
            "model_release_sha256",
            "evaluation_sha256",
            "policy_sha256",
            "rights_sha256",
            "binding_sha256",
        ):
            _sha256(getattr(self, name), name)
        if self.binding_sha256 != _canonical_json_sha256(self.content_payload(), omit=None):
            raise EntityResolutionContractError(
                "ER release binding SHA-256 does not match its identities"
            )

    def content_payload(self) -> dict[str, str]:
        return {
            "schema_version": ER_RELEASE_IDENTITY_SCHEMA_VERSION,
            "artifact_core_sha256": self.artifact_core_sha256,
            "model_file_sha256": self.model_file_sha256,
            "metadata_sha256": self.metadata_sha256,
            "calibrator_sha256": self.calibrator_sha256,
            "serving_evidence_sha256": self.serving_evidence_sha256,
            "model_release_sha256": self.model_release_sha256,
            "evaluation_sha256": self.evaluation_sha256,
            "policy_sha256": self.policy_sha256,
            "rights_sha256": self.rights_sha256,
        }

    def to_dict(self) -> dict[str, str]:
        return {**self.content_payload(), "binding_sha256": self.binding_sha256}


def build_entity_resolution_release_identity(
    *,
    artifact_core_sha256: str,
    model_file_sha256: str,
    metadata_sha256: str,
    calibrator_sha256: str,
    serving_evidence_sha256: str,
    model_release_sha256: str,
    evaluation_sha256: str,
    policy_sha256: str,
    rights_sha256: str,
) -> EntityResolutionReleaseIdentity:
    payload = {
        "schema_version": ER_RELEASE_IDENTITY_SCHEMA_VERSION,
        "artifact_core_sha256": artifact_core_sha256,
        "model_file_sha256": model_file_sha256,
        "metadata_sha256": metadata_sha256,
        "calibrator_sha256": calibrator_sha256,
        "serving_evidence_sha256": serving_evidence_sha256,
        "model_release_sha256": model_release_sha256,
        "evaluation_sha256": evaluation_sha256,
        "policy_sha256": policy_sha256,
        "rights_sha256": rights_sha256,
    }
    return EntityResolutionReleaseIdentity(
        artifact_core_sha256=artifact_core_sha256,
        model_file_sha256=model_file_sha256,
        metadata_sha256=metadata_sha256,
        calibrator_sha256=calibrator_sha256,
        serving_evidence_sha256=serving_evidence_sha256,
        model_release_sha256=model_release_sha256,
        evaluation_sha256=evaluation_sha256,
        policy_sha256=policy_sha256,
        rights_sha256=rights_sha256,
        binding_sha256=_canonical_json_sha256(payload),
    )
