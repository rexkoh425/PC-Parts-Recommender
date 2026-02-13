"""Fail-closed loading for entity-resolution serving artifacts.

The model files alone are not deployment authority.  A serving artifact must also carry
evidence produced by the human-review training workflow.  The evidence binds the model
bytes, feature contract, dataset version, source-use policy, and promotion decision.
Synthetic and cross-domain transfer artifacts are intentionally ineligible here, even
when a caller opts into an otherwise non-promoted human diagnostic.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn, Protocol, cast

from .features import FEATURE_NAMES
from .models import ARTIFACT_FORMAT_VERSION, LightGBMEntityResolver, load_entity_resolver
from .release_contracts import (
    ER_EVALUATION_SCHEMA_VERSION_V2,
    EntityResolutionPolicy,
    EntityResolutionProductionEvaluation,
    EntityResolutionReleaseIdentity,
    EntityResolutionRightsApproval,
    build_entity_resolution_release_identity,
    entity_resolution_file_sha256,
    load_entity_resolution_evaluation,
    load_entity_resolution_policy,
    load_entity_resolution_rights_approval,
)

ER_SERVING_EVIDENCE_SCHEMA_VERSION = "pc-build-recommender.er-serving-evidence.v1"
ER_SERVING_EVIDENCE_FILENAME = "serving_evidence.json"
ER_HUMAN_LABEL_SOURCE = "attributable_human_reviews"
ER_PRODUCTION_CLAIM_SCOPE = "pc_retailer_catalog"
ER_CATALOG_MATCHER_DECISION_VERSION = "catalog-er-decision-v1"
ER_SERVING_PROJECTION_VERSION = "governed-offer-er-projection-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_JSON_ARTIFACT_BYTES = 4 * 1024 * 1024


class EntityResolutionArtifactError(ValueError):
    """Raised when an artifact cannot be trusted for catalogue matching."""


class ProductionEntityResolutionEvaluation(Protocol):
    """Minimal production-evaluation contract used without importing the catalogue."""

    @property
    def schema_version(self) -> str: ...

    @property
    def dataset_version(self) -> str: ...

    @property
    def model_version(self) -> str: ...

    @property
    def label_source(self) -> str: ...

    @property
    def synthetic(self) -> bool: ...

    @property
    def precision(self) -> float: ...

    @property
    def labelled_pair_count(self) -> int: ...

    @property
    def artifact_sha256(self) -> str | None: ...

    @property
    def review_queue_sha256(self) -> str | None: ...

    @property
    def frozen_test_groups_sha256(self) -> str | None: ...

    @property
    def auto_match_threshold(self) -> float | None: ...

    @property
    def precision_numerator(self) -> int | None: ...

    @property
    def precision_denominator(self) -> int | None: ...

    @property
    def precision_ci_lower(self) -> float | None: ...

    @property
    def precision_ci_upper(self) -> float | None: ...

    @property
    def recall(self) -> float | None: ...

    @property
    def f1(self) -> float | None: ...

    @property
    def reportable(self) -> bool | None: ...

    @property
    def deployment_eligible(self) -> bool | None: ...


def _reject_json_constant(value: str) -> NoReturn:
    raise EntityResolutionArtifactError(f"non-finite JSON number is forbidden: {value}")


def _reject_json_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EntityResolutionArtifactError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size > _MAX_JSON_ARTIFACT_BYTES:
        raise EntityResolutionArtifactError(
            f"{path.name} exceeds the {_MAX_JSON_ARTIFACT_BYTES}-byte JSON artifact limit"
        )
    try:
        payload: Any = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_json_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EntityResolutionArtifactError(f"invalid UTF-8 JSON in {path.name}") from error
    if not isinstance(payload, dict):
        raise EntityResolutionArtifactError(f"{path.name} must contain a JSON object")
    return cast(dict[str, Any], payload)


def _required_text(payload: Mapping[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise EntityResolutionArtifactError(
            f"entity-resolution serving evidence requires {field_name}"
        )
    return value.strip()


def _required_bool(payload: Mapping[str, Any], field_name: str) -> bool:
    value = payload.get(field_name)
    if type(value) is not bool:
        raise EntityResolutionArtifactError(
            f"entity-resolution serving evidence requires boolean {field_name}"
        )
    return value


def _feature_contract_sha256() -> str:
    payload = json.dumps(FEATURE_NAMES, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _model_file(artifact_path: Path, metadata: Mapping[str, Any]) -> Path:
    relative_name = _required_text(metadata, "model_file")
    candidate = (artifact_path / relative_name).resolve()
    if candidate.parent != artifact_path:
        raise EntityResolutionArtifactError("entity-resolution model_file must stay in artifact")
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def _hash_files(artifact_path: Path, files: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for item in files:
        relative = item.relative_to(artifact_path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(item.stat().st_size.to_bytes(8, "big"))
        with item.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def entity_resolution_artifact_sha256(path: str | Path) -> str:
    """Hash the model metadata and bytes in a stable order."""

    artifact_path = Path(path).resolve()
    metadata_path = artifact_path / "metadata.json"
    metadata = _json_object(metadata_path)
    model_path = _model_file(artifact_path, metadata)
    return _hash_files(artifact_path, (metadata_path, model_path))


def entity_resolution_release_sha256(path: str | Path) -> str:
    """Hash model bytes and their serving authority as one immutable release."""

    artifact_path = Path(path).resolve()
    metadata_path = artifact_path / "metadata.json"
    metadata = _json_object(metadata_path)
    model_path = _model_file(artifact_path, metadata)
    evidence_path = artifact_path / ER_SERVING_EVIDENCE_FILENAME
    if not evidence_path.is_file():
        raise FileNotFoundError(evidence_path)
    return _hash_files(artifact_path, (metadata_path, model_path, evidence_path))


def entity_resolution_model_version(path: str | Path) -> str:
    """Return the content-derived version used by decisions and evaluation binding."""

    return f"er-lightgbm-{entity_resolution_release_sha256(path)[:16]}"


@dataclass(frozen=True, slots=True)
class EntityResolutionServingEvidence:
    """Inspectable provenance and promotion facts shipped beside one model."""

    artifact_core_sha256: str
    dataset_version: str
    label_source: str
    claim_scope: str
    synthetic_rows: int
    transfer_only: bool
    deployment_eligible: bool
    source_training_eligible: bool
    source_published_metrics_eligible: bool
    source_model_serving_eligible: bool
    listing_source: str
    catalogue_source: str
    source_scope_note: str
    review_queue_sha256: str
    frozen_test_groups_sha256: str
    feature_contract_sha256: str
    matcher_decision_version: str
    serving_projection_version: str
    auto_match_threshold: float
    manual_review_threshold: float

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EntityResolutionServingEvidence:
        if payload.get("schema_version") != ER_SERVING_EVIDENCE_SCHEMA_VERSION:
            raise EntityResolutionArtifactError(
                "unsupported entity-resolution serving evidence schema"
            )
        synthetic_rows = payload.get("synthetic_rows")
        if type(synthetic_rows) is not int or synthetic_rows < 0:
            raise EntityResolutionArtifactError("synthetic_rows must be a non-negative integer")
        source_policy = payload.get("source_policy")
        if not isinstance(source_policy, Mapping):
            raise EntityResolutionArtifactError("serving evidence requires source_policy")
        training_eligible = source_policy.get("training_eligible")
        metrics_eligible = source_policy.get("published_metrics_eligible")
        serving_eligible = source_policy.get("model_serving_eligible", False)
        if (
            type(training_eligible) is not bool
            or type(metrics_eligible) is not bool
            or type(serving_eligible) is not bool
        ):
            raise EntityResolutionArtifactError(
                "source_policy eligibility fields must be explicit booleans"
            )
        artifact_core_sha256 = _required_text(payload, "artifact_core_sha256")
        if _SHA256.fullmatch(artifact_core_sha256) is None:
            raise EntityResolutionArtifactError("artifact_core_sha256 must be lowercase SHA-256")
        thresholds = payload.get("thresholds")
        if not isinstance(thresholds, Mapping):
            raise EntityResolutionArtifactError("serving evidence requires thresholds")
        auto_match = thresholds.get("auto_match")
        manual_review = thresholds.get("manual_review")
        if not isinstance(auto_match, int | float) or isinstance(auto_match, bool):
            raise EntityResolutionArtifactError("auto_match threshold must be numeric")
        if not isinstance(manual_review, int | float) or isinstance(manual_review, bool):
            raise EntityResolutionArtifactError("manual_review threshold must be numeric")
        if not 0 <= float(manual_review) <= float(auto_match) <= 1:
            raise EntityResolutionArtifactError("invalid serving thresholds")
        dataset_version = _required_text(payload, "dataset_version")
        if _required_text(source_policy, "data_version") != dataset_version:
            raise EntityResolutionArtifactError(
                "source_policy data_version must match serving evidence"
            )
        digest_fields = {
            name: _required_text(payload, name)
            for name in (
                "review_queue_sha256",
                "frozen_test_groups_sha256",
                "feature_contract_sha256",
            )
        }
        if any(_SHA256.fullmatch(value) is None for value in digest_fields.values()):
            raise EntityResolutionArtifactError("serving evidence contains invalid SHA-256")
        return cls(
            artifact_core_sha256=artifact_core_sha256,
            dataset_version=dataset_version,
            label_source=_required_text(payload, "label_source"),
            claim_scope=_required_text(payload, "claim_scope"),
            synthetic_rows=synthetic_rows,
            transfer_only=_required_bool(payload, "transfer_only"),
            deployment_eligible=_required_bool(payload, "deployment_eligible"),
            source_training_eligible=training_eligible,
            source_published_metrics_eligible=metrics_eligible,
            source_model_serving_eligible=serving_eligible,
            listing_source=_required_text(source_policy, "listing_source"),
            catalogue_source=_required_text(source_policy, "catalogue_source"),
            source_scope_note=_required_text(source_policy, "scope_note"),
            review_queue_sha256=digest_fields["review_queue_sha256"],
            frozen_test_groups_sha256=digest_fields["frozen_test_groups_sha256"],
            feature_contract_sha256=digest_fields["feature_contract_sha256"],
            matcher_decision_version=_required_text(payload, "matcher_decision_version"),
            serving_projection_version=_required_text(payload, "serving_projection_version"),
            auto_match_threshold=float(auto_match),
            manual_review_threshold=float(manual_review),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": ER_SERVING_EVIDENCE_SCHEMA_VERSION,
            "artifact_core_sha256": self.artifact_core_sha256,
            "dataset_version": self.dataset_version,
            "label_source": self.label_source,
            "claim_scope": self.claim_scope,
            "synthetic_rows": self.synthetic_rows,
            "transfer_only": self.transfer_only,
            "deployment_eligible": self.deployment_eligible,
            "source_policy": {
                "training_eligible": self.source_training_eligible,
                "published_metrics_eligible": self.source_published_metrics_eligible,
                "model_serving_eligible": self.source_model_serving_eligible,
                "listing_source": self.listing_source,
                "catalogue_source": self.catalogue_source,
                "data_version": self.dataset_version,
                "scope_note": self.source_scope_note,
            },
            "thresholds": {
                "auto_match": self.auto_match_threshold,
                "manual_review": self.manual_review_threshold,
            },
            "review_queue_sha256": self.review_queue_sha256,
            "frozen_test_groups_sha256": self.frozen_test_groups_sha256,
            "feature_contract_sha256": self.feature_contract_sha256,
            "matcher_decision_version": self.matcher_decision_version,
            "serving_projection_version": self.serving_projection_version,
        }


@dataclass(frozen=True, slots=True)
class EntityResolutionRuntime:
    """Loaded LightGBM resolver plus immutable provenance used in every decision."""

    resolver: LightGBMEntityResolver
    evidence: EntityResolutionServingEvidence
    artifact_path: Path
    release_sha256: str
    release_identity: EntityResolutionReleaseIdentity | None = None
    release_policy: EntityResolutionPolicy | None = None

    @property
    def model_version(self) -> str:
        return f"er-lightgbm-{self.release_sha256[:16]}"

    @property
    def production_authorized(self) -> bool:
        """Whether exact policy, evaluation, and rights artifacts authorized this runtime."""

        return self.release_identity is not None and self.release_policy is not None

    def assert_production_evaluation(
        self,
        evaluation: ProductionEntityResolutionEvaluation | None,
        *,
        minimum_precision: float,
        minimum_labelled_pairs: int,
        minimum_auto_matches: int = 25,
        minimum_recall: float = 0.94,
        minimum_f1: float = 0.96,
    ) -> None:
        """Bind the activated bytes to non-synthetic, human-reviewed evaluation."""

        if not self.production_authorized and not self.evidence.deployment_eligible:
            raise EntityResolutionArtifactError(
                "entity-resolution artifact evidence is not deployment eligible"
            )
        if evaluation is None:
            raise EntityResolutionArtifactError(
                "a production entity-resolution evaluation is required for model activation"
            )
        blockers: list[str] = []
        if evaluation.schema_version != ER_EVALUATION_SCHEMA_VERSION_V2:
            blockers.append("evaluation schema is not production v2")
        if evaluation.model_version != self.model_version:
            blockers.append("evaluation model_version does not match activated artifact")
        if evaluation.dataset_version != self.evidence.dataset_version:
            blockers.append("evaluation dataset_version does not match serving evidence")
        if evaluation.label_source != "human_reviewed":
            blockers.append("evaluation label source is not human reviewed")
        if evaluation.synthetic:
            blockers.append("evaluation is synthetic")
        if evaluation.precision < minimum_precision:
            blockers.append("evaluation precision is below the production threshold")
        if evaluation.labelled_pair_count < minimum_labelled_pairs:
            blockers.append("evaluation labelled-pair count is below the production threshold")
        if evaluation.artifact_sha256 != self.release_sha256:
            blockers.append("evaluation artifact SHA-256 does not match activated release")
        for name, value in (
            ("review_queue_sha256", evaluation.review_queue_sha256),
            ("frozen_test_groups_sha256", evaluation.frozen_test_groups_sha256),
        ):
            if value is None or _SHA256.fullmatch(value) is None:
                blockers.append(f"evaluation is missing valid {name}")
        if evaluation.review_queue_sha256 != self.evidence.review_queue_sha256:
            blockers.append("evaluation review_queue_sha256 does not match serving evidence")
        if evaluation.frozen_test_groups_sha256 != self.evidence.frozen_test_groups_sha256:
            blockers.append("evaluation frozen_test_groups_sha256 does not match serving evidence")
        if evaluation.auto_match_threshold != self.resolver.thresholds.auto_match:
            blockers.append("evaluation threshold does not match activated resolver")
        numerator = evaluation.precision_numerator
        denominator = evaluation.precision_denominator
        if numerator is None or denominator is None or denominator < minimum_auto_matches:
            blockers.append("evaluation has insufficient automatic-match support")
        elif numerator < 0 or numerator > denominator:
            blockers.append("evaluation precision counts are invalid")
        elif abs((numerator / denominator) - evaluation.precision) > 1e-9:
            blockers.append("evaluation precision does not match its evidence counts")
        if (
            evaluation.precision_ci_lower is None
            or evaluation.precision_ci_lower < minimum_precision
        ):
            blockers.append("evaluation precision confidence lower bound is below threshold")
        if evaluation.precision_ci_upper is None:
            blockers.append("evaluation precision confidence interval is missing")
        if evaluation.recall is None or evaluation.recall < minimum_recall:
            blockers.append("evaluation recall is below the production threshold")
        if evaluation.f1 is None or evaluation.f1 < minimum_f1:
            blockers.append("evaluation F1 is below the production threshold")
        if not self.production_authorized:
            if evaluation.reportable is not True:
                blockers.append("evaluation is not reportable under its source policy")
            if evaluation.deployment_eligible is not True:
                blockers.append("evaluation is not deployment eligible")
        if blockers:
            raise EntityResolutionArtifactError("; ".join(blockers))

    def authorize_for_production(
        self,
        evaluation: ProductionEntityResolutionEvaluation | None,
        *,
        minimum_precision: float,
        minimum_labelled_pairs: int,
        minimum_auto_matches: int = 25,
        minimum_recall: float = 0.94,
        minimum_f1: float = 0.96,
    ) -> EntityResolutionRuntime:
        """Revalidate an already release-authorized runtime at an ingestion boundary.

        This method is deliberately not a promotion mechanism.  The only constructor of
        production authority is :func:`load_entity_resolution_release`, which verifies the
        exact policy and rights artifacts in addition to evaluation metrics.
        """

        if not self.production_authorized:
            raise EntityResolutionArtifactError(
                "direct entity-resolution authorization is disabled; load the complete "
                "model/evaluation/policy/rights release"
            )

        self.assert_production_evaluation(
            evaluation,
            minimum_precision=minimum_precision,
            minimum_labelled_pairs=minimum_labelled_pairs,
            minimum_auto_matches=minimum_auto_matches,
            minimum_recall=minimum_recall,
            minimum_f1=minimum_f1,
        )
        return self


def build_entity_resolution_serving_evidence(
    artifact_path: str | Path,
    *,
    dataset_version: str,
    source_policy: Mapping[str, Any],
    deployment_eligible: bool,
    review_queue_sha256: str,
    frozen_test_groups_sha256: str,
    end_to_end_matcher_evaluated: bool = False,
) -> dict[str, object]:
    """Build evidence for the human-review trainer after model files are frozen."""

    resolved = Path(artifact_path).resolve()
    artifact_core_sha256 = entity_resolution_artifact_sha256(resolved)
    metadata = _json_object(resolved / "metadata.json")
    thresholds = metadata.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise EntityResolutionArtifactError("model metadata requires thresholds")
    if deployment_eligible and not end_to_end_matcher_evaluated:
        raise EntityResolutionArtifactError(
            "pairwise evaluation cannot promote the deployed catalogue matcher"
        )
    return {
        "schema_version": ER_SERVING_EVIDENCE_SCHEMA_VERSION,
        "artifact_core_sha256": artifact_core_sha256,
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "model_type": LightGBMEntityResolver.model_type,
        "dataset_version": dataset_version,
        "label_source": ER_HUMAN_LABEL_SOURCE,
        "claim_scope": ER_PRODUCTION_CLAIM_SCOPE,
        "synthetic_rows": 0,
        "transfer_only": False,
        "deployment_eligible": deployment_eligible and end_to_end_matcher_evaluated,
        "source_policy": dict(source_policy),
        "thresholds": {
            "auto_match": thresholds.get("auto_match"),
            "manual_review": thresholds.get("manual_review"),
        },
        "review_queue_sha256": review_queue_sha256,
        "frozen_test_groups_sha256": frozen_test_groups_sha256,
        "feature_contract_sha256": _feature_contract_sha256(),
        "matcher_decision_version": ER_CATALOG_MATCHER_DECISION_VERSION,
        "serving_projection_version": ER_SERVING_PROJECTION_VERSION,
    }


def load_entity_resolution_runtime(
    path: str | Path,
    *,
    allow_unpromoted_human_diagnostic: bool = False,
) -> EntityResolutionRuntime:
    """Load a human-trained LightGBM artifact, rejecting synthetic/transfer shortcuts."""

    artifact_path = Path(path).resolve()
    metadata = _json_object(artifact_path / "metadata.json")
    if metadata.get("artifact_format_version") != ARTIFACT_FORMAT_VERSION:
        raise EntityResolutionArtifactError("unsupported entity-resolution artifact format")
    if metadata.get("model_type") != LightGBMEntityResolver.model_type:
        raise EntityResolutionArtifactError(
            "production catalogue matching requires a LightGBM entity resolver"
        )
    if metadata.get("is_fitted") is not True:
        raise EntityResolutionArtifactError("entity-resolution artifact is not fitted")
    calibrator = metadata.get("calibrator")
    if not isinstance(calibrator, Mapping) or calibrator.get("is_fitted") is not True:
        raise EntityResolutionArtifactError(
            "entity-resolution artifact requires a fitted probability calibrator"
        )
    evidence_payload = _json_object(artifact_path / ER_SERVING_EVIDENCE_FILENAME)
    evidence = EntityResolutionServingEvidence.from_dict(evidence_payload)
    actual_sha256 = entity_resolution_artifact_sha256(artifact_path)
    if evidence.artifact_core_sha256 != actual_sha256:
        raise EntityResolutionArtifactError(
            "entity-resolution serving evidence does not match artifact bytes"
        )
    metadata_thresholds = metadata.get("thresholds")
    if not isinstance(metadata_thresholds, Mapping) or (
        float(metadata_thresholds.get("auto_match", -1)) != evidence.auto_match_threshold
        or float(metadata_thresholds.get("manual_review", -1)) != evidence.manual_review_threshold
    ):
        raise EntityResolutionArtifactError(
            "entity-resolution serving thresholds do not match model metadata"
        )
    if evidence.label_source != ER_HUMAN_LABEL_SOURCE:
        raise EntityResolutionArtifactError(
            "only attributable human-reviewed artifacts may enter catalogue matching"
        )
    if evidence.claim_scope != ER_PRODUCTION_CLAIM_SCOPE or evidence.transfer_only:
        raise EntityResolutionArtifactError(
            "transfer-only entity-resolution artifacts cannot enter catalogue matching"
        )
    if evidence.synthetic_rows:
        raise EntityResolutionArtifactError(
            "synthetic entity-resolution artifacts cannot enter catalogue matching"
        )
    if not evidence.source_training_eligible:
        raise EntityResolutionArtifactError("artifact source policy forbids model training")
    if not evidence.source_published_metrics_eligible:
        raise EntityResolutionArtifactError(
            "artifact source policy forbids production metric evidence"
        )
    if not evidence.source_model_serving_eligible:
        raise EntityResolutionArtifactError(
            "artifact source policy does not authorize derived-model serving"
        )
    if evidence.feature_contract_sha256 != _feature_contract_sha256():
        raise EntityResolutionArtifactError("entity-resolution feature contract has drifted")
    if evidence.matcher_decision_version != ER_CATALOG_MATCHER_DECISION_VERSION:
        raise EntityResolutionArtifactError("catalogue entity-resolution policy has drifted")
    if evidence.serving_projection_version != ER_SERVING_PROJECTION_VERSION:
        raise EntityResolutionArtifactError("entity-resolution serving projection has drifted")
    if not evidence.deployment_eligible and not allow_unpromoted_human_diagnostic:
        raise EntityResolutionArtifactError(
            "entity-resolution artifact is not promoted; explicit diagnostic opt-in is required"
        )
    resolver = load_entity_resolver(artifact_path)
    if not isinstance(resolver, LightGBMEntityResolver):
        raise EntityResolutionArtifactError(
            "production catalogue matching requires a LightGBM entity resolver"
        )
    return EntityResolutionRuntime(
        resolver=resolver,
        evidence=evidence,
        artifact_path=artifact_path,
        release_sha256=entity_resolution_release_sha256(artifact_path),
    )


@dataclass(frozen=True, slots=True)
class EntityResolutionRelease:
    """A production-authorized runtime and the four artifacts that authorize it."""

    runtime: EntityResolutionRuntime
    evaluation: EntityResolutionProductionEvaluation
    policy: EntityResolutionPolicy
    rights: EntityResolutionRightsApproval
    identity: EntityResolutionReleaseIdentity

    def __post_init__(self) -> None:
        if not self.runtime.production_authorized:
            raise EntityResolutionArtifactError("ER release runtime is not production authorized")
        if self.runtime.release_identity != self.identity:
            raise EntityResolutionArtifactError("ER runtime identity does not match release")
        if self.runtime.release_policy != self.policy:
            raise EntityResolutionArtifactError("ER runtime policy does not match release")


def _canonical_value_sha256(value: object) -> str:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise EntityResolutionArtifactError(
            "entity-resolution artifact contains non-canonical metadata"
        ) from error
    return hashlib.sha256(payload).hexdigest()


def _release_blockers(
    runtime: EntityResolutionRuntime,
    evaluation: EntityResolutionProductionEvaluation,
    policy: EntityResolutionPolicy,
    rights: EntityResolutionRightsApproval,
    *,
    evaluation_sha256: str,
) -> tuple[str, ...]:
    """Return every immutable-evidence mismatch without trusting eligibility flags."""

    evidence = runtime.evidence
    blockers: list[str] = []
    if evaluation.schema_version != ER_EVALUATION_SCHEMA_VERSION_V2:
        blockers.append("evaluation schema is not production v2")
    if policy.claim_scope != ER_PRODUCTION_CLAIM_SCOPE:
        blockers.append("policy claim scope is not the PC retailer catalogue")
    if policy.required_label_source != "human_reviewed":
        blockers.append("policy does not require human-reviewed labels")
    if policy.required_model_type != LightGBMEntityResolver.model_type:
        blockers.append("policy does not require the LightGBM entity resolver")
    if policy.required_matcher_decision_version != ER_CATALOG_MATCHER_DECISION_VERSION:
        blockers.append("policy matcher decision version does not match this runtime")
    if policy.required_serving_projection_version != ER_SERVING_PROJECTION_VERSION:
        blockers.append("policy serving projection version does not match this runtime")
    if not policy.require_promoted_entity_resolution_model:
        blockers.append("policy does not require a promoted entity-resolution model")

    if evaluation.model_version != runtime.model_version:
        blockers.append("evaluation model_version does not match model release")
    if evaluation.dataset_version != evidence.dataset_version:
        blockers.append("evaluation dataset_version does not match serving evidence")
    if evaluation.label_source != policy.required_label_source:
        blockers.append("evaluation label source does not match policy")
    if evaluation.synthetic:
        blockers.append("synthetic evaluation cannot authorize production")
    if evaluation.artifact_sha256 != runtime.release_sha256:
        blockers.append("evaluation artifact SHA-256 does not match model release")
    if evaluation.review_queue_sha256 != evidence.review_queue_sha256:
        blockers.append("evaluation review_queue_sha256 does not match serving evidence")
    if evaluation.frozen_test_groups_sha256 != evidence.frozen_test_groups_sha256:
        blockers.append("evaluation frozen_test_groups_sha256 does not match serving evidence")
    if evaluation.auto_match_threshold != policy.auto_match_threshold:
        blockers.append("evaluation auto-match threshold does not match policy")
    if runtime.resolver.thresholds.auto_match != policy.auto_match_threshold:
        blockers.append("model auto-match threshold does not match policy")
    if runtime.resolver.thresholds.manual_review != policy.manual_review_threshold:
        blockers.append("model manual-review threshold does not match policy")
    if evaluation.precision < policy.minimum_precision:
        blockers.append("evaluation precision is below policy")
    if evaluation.labelled_pair_count < policy.minimum_labelled_pairs:
        blockers.append("evaluation labelled-pair count is below policy")
    numerator = evaluation.precision_numerator
    denominator = evaluation.precision_denominator
    if numerator is None or denominator is None:
        blockers.append("evaluation precision evidence counts are missing")
    else:
        if denominator < policy.minimum_auto_matches:
            blockers.append("evaluation automatic-match support is below policy")
        if denominator > evaluation.labelled_pair_count:
            blockers.append("evaluation automatic-match support exceeds labelled pairs")
        if abs(numerator / denominator - evaluation.precision) > 1e-9:
            blockers.append("evaluation precision does not match evidence counts")
    if (
        evaluation.precision_ci_lower is None
        or evaluation.precision_ci_lower < policy.minimum_precision
    ):
        blockers.append("evaluation precision confidence lower bound is below policy")
    if evaluation.recall is None or evaluation.recall < policy.minimum_recall:
        blockers.append("evaluation recall is below policy")
    if evaluation.f1 is None or evaluation.f1 < policy.minimum_f1:
        blockers.append("evaluation F1 is below policy")

    # These exact references make the operator-reviewed rights record the trust root.
    # The legacy evaluation `reportable` and `deployment_eligible` booleans are parsed
    # for schema compatibility but intentionally do not confer or deny authority here.
    if rights.model_release_sha256 != runtime.release_sha256:
        blockers.append("rights approval model release does not match activated release")
    if rights.evaluation_sha256 != evaluation_sha256:
        blockers.append("rights approval evaluation digest does not match exact report bytes")
    if rights.policy_sha256 != policy.policy_sha256:
        blockers.append("rights approval policy digest does not match exact policy")
    if rights.dataset_version != evaluation.dataset_version:
        blockers.append("rights approval dataset_version does not match evaluation")
    if rights.model_version != runtime.model_version:
        blockers.append("rights approval model_version does not match model release")
    if rights.review_queue_sha256 != evaluation.review_queue_sha256:
        blockers.append("rights approval review_queue_sha256 does not match evaluation")
    if rights.frozen_test_groups_sha256 != evaluation.frozen_test_groups_sha256:
        blockers.append("rights approval frozen_test_groups_sha256 does not match evaluation")
    if rights.approved_at < evaluation.evaluated_at.astimezone(UTC):
        blockers.append("rights approval predates the evaluation it approves")
    return tuple(blockers)


def load_entity_resolution_release(
    model_dir: str | Path,
    evaluation_path: str | Path,
    policy_path: str | Path,
    rights_path: str | Path,
    *,
    as_of: datetime | None = None,
) -> EntityResolutionRelease:
    """Load the sole production ER authority from four exact persisted artifacts.

    Content addressing proves integrity, while the operator-pinned rights approval is the
    trust root.  Bare CLI booleans and legacy eligibility fields never authorize serving.
    """

    policy = load_entity_resolution_policy(policy_path)
    evaluation = load_entity_resolution_evaluation(evaluation_path)
    if evaluation is None:  # pragma: no cover - the non-optional path makes this defensive.
        raise EntityResolutionArtifactError("production ER evaluation is required")
    rights = load_entity_resolution_rights_approval(rights_path)
    evaluation_sha256 = entity_resolution_file_sha256(evaluation_path)

    if as_of is not None and (as_of.tzinfo is None or as_of.utcoffset() is None):
        raise EntityResolutionArtifactError("ER release as_of must include a timezone")
    rights.assert_active(territory=policy.territory, as_of=as_of)

    # The legacy loader is diagnostic-only.  It verifies model/calibrator/evidence bytes,
    # while the checks below derive production authority from policy, metrics, and rights.
    runtime = load_entity_resolution_runtime(
        model_dir,
        allow_unpromoted_human_diagnostic=True,
    )
    blockers = _release_blockers(
        runtime,
        evaluation,
        policy,
        rights,
        evaluation_sha256=evaluation_sha256,
    )
    if blockers:
        raise EntityResolutionArtifactError("; ".join(blockers))

    artifact_path = runtime.artifact_path
    metadata_path = artifact_path / "metadata.json"
    metadata = _json_object(metadata_path)
    model_path = _model_file(artifact_path, metadata)
    calibrator = metadata.get("calibrator")
    if not isinstance(calibrator, Mapping) or calibrator.get("is_fitted") is not True:
        raise EntityResolutionArtifactError("ER release requires an embedded fitted calibrator")
    identity = build_entity_resolution_release_identity(
        artifact_core_sha256=entity_resolution_artifact_sha256(artifact_path),
        model_file_sha256=entity_resolution_file_sha256(model_path),
        metadata_sha256=entity_resolution_file_sha256(metadata_path),
        calibrator_sha256=_canonical_value_sha256(calibrator),
        serving_evidence_sha256=entity_resolution_file_sha256(
            artifact_path / ER_SERVING_EVIDENCE_FILENAME
        ),
        model_release_sha256=runtime.release_sha256,
        evaluation_sha256=evaluation_sha256,
        policy_sha256=policy.policy_sha256,
        rights_sha256=rights.rights_sha256,
    )
    authorized_runtime = replace(
        runtime,
        release_identity=identity,
        release_policy=policy,
    )
    return EntityResolutionRelease(
        runtime=authorized_runtime,
        evaluation=evaluation,
        policy=policy,
        rights=rights,
        identity=identity,
    )
