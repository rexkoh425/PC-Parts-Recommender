"""Immutable, human-reviewed canonical-identity resolution artifacts.

The source catalogue remains untouched.  A production import may materialise an
effective catalogue only after an artifact is bound to the exact source bytes and
the complete deterministic preflight finding set.  Every finding requires two
independent reviews and a third-party adjudication.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal
from urllib.parse import urlsplit

from pc_build_recommender.domain import CanonicalProduct
from pc_build_recommender.entity_resolution.normalization import normalize_identifier

from .canonical_identity import (
    CANONICAL_IDENTITY_PREFLIGHT_SCHEMA_VERSION,
    CanonicalIdentityMember,
    CanonicalIdentityPreflightReport,
    audit_canonical_product_identities,
)

CANONICAL_IDENTITY_RESOLUTION_SCHEMA_VERSION = (
    "pc-build-recommender.canonical-identity-resolution.v1"
)
_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")


class CanonicalIdentityResolutionError(ValueError):
    """Raised when an identity-resolution artifact cannot be trusted or applied."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _content_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CanonicalIdentityResolutionError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _object(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CanonicalIdentityResolutionError(f"{label} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise CanonicalIdentityResolutionError(
            f"{label} fields do not match the contract; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CanonicalIdentityResolutionError(f"{label} must be non-empty text")
    return value.strip()


def _sha256(value: object, *, label: str) -> str:
    result = _text(value, label=label)
    if _SHA256.fullmatch(result) is None:
        raise CanonicalIdentityResolutionError(f"{label} must be a lowercase SHA-256")
    return result


def _timestamp(value: object, *, label: str) -> datetime:
    raw = _text(value, label=label)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as error:
        raise CanonicalIdentityResolutionError(f"{label} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CanonicalIdentityResolutionError(f"{label} must be timezone-aware")
    return parsed


@dataclass(frozen=True, slots=True)
class CanonicalIdentityResolutionFinding:
    """One deterministic, manually resolvable preflight finding."""

    finding_id: str
    finding_type: Literal["missing_mpn", "duplicate_brand_mpn"]
    products: tuple[CanonicalIdentityMember, ...]
    normalized_brand: str | None = None
    normalized_manufacturer_part_number: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "finding_type": self.finding_type,
            "normalized_brand": self.normalized_brand,
            "normalized_manufacturer_part_number": (self.normalized_manufacturer_part_number),
            "products": [product.to_dict() for product in self.products],
        }


def canonical_identity_resolution_findings(
    report: CanonicalIdentityPreflightReport,
) -> tuple[CanonicalIdentityResolutionFinding, ...]:
    """Return the exact, ordered finding universe accepted by the v1 artifact."""

    findings: list[CanonicalIdentityResolutionFinding] = []
    for product in report.missing_mpn_products:
        normalized_brand = normalize_identifier(product.brand) or None
        body = {
            "finding_type": "missing_mpn",
            "normalized_brand": normalized_brand,
            "normalized_manufacturer_part_number": None,
            "products": [product.to_dict()],
        }
        findings.append(
            CanonicalIdentityResolutionFinding(
                finding_id=f"missing_mpn:{_content_sha256(body)}",
                finding_type="missing_mpn",
                products=(product,),
                normalized_brand=normalized_brand,
            )
        )
    for group in report.duplicate_identity_groups:
        body = {
            "finding_type": "duplicate_brand_mpn",
            "normalized_brand": group.normalized_brand,
            "normalized_manufacturer_part_number": (group.normalized_manufacturer_part_number),
            "products": [product.to_dict() for product in group.products],
        }
        findings.append(
            CanonicalIdentityResolutionFinding(
                finding_id=f"duplicate_brand_mpn:{_content_sha256(body)}",
                finding_type="duplicate_brand_mpn",
                products=group.products,
                normalized_brand=group.normalized_brand,
                normalized_manufacturer_part_number=(group.normalized_manufacturer_part_number),
            )
        )
    return tuple(sorted(findings, key=lambda item: item.finding_id))


def canonical_identity_conflict_set_sha256(
    report: CanonicalIdentityPreflightReport,
) -> str:
    """Hash the exact ordered finding universe, including every source member."""

    return _content_sha256(
        [finding.to_dict() for finding in canonical_identity_resolution_findings(report)]
    )


@dataclass(frozen=True, slots=True)
class CanonicalIdentityResolutionSummary:
    """Auditable lineage for one applied resolution artifact."""

    resolution_id: str
    artifact_file_sha256: str
    artifact_content_sha256: str
    source_catalog_sha256: str
    source_preflight_content_sha256: str
    conflict_set_sha256: str
    resolved_finding_count: int
    source_product_count: int
    effective_product_count: int
    mpn_overrides: Mapping[str, str]
    aliases: Mapping[str, str]
    source_preflight: CanonicalIdentityPreflightReport
    effective_preflight: CanonicalIdentityPreflightReport

    def __post_init__(self) -> None:
        object.__setattr__(self, "mpn_overrides", MappingProxyType(dict(self.mpn_overrides)))
        object.__setattr__(self, "aliases", MappingProxyType(dict(self.aliases)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CANONICAL_IDENTITY_RESOLUTION_SCHEMA_VERSION,
            "resolution_id": self.resolution_id,
            "artifact_file_sha256": self.artifact_file_sha256,
            "artifact_content_sha256": self.artifact_content_sha256,
            "source_catalog_sha256": self.source_catalog_sha256,
            "source_preflight_content_sha256": self.source_preflight_content_sha256,
            "conflict_set_sha256": self.conflict_set_sha256,
            "resolved_finding_count": self.resolved_finding_count,
            "source_product_count": self.source_product_count,
            "effective_product_count": self.effective_product_count,
            "mpn_overrides": dict(sorted(self.mpn_overrides.items())),
            "aliases": dict(sorted(self.aliases.items())),
            "source_preflight": self.source_preflight.to_dict(),
            "effective_preflight": self.effective_preflight.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class CanonicalIdentityResolutionApplication:
    """Effective products plus immutable resolution lineage."""

    products: tuple[CanonicalProduct, ...]
    summary: CanonicalIdentityResolutionSummary


def _evidence(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise CanonicalIdentityResolutionError(f"{label} must contain evidence")
    evidence_ids: list[str] = []
    for index, raw_item in enumerate(value):
        item_label = f"{label}[{index}]"
        item = _object(raw_item, label=item_label)
        _exact_keys(
            item,
            {"evidence_id", "source_url", "content_sha256", "note"},
            label=item_label,
        )
        evidence_id = _text(item["evidence_id"], label=f"{item_label}.evidence_id")
        source_url = _text(item["source_url"], label=f"{item_label}.source_url")
        parsed = urlsplit(source_url)
        if (
            parsed.scheme != "https"
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise CanonicalIdentityResolutionError(
                f"{item_label}.source_url must be a credential-free HTTPS URL"
            )
        _sha256(item["content_sha256"], label=f"{item_label}.content_sha256")
        _text(item["note"], label=f"{item_label}.note")
        evidence_ids.append(evidence_id)
    if len(evidence_ids) != len(set(evidence_ids)):
        raise CanonicalIdentityResolutionError(f"{label} contains duplicate evidence IDs")
    return tuple(evidence_ids)


def _decision(
    value: object,
    *,
    finding: CanonicalIdentityResolutionFinding,
    label: str,
) -> dict[str, Any]:
    decision = _object(value, label=label)
    _exact_keys(decision, {"assignments"}, label=label)
    raw_assignments = decision["assignments"]
    if not isinstance(raw_assignments, list) or not raw_assignments:
        raise CanonicalIdentityResolutionError(f"{label}.assignments must be non-empty")
    assignments: list[dict[str, str]] = []
    for index, raw_assignment in enumerate(raw_assignments):
        assignment_label = f"{label}.assignments[{index}]"
        assignment = _object(raw_assignment, label=assignment_label)
        _exact_keys(
            assignment,
            {"source_product_id", "canonical_product_id", "manufacturer_part_number"},
            label=assignment_label,
        )
        assignments.append(
            {
                "source_product_id": _text(
                    assignment["source_product_id"],
                    label=f"{assignment_label}.source_product_id",
                ),
                "canonical_product_id": _text(
                    assignment["canonical_product_id"],
                    label=f"{assignment_label}.canonical_product_id",
                ),
                "manufacturer_part_number": _text(
                    assignment["manufacturer_part_number"],
                    label=f"{assignment_label}.manufacturer_part_number",
                ),
            }
        )
    assignments.sort(key=lambda item: item["source_product_id"])
    source_ids = [assignment["source_product_id"] for assignment in assignments]
    expected_ids = sorted(product.product_id for product in finding.products)
    if source_ids != expected_ids:
        raise CanonicalIdentityResolutionError(
            f"{label} must resolve the exact member set for {finding.finding_id}"
        )
    target_ids = {assignment["canonical_product_id"] for assignment in assignments}
    if not target_ids.issubset(set(expected_ids)):
        raise CanonicalIdentityResolutionError(
            f"{label} canonical targets must belong to {finding.finding_id}"
        )
    if finding.finding_type == "missing_mpn" and source_ids != sorted(target_ids):
        raise CanonicalIdentityResolutionError(
            f"{label} missing-MPN decisions must retain their source product ID"
        )
    by_source = {assignment["source_product_id"]: assignment for assignment in assignments}
    members = {product.product_id: product for product in finding.products}
    for target_id in target_ids:
        target = by_source.get(target_id)
        if target is None or target["canonical_product_id"] != target_id:
            raise CanonicalIdentityResolutionError(
                f"{label} canonical target {target_id} must be retained explicitly"
            )
        target_mpn = normalize_identifier(target["manufacturer_part_number"])
        if not target_mpn:
            raise CanonicalIdentityResolutionError(
                f"{label} canonical target {target_id} requires a non-empty MPN"
            )
        for assignment in assignments:
            if assignment["canonical_product_id"] != target_id:
                continue
            if members[assignment["source_product_id"]].category != members[target_id].category:
                raise CanonicalIdentityResolutionError(
                    f"{label} cannot alias products across component categories"
                )
            if normalize_identifier(assignment["manufacturer_part_number"]) != target_mpn:
                raise CanonicalIdentityResolutionError(
                    f"{label} aliases for {target_id} must agree on the verified MPN"
                )
    return {"assignments": assignments}


def _review(
    value: object,
    *,
    finding: CanonicalIdentityResolutionFinding,
    label: str,
) -> tuple[str, str, datetime, dict[str, Any]]:
    review = _object(value, label=label)
    _exact_keys(
        review,
        {"review_id", "reviewer_id", "reviewed_at", "rationale", "evidence", "decision"},
        label=label,
    )
    review_id = _text(review["review_id"], label=f"{label}.review_id")
    reviewer_id = _text(review["reviewer_id"], label=f"{label}.reviewer_id")
    reviewed_at = _timestamp(review["reviewed_at"], label=f"{label}.reviewed_at")
    _text(review["rationale"], label=f"{label}.rationale")
    _evidence(review["evidence"], label=f"{label}.evidence")
    decision = _decision(review["decision"], finding=finding, label=f"{label}.decision")
    return review_id, reviewer_id, reviewed_at, decision


def _adjudication(
    value: object,
    *,
    finding: CanonicalIdentityResolutionFinding,
    label: str,
) -> tuple[str, str, datetime, dict[str, Any]]:
    adjudication = _object(value, label=label)
    _exact_keys(
        adjudication,
        {
            "adjudication_id",
            "adjudicator_id",
            "adjudicated_at",
            "rationale",
            "evidence",
            "decision",
        },
        label=label,
    )
    adjudication_id = _text(adjudication["adjudication_id"], label=f"{label}.adjudication_id")
    adjudicator_id = _text(adjudication["adjudicator_id"], label=f"{label}.adjudicator_id")
    adjudicated_at = _timestamp(adjudication["adjudicated_at"], label=f"{label}.adjudicated_at")
    _text(adjudication["rationale"], label=f"{label}.rationale")
    _evidence(adjudication["evidence"], label=f"{label}.evidence")
    decision = _decision(adjudication["decision"], finding=finding, label=f"{label}.decision")
    return adjudication_id, adjudicator_id, adjudicated_at, decision


def load_and_apply_canonical_identity_resolution(
    artifact_path: str | Path,
    *,
    expected_artifact_sha256: str,
    source_catalog_path: str | Path,
    products: Iterable[CanonicalProduct],
    source_preflight: CanonicalIdentityPreflightReport | None = None,
) -> CanonicalIdentityResolutionApplication:
    """Verify and apply one complete resolution artifact without mutating source rows."""

    expected_file_hash = _sha256(
        expected_artifact_sha256,
        label="expected canonical identity resolution artifact SHA-256",
    )
    path = Path(artifact_path).resolve()
    source_path = Path(source_catalog_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"canonical identity resolution artifact not found: {path}")
    if not source_path.is_file():
        raise FileNotFoundError(f"source catalogue not found: {source_path}")
    actual_file_hash = _file_sha256(path)
    if actual_file_hash != expected_file_hash:
        raise CanonicalIdentityResolutionError(
            "canonical identity resolution artifact file SHA-256 does not match its release binding"
        )
    try:
        raw_artifact = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CanonicalIdentityResolutionError(
            f"invalid canonical identity resolution artifact: {path}"
        ) from error
    artifact = _object(raw_artifact, label="canonical identity resolution artifact")
    _exact_keys(
        artifact,
        {
            "schema_version",
            "resolution_id",
            "created_at",
            "source_catalog",
            "resolutions",
            "content_sha256",
        },
        label="canonical identity resolution artifact",
    )
    if artifact["schema_version"] != CANONICAL_IDENTITY_RESOLUTION_SCHEMA_VERSION:
        raise CanonicalIdentityResolutionError("unsupported identity resolution schema version")
    resolution_id = _text(artifact["resolution_id"], label="resolution_id")
    created_at = _timestamp(artifact["created_at"], label="created_at")
    declared_content_hash = _sha256(artifact["content_sha256"], label="content_sha256")
    content_payload = dict(artifact)
    content_payload.pop("content_sha256")
    if _content_sha256(content_payload) != declared_content_hash:
        raise CanonicalIdentityResolutionError(
            "canonical identity resolution content SHA-256 is invalid"
        )

    source_products = tuple(products)
    product_ids = [product.product_id for product in source_products]
    if len(product_ids) != len(set(product_ids)):
        raise CanonicalIdentityResolutionError(
            "duplicate product IDs cannot be repaired by an identity resolution artifact"
        )
    report = source_preflight or audit_canonical_product_identities(source_products)
    if report.record_count != len(source_products):
        raise CanonicalIdentityResolutionError(
            "source preflight record count does not match the supplied catalogue"
        )
    if report.missing_brand_products:
        raise CanonicalIdentityResolutionError(
            "missing brands cannot be repaired by the v1 identity resolution artifact"
        )
    if report.duplicate_product_id_groups:
        raise CanonicalIdentityResolutionError(
            "duplicate product IDs cannot be repaired by the v1 identity resolution artifact"
        )

    source = _object(artifact["source_catalog"], label="source_catalog")
    _exact_keys(
        source,
        {
            "sha256",
            "size_bytes",
            "preflight_schema_version",
            "preflight_content_sha256",
            "conflict_set_sha256",
        },
        label="source_catalog",
    )
    source_hash = _sha256(source["sha256"], label="source_catalog.sha256")
    if source_hash != _file_sha256(source_path):
        raise CanonicalIdentityResolutionError(
            "identity resolution artifact is stale for the source catalogue bytes"
        )
    source_size = source["size_bytes"]
    if type(source_size) is not int or source_size < 0 or source_size != source_path.stat().st_size:
        raise CanonicalIdentityResolutionError(
            "identity resolution artifact source catalogue size is stale"
        )
    if source["preflight_schema_version"] != CANONICAL_IDENTITY_PREFLIGHT_SCHEMA_VERSION:
        raise CanonicalIdentityResolutionError("source preflight schema version is stale")
    report_hash = report.to_dict()["content_sha256"]
    if source["preflight_content_sha256"] != report_hash:
        raise CanonicalIdentityResolutionError("source preflight content hash is stale")
    conflict_set_hash = canonical_identity_conflict_set_sha256(report)
    if source["conflict_set_sha256"] != conflict_set_hash:
        raise CanonicalIdentityResolutionError("source identity conflict set hash is stale")

    findings = canonical_identity_resolution_findings(report)
    finding_by_id = {finding.finding_id: finding for finding in findings}
    raw_resolutions = artifact["resolutions"]
    if not isinstance(raw_resolutions, list):
        raise CanonicalIdentityResolutionError("resolutions must be an array")
    final_decisions: dict[str, dict[str, Any]] = {}
    adjudicated_times: list[datetime] = []
    for index, raw_resolution in enumerate(raw_resolutions):
        label = f"resolutions[{index}]"
        resolution = _object(raw_resolution, label=label)
        _exact_keys(resolution, {"finding_id", "reviews", "adjudication"}, label=label)
        finding_id = _text(resolution["finding_id"], label=f"{label}.finding_id")
        if finding_id in final_decisions:
            raise CanonicalIdentityResolutionError(f"duplicate resolution: {finding_id}")
        finding = finding_by_id.get(finding_id)
        if finding is None:
            raise CanonicalIdentityResolutionError(
                f"resolution references a stale or unknown finding: {finding_id}"
            )
        raw_reviews = resolution["reviews"]
        if not isinstance(raw_reviews, list) or len(raw_reviews) != 2:
            raise CanonicalIdentityResolutionError(
                f"{label}.reviews must contain exactly two independent reviews"
            )
        reviews = [
            _review(raw_review, finding=finding, label=f"{label}.reviews[{review_index}]")
            for review_index, raw_review in enumerate(raw_reviews)
        ]
        review_ids = {review[0] for review in reviews}
        reviewer_ids = {review[1] for review in reviews}
        if len(review_ids) != 2 or len(reviewer_ids) != 2:
            raise CanonicalIdentityResolutionError(
                f"{label} requires distinct review IDs and reviewer IDs"
            )
        adjudication = _adjudication(
            resolution["adjudication"], finding=finding, label=f"{label}.adjudication"
        )
        if adjudication[0] in review_ids or adjudication[1] in reviewer_ids:
            raise CanonicalIdentityResolutionError(
                f"{label} adjudicator and adjudication ID must be independent of reviewers"
            )
        if adjudication[2] < max(review[2] for review in reviews):
            raise CanonicalIdentityResolutionError(
                f"{label} adjudication must occur after both reviews"
            )
        adjudicated_times.append(adjudication[2])
        final_decisions[finding_id] = adjudication[3]
    expected_findings = set(finding_by_id)
    if set(final_decisions) != expected_findings:
        raise CanonicalIdentityResolutionError(
            "identity resolution artifact is incomplete for the exact conflict set; "
            f"missing={sorted(expected_findings - set(final_decisions))}, "
            f"extra={sorted(set(final_decisions) - expected_findings)}"
        )
    if adjudicated_times and created_at < max(adjudicated_times):
        raise CanonicalIdentityResolutionError(
            "identity resolution artifact cannot be created before its adjudications"
        )

    assignment_by_source: dict[str, dict[str, str]] = {}
    for finding_id in sorted(final_decisions):
        for assignment in final_decisions[finding_id]["assignments"]:
            source_product_id = assignment["source_product_id"]
            if source_product_id in assignment_by_source:
                raise CanonicalIdentityResolutionError(
                    f"product appears in more than one final decision: {source_product_id}"
                )
            assignment_by_source[source_product_id] = assignment
    aliases: dict[str, str] = {}
    mpn_overrides: dict[str, str] = {}
    effective_by_id: dict[str, CanonicalProduct] = {}
    for product in source_products:
        assignment = assignment_by_source.get(product.product_id)
        if assignment is None:
            effective_by_id[product.product_id] = product
            continue
        target_id = assignment["canonical_product_id"]
        if target_id != product.product_id:
            aliases[product.product_id] = target_id
            continue
        verified_mpn = assignment["manufacturer_part_number"]
        if product.manufacturer_part_number != verified_mpn:
            mpn_overrides[product.product_id] = verified_mpn
            effective_by_id[product.product_id] = product.model_copy(
                update={"manufacturer_part_number": verified_mpn}
            )
        else:
            effective_by_id[product.product_id] = product

    # A true-duplicate alias changes canonical ownership, not source lineage. Move every
    # source provenance row to the adjudicated retained product so the effective release
    # remains fully auditable and the persistence layer can verify the complete set.
    provenance_by_target: dict[str, list[Any]] = {
        product_id: [] for product_id in effective_by_id
    }
    provenance_owners: dict[str, str] = {}
    for product in source_products:
        target_id = aliases.get(product.product_id, product.product_id)
        if target_id not in effective_by_id:
            raise CanonicalIdentityResolutionError(
                f"identity resolution references a missing retained product: {target_id}"
            )
        for provenance in product.provenance:
            previous_owner = provenance_owners.get(provenance.provenance_id)
            if previous_owner is not None:
                raise CanonicalIdentityResolutionError(
                    "canonical identity resolution contains a duplicate provenance ID "
                    f"across source records: {provenance.provenance_id} "
                    f"({previous_owner}, {target_id})"
                )
            provenance_owners[provenance.provenance_id] = target_id
            provenance_by_target[target_id].append(
                provenance.model_copy(update={"product_id": target_id, "listing_id": None})
            )
    effective_products = [
        effective_by_id[product_id].model_copy(
            update={
                "provenance": sorted(
                    provenance_by_target[product_id],
                    key=lambda item: item.provenance_id,
                )
            }
        )
        for product_id in sorted(effective_by_id)
    ]
    effective_preflight = audit_canonical_product_identities(effective_products)
    if not effective_preflight.production_ready:
        raise CanonicalIdentityResolutionError(
            "adjudicated identity decisions do not produce a production-ready catalogue: "
            + "; ".join(effective_preflight.blockers())
        )
    summary = CanonicalIdentityResolutionSummary(
        resolution_id=resolution_id,
        artifact_file_sha256=actual_file_hash,
        artifact_content_sha256=declared_content_hash,
        source_catalog_sha256=source_hash,
        source_preflight_content_sha256=report_hash,
        conflict_set_sha256=conflict_set_hash,
        resolved_finding_count=len(final_decisions),
        source_product_count=len(source_products),
        effective_product_count=len(effective_products),
        mpn_overrides=mpn_overrides,
        aliases=aliases,
        source_preflight=report,
        effective_preflight=effective_preflight,
    )
    return CanonicalIdentityResolutionApplication(
        products=tuple(effective_products),
        summary=summary,
    )


__all__ = [
    "CANONICAL_IDENTITY_RESOLUTION_SCHEMA_VERSION",
    "CanonicalIdentityResolutionApplication",
    "CanonicalIdentityResolutionError",
    "CanonicalIdentityResolutionFinding",
    "CanonicalIdentityResolutionSummary",
    "canonical_identity_conflict_set_sha256",
    "canonical_identity_resolution_findings",
    "load_and_apply_canonical_identity_resolution",
]
