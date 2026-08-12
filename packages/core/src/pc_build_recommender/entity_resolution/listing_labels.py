"""Frozen, independently reviewed labels for listing-level ER evaluation.

Candidate generation and model scores are deliberately absent from this contract.  The
dataset contains only source snapshots and attributable human judgments, so evaluation can
replay the deployed matcher without converting heuristics into labels.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Self, cast

from pc_build_recommender.evaluation.manifest import sha256_json

from .candidate_generation import canonical_pc_category
from .records import CanonicalProductRecord, ListingRecord
from .review import HumanMatchLabel, SourceUsePolicy

ER_LISTING_LABEL_SET_SCHEMA_VERSION = "pc-build-recommender.er-listing-label-set.v2"
ER_CANONICAL_CATALOGUE_SCHEMA_VERSION = (
    "pc-build-recommender.er-canonical-catalogue-release.v1"
)
ER_LISTING_REVIEW_PROTOCOL = "blinded-independent-dual-review-with-adjudication-v1"
ER_LISTING_LABEL_SOURCE = "independently_reviewed_human"
ER_LISTING_LABEL_DOMAIN = "pc_components"
ER_LISTING_LABEL_TERRITORY = "SG"

_MAX_LABEL_SET_BYTES = 128 * 1024 * 1024
_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_LABEL_SET_FIELDS = frozenset(
    {
        "schema_version",
        "dataset_version",
        "territory",
        "domain",
        "label_source",
        "review_protocol",
        "created_at",
        "source_review_queue_sha256",
        "canonical_catalogue_version",
        "canonical_catalogue_sha256",
        "canonical_catalogue_file_sha256",
        "source_policy",
        "products",
        "listing_groups",
        "dataset_sha256",
    }
)
_JUDGMENT_FIELDS = frozenset(
    {
        "reviewer_id",
        "assignment_id",
        "label",
        "reviewed_at",
        "evidence_reference",
    }
)
_PAIR_LABEL_FIELDS = frozenset(
    {"product_id", "judgments", "adjudication", "resolved_label"}
)
_LISTING_GROUP_FIELDS = frozenset({"listing", "match_disposition", "pair_labels"})
_CATALOGUE_FIELDS = frozenset(
    {"schema_version", "catalogue_version", "products", "catalogue_sha256"}
)
_SOURCE_POLICY_FIELDS = frozenset(
    {
        "listing_source",
        "catalogue_source",
        "data_version",
        "training_eligible",
        "published_metrics_eligible",
        "model_serving_eligible",
        "scope_note",
    }
)


class ListingLabelSetError(ValueError):
    """Raised when frozen listing labels are malformed or not independently reviewed."""


def canonical_catalogue_sha256(
    catalogue_version: str,
    products: Sequence[CanonicalProductRecord],
) -> str:
    """Return the semantic commitment for one complete canonical catalogue release."""

    return sha256_json(
        {
            "schema_version": ER_CANONICAL_CATALOGUE_SCHEMA_VERSION,
            "catalogue_version": _text(catalogue_version, name="catalogue_version"),
            "products": [product.to_dict() for product in products],
        }
    )


def _reject_constant(value: str) -> None:
    raise ListingLabelSetError(f"non-finite JSON number is forbidden: {value}")


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ListingLabelSetError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _object(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ListingLabelSetError(f"{name} must be an object")
    return {str(key): nested for key, nested in value.items()}


def _array(value: object, *, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ListingLabelSetError(f"{name} must be an array")
    return value


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ListingLabelSetError(f"{name} must be a non-empty string")
    return value.strip()


def _sha256(value: object, *, name: str) -> str:
    digest = _text(value, name=name)
    if len(digest) != 64 or any(character not in _SHA256_CHARACTERS for character in digest):
        raise ListingLabelSetError(f"{name} must be a lowercase SHA-256")
    return digest


def _timestamp(value: object, *, name: str) -> str:
    raw = _text(value, name=name)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as error:
        raise ListingLabelSetError(f"{name} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ListingLabelSetError(f"{name} must include a timezone")
    return raw


def _exact_fields(payload: Mapping[str, Any], expected: frozenset[str], *, name: str) -> None:
    missing = sorted(expected - set(payload))
    extra = sorted(set(payload) - expected)
    if missing:
        raise ListingLabelSetError(f"{name} missing fields: {missing}")
    if extra:
        raise ListingLabelSetError(f"{name} contains unknown fields: {extra}")


def _binary_label(value: object, *, name: str) -> HumanMatchLabel:
    try:
        label = HumanMatchLabel(_text(value, name=name))
    except ValueError as error:
        raise ListingLabelSetError(f"{name} must be MATCH or NON_MATCH") from error
    if label not in {HumanMatchLabel.MATCH, HumanMatchLabel.NON_MATCH}:
        raise ListingLabelSetError(f"{name} must be MATCH or NON_MATCH")
    return label


@dataclass(frozen=True, slots=True)
class IndependentReviewJudgment:
    reviewer_id: str
    assignment_id: str
    label: HumanMatchLabel
    reviewed_at: str
    evidence_reference: str

    def __post_init__(self) -> None:
        for name in ("reviewer_id", "assignment_id", "evidence_reference"):
            _text(getattr(self, name), name=name)
        _timestamp(self.reviewed_at, name="reviewed_at")
        if self.label not in {HumanMatchLabel.MATCH, HumanMatchLabel.NON_MATCH}:
            raise ListingLabelSetError("independent judgments must be binary")

    def to_dict(self) -> dict[str, str]:
        return {
            "reviewer_id": self.reviewer_id,
            "assignment_id": self.assignment_id,
            "label": self.label.value,
            "reviewed_at": self.reviewed_at,
            "evidence_reference": self.evidence_reference,
        }

    @classmethod
    def from_dict(cls, value: object, *, name: str) -> Self:
        payload = _object(value, name=name)
        _exact_fields(payload, _JUDGMENT_FIELDS, name=name)
        return cls(
            reviewer_id=_text(payload["reviewer_id"], name=f"{name}.reviewer_id"),
            assignment_id=_text(payload["assignment_id"], name=f"{name}.assignment_id"),
            label=_binary_label(payload["label"], name=f"{name}.label"),
            reviewed_at=_timestamp(payload["reviewed_at"], name=f"{name}.reviewed_at"),
            evidence_reference=_text(
                payload["evidence_reference"], name=f"{name}.evidence_reference"
            ),
        )


@dataclass(frozen=True, slots=True)
class IndependentlyReviewedPairLabel:
    product_id: str
    judgments: tuple[IndependentReviewJudgment, IndependentReviewJudgment]
    resolved_label: HumanMatchLabel
    adjudication: IndependentReviewJudgment | None = None

    def __post_init__(self) -> None:
        _text(self.product_id, name="product_id")
        if len(self.judgments) != 2:
            raise ListingLabelSetError("each pair requires exactly two primary judgments")
        reviewer_ids = {judgment.reviewer_id for judgment in self.judgments}
        assignment_ids = {judgment.assignment_id for judgment in self.judgments}
        if len(reviewer_ids) != 2 or len(assignment_ids) != 2:
            raise ListingLabelSetError(
                "primary judgments must have independent reviewers and assignments"
            )
        if self.judgments != tuple(
            sorted(self.judgments, key=lambda item: (item.reviewer_id, item.assignment_id))
        ):
            raise ListingLabelSetError("primary judgments must use stable reviewer ordering")
        primary_labels = {judgment.label for judgment in self.judgments}
        if len(primary_labels) == 1:
            if self.adjudication is not None:
                raise ListingLabelSetError("agreeing primary judgments must not be adjudicated")
            if self.resolved_label is not self.judgments[0].label:
                raise ListingLabelSetError("resolved label does not match agreeing reviewers")
        else:
            if self.adjudication is None:
                raise ListingLabelSetError("disagreeing primary judgments require adjudication")
            if self.adjudication.reviewer_id in reviewer_ids:
                raise ListingLabelSetError("adjudicator must be independent of primary reviewers")
            if self.adjudication.assignment_id in assignment_ids:
                raise ListingLabelSetError("adjudication requires a distinct assignment")
            if self.resolved_label is not self.adjudication.label:
                raise ListingLabelSetError("resolved label does not match adjudication")

    def to_dict(self) -> dict[str, object]:
        return {
            "product_id": self.product_id,
            "judgments": [judgment.to_dict() for judgment in self.judgments],
            "adjudication": self.adjudication.to_dict() if self.adjudication else None,
            "resolved_label": self.resolved_label.value,
        }

    @classmethod
    def from_dict(cls, value: object, *, name: str) -> Self:
        payload = _object(value, name=name)
        _exact_fields(payload, _PAIR_LABEL_FIELDS, name=name)
        raw_judgments = _array(payload["judgments"], name=f"{name}.judgments")
        if len(raw_judgments) != 2:
            raise ListingLabelSetError(f"{name}.judgments must contain exactly two items")
        judgments = tuple(
            IndependentReviewJudgment.from_dict(item, name=f"{name}.judgments[{index}]")
            for index, item in enumerate(raw_judgments)
        )
        adjudication_payload = payload["adjudication"]
        adjudication = (
            None
            if adjudication_payload is None
            else IndependentReviewJudgment.from_dict(
                adjudication_payload, name=f"{name}.adjudication"
            )
        )
        return cls(
            product_id=_text(payload["product_id"], name=f"{name}.product_id"),
            judgments=cast(
                tuple[IndependentReviewJudgment, IndependentReviewJudgment], judgments
            ),
            adjudication=adjudication,
            resolved_label=_binary_label(
                payload["resolved_label"], name=f"{name}.resolved_label"
            ),
        )


@dataclass(frozen=True, slots=True)
class FrozenCanonicalCatalogue:
    """Content-addressed complete catalogue used by blocking and matcher replay."""

    catalogue_version: str
    products: tuple[CanonicalProductRecord, ...]
    catalogue_sha256: str
    schema_version: str = ER_CANONICAL_CATALOGUE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ER_CANONICAL_CATALOGUE_SCHEMA_VERSION:
            raise ListingLabelSetError("unsupported canonical-catalogue schema")
        _text(self.catalogue_version, name="catalogue_version")
        _sha256(self.catalogue_sha256, name="catalogue_sha256")
        if not self.products:
            raise ListingLabelSetError("canonical catalogue must contain products")
        product_ids = [product.product_id for product in self.products]
        if len(product_ids) != len(set(product_ids)):
            raise ListingLabelSetError("canonical catalogue contains duplicate product IDs")
        if product_ids != sorted(product_ids):
            raise ListingLabelSetError("canonical catalogue products must be sorted by product_id")
        if any(product.is_synthetic for product in self.products):
            raise ListingLabelSetError("synthetic products cannot enter a canonical release")
        if (
            canonical_catalogue_sha256(self.catalogue_version, self.products)
            != self.catalogue_sha256
        ):
            raise ListingLabelSetError("canonical-catalogue self-hash mismatch")

    def content_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "catalogue_version": self.catalogue_version,
            "products": [product.to_dict() for product in self.products],
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.content_payload(), "catalogue_sha256": self.catalogue_sha256}


@dataclass(frozen=True, slots=True)
class FrozenListingLabelGroup:
    listing: ListingRecord
    match_disposition: str
    pair_labels: tuple[IndependentlyReviewedPairLabel, ...]

    def __post_init__(self) -> None:
        if not self.pair_labels:
            raise ListingLabelSetError("each listing group requires pair labels")
        product_ids = [pair.product_id for pair in self.pair_labels]
        if len(product_ids) != len(set(product_ids)):
            raise ListingLabelSetError("listing group contains duplicate labelled pairs")
        if product_ids != sorted(product_ids):
            raise ListingLabelSetError("listing pair labels must be sorted by product_id")
        if self.match_disposition not in {"in_catalogue_match", "no_catalogue_match"}:
            raise ListingLabelSetError(
                "match_disposition must be in_catalogue_match or no_catalogue_match"
            )
        positives = [
            pair.product_id
            for pair in self.pair_labels
            if pair.resolved_label is HumanMatchLabel.MATCH
        ]
        expected_positive_count = int(self.match_disposition == "in_catalogue_match")
        if len(positives) != expected_positive_count:
            raise ListingLabelSetError(
                "listing-level match disposition disagrees with independently reviewed matches"
            )

    @property
    def gold_product_id(self) -> str | None:
        return next(
            (
                pair.product_id
                for pair in self.pair_labels
                if pair.resolved_label is HumanMatchLabel.MATCH
            ),
            None,
        )

    @property
    def is_out_of_catalogue(self) -> bool:
        return self.match_disposition == "no_catalogue_match"

    def to_dict(self) -> dict[str, object]:
        return {
            "listing": self.listing.to_dict(),
            "match_disposition": self.match_disposition,
            "pair_labels": [pair.to_dict() for pair in self.pair_labels],
        }

    @classmethod
    def from_dict(cls, value: object, *, name: str) -> Self:
        payload = _object(value, name=name)
        _exact_fields(payload, _LISTING_GROUP_FIELDS, name=name)
        listing_payload = _object(payload["listing"], name=f"{name}.listing")
        raw_labels = _array(payload["pair_labels"], name=f"{name}.pair_labels")
        return cls(
            listing=ListingRecord.from_dict(listing_payload),
            match_disposition=_text(
                payload["match_disposition"], name=f"{name}.match_disposition"
            ),
            pair_labels=tuple(
                IndependentlyReviewedPairLabel.from_dict(
                    item, name=f"{name}.pair_labels[{index}]"
                )
                for index, item in enumerate(raw_labels)
            ),
        )


@dataclass(frozen=True, slots=True)
class FrozenListingLabelSet:
    dataset_version: str
    created_at: str
    source_review_queue_sha256: str
    canonical_catalogue_version: str
    canonical_catalogue_sha256: str
    canonical_catalogue_file_sha256: str
    source_policy: SourceUsePolicy
    products: tuple[CanonicalProductRecord, ...]
    listing_groups: tuple[FrozenListingLabelGroup, ...]
    dataset_sha256: str
    territory: str = ER_LISTING_LABEL_TERRITORY
    domain: str = ER_LISTING_LABEL_DOMAIN
    label_source: str = ER_LISTING_LABEL_SOURCE
    review_protocol: str = ER_LISTING_REVIEW_PROTOCOL
    schema_version: str = ER_LISTING_LABEL_SET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ER_LISTING_LABEL_SET_SCHEMA_VERSION:
            raise ListingLabelSetError("unsupported listing-label-set schema")
        if self.territory != ER_LISTING_LABEL_TERRITORY:
            raise ListingLabelSetError("listing-label set must be scoped to Singapore")
        if self.domain != ER_LISTING_LABEL_DOMAIN:
            raise ListingLabelSetError("listing-label set must contain PC components")
        if self.label_source != ER_LISTING_LABEL_SOURCE:
            raise ListingLabelSetError("listing-label set must use independent human review")
        if self.review_protocol != ER_LISTING_REVIEW_PROTOCOL:
            raise ListingLabelSetError("unsupported independent-review protocol")
        _text(self.dataset_version, name="dataset_version")
        _timestamp(self.created_at, name="created_at")
        _sha256(self.source_review_queue_sha256, name="source_review_queue_sha256")
        _text(self.canonical_catalogue_version, name="canonical_catalogue_version")
        _sha256(self.canonical_catalogue_sha256, name="canonical_catalogue_sha256")
        _sha256(
            self.canonical_catalogue_file_sha256,
            name="canonical_catalogue_file_sha256",
        )
        _sha256(self.dataset_sha256, name="dataset_sha256")
        if self.source_policy.data_version != self.dataset_version:
            raise ListingLabelSetError("source policy data_version does not match dataset")
        if not self.products or not self.listing_groups:
            raise ListingLabelSetError("label set requires products and listing groups")
        product_ids = [product.product_id for product in self.products]
        listing_ids = [group.listing.listing_id for group in self.listing_groups]
        if len(product_ids) != len(set(product_ids)):
            raise ListingLabelSetError("label set contains duplicate canonical products")
        if len(listing_ids) != len(set(listing_ids)):
            raise ListingLabelSetError("label set contains duplicate listing groups")
        if product_ids != sorted(product_ids):
            raise ListingLabelSetError("canonical products must be sorted by product_id")
        if listing_ids != sorted(listing_ids):
            raise ListingLabelSetError("listing groups must be sorted by listing_id")
        if any(product.is_synthetic for product in self.products) or any(
            group.listing.is_synthetic for group in self.listing_groups
        ):
            raise ListingLabelSetError("synthetic records cannot enter human promotion evidence")
        if (
            canonical_catalogue_sha256(
                self.canonical_catalogue_version,
                self.products,
            )
            != self.canonical_catalogue_sha256
        ):
            raise ListingLabelSetError(
                "label products do not match the bound canonical catalogue release"
            )
        product_by_id = {product.product_id: product for product in self.products}
        for group in self.listing_groups:
            listing_category = canonical_pc_category(group.listing.category)
            if listing_category is None:
                raise ListingLabelSetError("listing group has unsupported PC category")
            for pair in group.pair_labels:
                product = product_by_id.get(pair.product_id)
                if product is None:
                    raise ListingLabelSetError(
                        f"labelled product {pair.product_id!r} is absent from frozen catalogue"
                    )
                if canonical_pc_category(product.category) != listing_category:
                    raise ListingLabelSetError("labelled pair crosses component categories")
        if sha256_json(self.content_payload()) != self.dataset_sha256:
            raise ListingLabelSetError("listing-label-set self-hash mismatch")

    @property
    def labelled_pair_count(self) -> int:
        return sum(len(group.pair_labels) for group in self.listing_groups)

    @property
    def independent_reviewer_count(self) -> int:
        reviewers = {
            judgment.reviewer_id
            for group in self.listing_groups
            for pair in group.pair_labels
            for judgment in (
                *pair.judgments,
                *((pair.adjudication,) if pair.adjudication is not None else ()),
            )
        }
        return len(reviewers)

    def content_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "dataset_version": self.dataset_version,
            "territory": self.territory,
            "domain": self.domain,
            "label_source": self.label_source,
            "review_protocol": self.review_protocol,
            "created_at": self.created_at,
            "source_review_queue_sha256": self.source_review_queue_sha256,
            "canonical_catalogue_version": self.canonical_catalogue_version,
            "canonical_catalogue_sha256": self.canonical_catalogue_sha256,
            "canonical_catalogue_file_sha256": self.canonical_catalogue_file_sha256,
            "source_policy": self.source_policy.to_dict(),
            "products": [product.to_dict() for product in self.products],
            "listing_groups": [group.to_dict() for group in self.listing_groups],
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.content_payload(), "dataset_sha256": self.dataset_sha256}


def load_frozen_listing_label_set(path: str | Path) -> FrozenListingLabelSet:
    """Load a content-addressed label set without inferring missing judgments."""

    source = Path(path).resolve(strict=True)
    if source.stat().st_size > _MAX_LABEL_SET_BYTES:
        raise ListingLabelSetError("listing-label set exceeds the 128 MiB safety limit")
    try:
        payload: Any = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ListingLabelSetError("listing-label set must be valid UTF-8 JSON") from error
    root = _object(payload, name="listing-label set")
    _exact_fields(root, _LABEL_SET_FIELDS, name="listing-label set")
    source_policy_payload = _object(root["source_policy"], name="source_policy")
    _exact_fields(source_policy_payload, _SOURCE_POLICY_FIELDS, name="source_policy")
    raw_products = _array(root["products"], name="products")
    raw_groups = _array(root["listing_groups"], name="listing_groups")
    return FrozenListingLabelSet(
        schema_version=_text(root["schema_version"], name="schema_version"),
        dataset_version=_text(root["dataset_version"], name="dataset_version"),
        territory=_text(root["territory"], name="territory"),
        domain=_text(root["domain"], name="domain"),
        label_source=_text(root["label_source"], name="label_source"),
        review_protocol=_text(root["review_protocol"], name="review_protocol"),
        created_at=_timestamp(root["created_at"], name="created_at"),
        source_review_queue_sha256=_sha256(
            root["source_review_queue_sha256"], name="source_review_queue_sha256"
        ),
        canonical_catalogue_version=_text(
            root["canonical_catalogue_version"], name="canonical_catalogue_version"
        ),
        canonical_catalogue_sha256=_sha256(
            root["canonical_catalogue_sha256"], name="canonical_catalogue_sha256"
        ),
        canonical_catalogue_file_sha256=_sha256(
            root["canonical_catalogue_file_sha256"],
            name="canonical_catalogue_file_sha256",
        ),
        source_policy=SourceUsePolicy.from_dict(source_policy_payload),
        products=tuple(
            CanonicalProductRecord.from_dict(_object(item, name=f"products[{index}]"))
            for index, item in enumerate(raw_products)
        ),
        listing_groups=tuple(
            FrozenListingLabelGroup.from_dict(item, name=f"listing_groups[{index}]")
            for index, item in enumerate(raw_groups)
        ),
        dataset_sha256=_sha256(root["dataset_sha256"], name="dataset_sha256"),
    )


def load_frozen_canonical_catalogue(path: str | Path) -> FrozenCanonicalCatalogue:
    """Load the exact, complete canonical catalogue release used for ER replay."""

    source = Path(path).resolve(strict=True)
    if source.stat().st_size > _MAX_LABEL_SET_BYTES:
        raise ListingLabelSetError("canonical catalogue exceeds the 128 MiB safety limit")
    try:
        payload: Any = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ListingLabelSetError("canonical catalogue must be valid UTF-8 JSON") from error
    root = _object(payload, name="canonical catalogue")
    _exact_fields(root, _CATALOGUE_FIELDS, name="canonical catalogue")
    raw_products = _array(root["products"], name="canonical catalogue.products")
    return FrozenCanonicalCatalogue(
        schema_version=_text(root["schema_version"], name="schema_version"),
        catalogue_version=_text(root["catalogue_version"], name="catalogue_version"),
        products=tuple(
            CanonicalProductRecord.from_dict(
                _object(item, name=f"canonical catalogue.products[{index}]")
            )
            for index, item in enumerate(raw_products)
        ),
        catalogue_sha256=_sha256(root["catalogue_sha256"], name="catalogue_sha256"),
    )
