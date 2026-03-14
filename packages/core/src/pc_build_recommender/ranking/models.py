"""Storage-agnostic contracts for component learning-to-rank."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from pc_build_recommender.retrieval import RetrievedCandidate


def _finite_mapping(name: str, values: Mapping[str, float]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in values.items():
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{name}[{key!r}] must be finite")
        result[str(key)] = numeric
    return result


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    """A filtered component plus soft evidence used for ranking.

    There is deliberately no ``compatible`` feature.  Hard compatibility is an
    upstream eligibility decision; incompatible products must never be passed
    to a ranker.
    """

    product_id: str
    category: str
    price_sgd: float | None = None
    brand: str | None = None
    retrieval_scores: Mapping[str, float] = field(default_factory=dict)
    workload_scores: Mapping[str, float] = field(default_factory=dict)
    signals: Mapping[str, float] = field(default_factory=dict)
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.product_id:
            raise ValueError("product_id must not be empty")
        if not self.category:
            raise ValueError("category must not be empty")
        if self.price_sgd is not None and (not math.isfinite(self.price_sgd) or self.price_sgd < 0):
            raise ValueError("price_sgd must be finite and non-negative")
        forbidden = {key for key in self.signals if "compatib" in key.casefold()}
        if forbidden:
            raise ValueError(
                "hard compatibility must be filtered before ranking; "
                f"forbidden signals: {sorted(forbidden)}"
            )
        object.__setattr__(self, "category", self.category.casefold())
        object.__setattr__(
            self, "retrieval_scores", _finite_mapping("retrieval_scores", self.retrieval_scores)
        )
        object.__setattr__(
            self, "workload_scores", _finite_mapping("workload_scores", self.workload_scores)
        )
        object.__setattr__(self, "signals", _finite_mapping("signals", self.signals))
        object.__setattr__(self, "attributes", dict(self.attributes))

    @classmethod
    def from_retrieved(
        cls,
        candidate: RetrievedCandidate,
        *,
        workload_scores: Mapping[str, float] | None = None,
        signals: Mapping[str, float] | None = None,
    ) -> ScoredCandidate:
        return cls(
            product_id=candidate.product_id,
            category=candidate.product.category,
            price_sgd=candidate.product.price_sgd,
            brand=candidate.product.brand,
            retrieval_scores={
                "bm25_score": candidate.bm25_score,
                "lexical_score": candidate.lexical_score,
                "vector_similarity": candidate.vector_similarity,
                "rrf_score": candidate.rrf_score,
            },
            workload_scores=workload_scores or {},
            signals=signals or {},
            attributes=candidate.product.attributes,
        )

    def get(self, name: str, default: Any = None) -> Any:
        if name == "product_id":
            return self.product_id
        if name == "category":
            return self.category
        if name == "price_sgd":
            return self.price_sgd
        if name == "brand":
            return self.brand
        current: Any = self.attributes
        for part in name.split("."):
            if not isinstance(current, Mapping) or part not in current:
                return default
            current = current[part]
        return current


@dataclass(frozen=True, slots=True)
class RankingContext:
    """Query-level values used to calculate preference and value features."""

    query_id: str
    query_text: str = ""
    budget_sgd: float | None = None
    workload_weights: Mapping[str, float] = field(default_factory=dict)
    requirements: Mapping[str, Any] = field(default_factory=dict)
    preferences: Mapping[str, Any] = field(default_factory=dict)
    data_version: str | None = None
    candidate_set_version: str | None = None

    def __post_init__(self) -> None:
        if not self.query_id:
            raise ValueError("query_id must not be empty")
        if self.budget_sgd is not None and (
            not math.isfinite(self.budget_sgd) or self.budget_sgd <= 0
        ):
            raise ValueError("budget_sgd must be finite and positive")
        weights = _finite_mapping("workload_weights", self.workload_weights)
        if any(value < 0 for value in weights.values()):
            raise ValueError("workload weights must be non-negative")
        if weights and sum(weights.values()) <= 0:
            raise ValueError("at least one workload weight must be positive")
        object.__setattr__(self, "workload_weights", weights)
        object.__setattr__(self, "requirements", dict(self.requirements))
        object.__setattr__(self, "preferences", dict(self.preferences))


@dataclass(frozen=True, slots=True)
class RankingQuery:
    context: RankingContext
    candidates: tuple[ScoredCandidate, ...]

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ValueError("a ranking query must contain candidates")
        ids = [candidate.product_id for candidate in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate product IDs must be unique within a query")

    @classmethod
    def create(
        cls, context: RankingContext, candidates: Sequence[ScoredCandidate]
    ) -> RankingQuery:
        return cls(context=context, candidates=tuple(candidates))


@dataclass(frozen=True, slots=True)
class LabeledRankingQuery:
    context: RankingContext
    candidates: tuple[ScoredCandidate, ...]
    relevance_grades: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ValueError("a labeled query must contain candidates")
        if len(self.candidates) != len(self.relevance_grades):
            raise ValueError("one relevance grade is required per candidate")
        if len({candidate.product_id for candidate in self.candidates}) != len(self.candidates):
            raise ValueError("candidate product IDs must be unique within a query")
        if any(grade < 0 or grade > 4 for grade in self.relevance_grades):
            raise ValueError("relevance grades must be integers from 0 to 4")

    @classmethod
    def create(
        cls,
        context: RankingContext,
        candidates: Sequence[ScoredCandidate],
        relevance_grades: Sequence[int],
    ) -> LabeledRankingQuery:
        return cls(
            context=context,
            candidates=tuple(candidates),
            relevance_grades=tuple(int(grade) for grade in relevance_grades),
        )


@dataclass(frozen=True, slots=True)
class RankerArtifactIdentity:
    """Exact digests for the committed model, metadata, and manifest bytes."""

    model_sha256: str
    metadata_sha256: str
    manifest_sha256: str

    def __post_init__(self) -> None:
        for name, digest in (
            ("model_sha256", self.model_sha256),
            ("metadata_sha256", self.metadata_sha256),
            ("manifest_sha256", self.manifest_sha256),
        ):
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")

    def to_dict(self) -> dict[str, str]:
        return {
            "model_sha256": self.model_sha256,
            "metadata_sha256": self.metadata_sha256,
            "manifest_sha256": self.manifest_sha256,
        }


@dataclass(frozen=True, slots=True)
class RankerMetadata:
    """Version and evidence basis emitted with every ranking."""

    ranker_version: str
    ranking_basis: str
    feature_version: str
    model_type: str
    feature_names: tuple[str, ...]
    created_at_utc: str
    training_data_version: str | None = None
    candidate_set_version: str | None = None
    training_query_count: int = 0
    training_row_count: int = 0
    parameters: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, float] = field(default_factory=dict)
    training_label_source: str = "unverified"
    training_adjudication_complete: bool = False
    contains_synthetic_labels: bool = False
    training_judgment_manifest_sha256: str | None = None
    training_dataset_manifest_sha256: str | None = None
    training_prelabel_snapshot_sha256: str | None = None
    training_feature_contract_sha256: str | None = None
    query_group_split_checksum: str | None = None
    query_split_membership_verified: bool = False
    model_sha256: str | None = None
    promotion_eligible: bool = False
    promotion_block_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", _finite_mapping("metrics", self.metrics))
        for name, boolean_value in (
            ("training_adjudication_complete", self.training_adjudication_complete),
            ("contains_synthetic_labels", self.contains_synthetic_labels),
            ("query_split_membership_verified", self.query_split_membership_verified),
            ("promotion_eligible", self.promotion_eligible),
        ):
            if type(boolean_value) is not bool:
                raise TypeError(f"{name} must be a boolean")
        allowed_sources = {"human", "silver", "synthetic", "unverified"}
        if self.training_label_source not in allowed_sources:
            raise ValueError("unsupported training_label_source")
        for name, digest in (
            ("training_judgment_manifest_sha256", self.training_judgment_manifest_sha256),
            ("training_dataset_manifest_sha256", self.training_dataset_manifest_sha256),
            ("training_prelabel_snapshot_sha256", self.training_prelabel_snapshot_sha256),
            ("training_feature_contract_sha256", self.training_feature_contract_sha256),
            ("query_group_split_checksum", self.query_group_split_checksum),
            ("model_sha256", self.model_sha256),
        ):
            if digest is not None and (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if self.promotion_eligible:
            if self.training_label_source != "human":
                raise ValueError("only human relevance labels can be promotion-eligible")
            if not self.training_adjudication_complete:
                raise ValueError("promotion-eligible labels must be fully adjudicated")
            if self.contains_synthetic_labels:
                raise ValueError("promotion-eligible labels cannot contain synthetic rows")
            if self.training_judgment_manifest_sha256 is None:
                raise ValueError("promotion-eligible metadata needs a judgment manifest hash")
            if self.training_dataset_manifest_sha256 is None:
                raise ValueError("promotion-eligible metadata needs a dataset manifest hash")
            if self.training_prelabel_snapshot_sha256 is None:
                raise ValueError("promotion-eligible metadata needs a pre-label snapshot hash")
            if self.training_feature_contract_sha256 is None:
                raise ValueError("promotion-eligible metadata needs a feature contract hash")
            if self.query_group_split_checksum is None:
                raise ValueError("promotion-eligible metadata needs a query-group split hash")
            if not self.query_split_membership_verified:
                raise ValueError("promotion-eligible metadata needs verified split membership")
            if self.promotion_block_reasons:
                raise ValueError("promotion-eligible metadata cannot have block reasons")

    def to_dict(self) -> dict[str, Any]:
        return {
            "ranker_version": self.ranker_version,
            "ranking_basis": self.ranking_basis,
            "feature_version": self.feature_version,
            "model_type": self.model_type,
            "feature_names": list(self.feature_names),
            "created_at_utc": self.created_at_utc,
            "training_data_version": self.training_data_version,
            "candidate_set_version": self.candidate_set_version,
            "training_query_count": self.training_query_count,
            "training_row_count": self.training_row_count,
            "parameters": dict(self.parameters),
            "metrics": dict(self.metrics),
            "training_label_source": self.training_label_source,
            "training_adjudication_complete": self.training_adjudication_complete,
            "contains_synthetic_labels": self.contains_synthetic_labels,
            "training_judgment_manifest_sha256": self.training_judgment_manifest_sha256,
            "training_dataset_manifest_sha256": self.training_dataset_manifest_sha256,
            "training_prelabel_snapshot_sha256": self.training_prelabel_snapshot_sha256,
            "training_feature_contract_sha256": self.training_feature_contract_sha256,
            "query_group_split_checksum": self.query_group_split_checksum,
            "query_split_membership_verified": self.query_split_membership_verified,
            "model_sha256": self.model_sha256,
            "promotion_eligible": self.promotion_eligible,
            "promotion_block_reasons": list(self.promotion_block_reasons),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RankerMetadata:
        return cls(
            ranker_version=str(payload["ranker_version"]),
            ranking_basis=str(payload["ranking_basis"]),
            feature_version=str(payload["feature_version"]),
            model_type=str(payload["model_type"]),
            feature_names=tuple(payload["feature_names"]),
            created_at_utc=str(payload["created_at_utc"]),
            training_data_version=payload.get("training_data_version"),
            candidate_set_version=payload.get("candidate_set_version"),
            training_query_count=int(payload.get("training_query_count", 0)),
            training_row_count=int(payload.get("training_row_count", 0)),
            parameters=dict(payload.get("parameters", {})),
            metrics={key: float(value) for key, value in payload.get("metrics", {}).items()},
            training_label_source=str(payload.get("training_label_source", "unverified")),
            training_adjudication_complete=payload.get("training_adjudication_complete", False),
            contains_synthetic_labels=payload.get("contains_synthetic_labels", False),
            training_judgment_manifest_sha256=payload.get("training_judgment_manifest_sha256"),
            training_dataset_manifest_sha256=payload.get("training_dataset_manifest_sha256"),
            training_prelabel_snapshot_sha256=payload.get(
                "training_prelabel_snapshot_sha256"
            ),
            training_feature_contract_sha256=payload.get(
                "training_feature_contract_sha256"
            ),
            query_group_split_checksum=payload.get("query_group_split_checksum"),
            query_split_membership_verified=payload.get("query_split_membership_verified", False),
            model_sha256=payload.get("model_sha256"),
            promotion_eligible=payload.get("promotion_eligible", False),
            promotion_block_reasons=tuple(payload.get("promotion_block_reasons", ())),
        )


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    candidate: ScoredCandidate
    score: float
    rank: int
    ranker_version: str
    ranking_basis: str
    feature_contributions: Mapping[str, float] = field(default_factory=dict)

    @property
    def product_id(self) -> str:
        return self.candidate.product_id


@runtime_checkable
class ProductRanker(Protocol):
    """Interface consumed by API/application orchestration."""

    @property
    def metadata(self) -> RankerMetadata:
        """Return immutable version and evidence-basis metadata."""

    @property
    def artifact_identity(self) -> RankerArtifactIdentity:
        """Return exact persisted artifact digests or raise when unpersisted."""

    @property
    def verified_artifact_loaded(self) -> bool:
        """Whether scoring uses bytes loaded through the verified artifact manifest."""

    def rank_query(
        self, context: RankingContext, candidates: Sequence[ScoredCandidate]
    ) -> list[RankedCandidate]:
        """Rank a pre-filtered candidate set."""
