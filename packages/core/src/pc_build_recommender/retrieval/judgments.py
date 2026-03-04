"""Human relevance-label schema and deterministic adjudication tooling."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .evaluation import (
    FrozenCandidateQuery,
    PinnedCandidateSet,
    RelevanceLabelSource,
)

HUMAN_LABEL_SCHEMA_VERSION = "pc-build-recommender.human-relevance-labels.v1"


def _normalise_utc(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid ISO-8601 timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        raise ValueError("human-label timestamps must be timezone-aware")
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _content_sha256(payload: object) -> str:
    serialised = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(serialised).hexdigest()


def _relevance_grade(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > 4:
        raise ValueError(f"{field_name} must be an integer from 0 to 4")
    return value


@dataclass(frozen=True, slots=True)
class LabelingQuery:
    query_id: str
    query_group_id: str
    query_text: str
    category: str
    candidate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not all((self.query_id, self.query_group_id, self.query_text, self.category)):
            raise ValueError("labeling query fields must not be empty")
        if not self.candidate_ids:
            raise ValueError("a labeling query must contain candidates")
        if len(self.candidate_ids) != len(set(self.candidate_ids)):
            raise ValueError("candidate IDs must be unique within a labeling query")
        object.__setattr__(self, "category", self.category.casefold())

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> LabelingQuery:
        candidate_ids = payload.get("candidate_ids")
        if not isinstance(candidate_ids, list):
            raise TypeError("candidate_ids must be a list")
        return cls(
            query_id=str(payload["query_id"]),
            query_group_id=str(payload["query_group_id"]),
            query_text=str(payload["query_text"]),
            category=str(payload["category"]),
            candidate_ids=tuple(str(item) for item in candidate_ids),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "query_group_id": self.query_group_id,
            "query_text": self.query_text,
            "category": self.category,
            "candidate_ids": list(self.candidate_ids),
        }


@dataclass(frozen=True, slots=True)
class ReviewerJudgment:
    query_id: str
    product_id: str
    reviewer_id: str
    grade: int
    rationale: str
    reviewed_at_utc: str
    is_synthetic: bool = False

    def __post_init__(self) -> None:
        if not all((self.query_id, self.product_id, self.reviewer_id, self.rationale)):
            raise ValueError("review judgment identifiers and rationale must not be empty")
        _relevance_grade(self.grade, field_name="reviewer grade")
        object.__setattr__(self, "reviewed_at_utc", _normalise_utc(self.reviewed_at_utc))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ReviewerJudgment:
        return cls(
            query_id=str(payload["query_id"]),
            product_id=str(payload["product_id"]),
            reviewer_id=str(payload["reviewer_id"]),
            grade=_relevance_grade(payload["grade"], field_name="reviewer grade"),
            rationale=str(payload["rationale"]),
            reviewed_at_utc=str(payload["reviewed_at_utc"]),
            is_synthetic=bool(payload.get("is_synthetic", False)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "product_id": self.product_id,
            "reviewer_id": self.reviewer_id,
            "grade": self.grade,
            "rationale": self.rationale,
            "reviewed_at_utc": self.reviewed_at_utc,
            "is_synthetic": self.is_synthetic,
        }


@dataclass(frozen=True, slots=True)
class AdjudicationDecision:
    query_id: str
    product_id: str
    adjudicator_id: str
    grade: int
    rationale: str
    adjudicated_at_utc: str

    def __post_init__(self) -> None:
        if not all((self.query_id, self.product_id, self.adjudicator_id, self.rationale)):
            raise ValueError("adjudication identifiers and rationale must not be empty")
        _relevance_grade(self.grade, field_name="adjudicated grade")
        object.__setattr__(
            self,
            "adjudicated_at_utc",
            _normalise_utc(self.adjudicated_at_utc),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> AdjudicationDecision:
        return cls(
            query_id=str(payload["query_id"]),
            product_id=str(payload["product_id"]),
            adjudicator_id=str(payload["adjudicator_id"]),
            grade=_relevance_grade(payload["grade"], field_name="adjudicated grade"),
            rationale=str(payload["rationale"]),
            adjudicated_at_utc=str(payload["adjudicated_at_utc"]),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "product_id": self.product_id,
            "adjudicator_id": self.adjudicator_id,
            "grade": self.grade,
            "rationale": self.rationale,
            "adjudicated_at_utc": self.adjudicated_at_utc,
        }


@dataclass(frozen=True, slots=True)
class AdjudicationSummary:
    candidate_pair_count: int
    unanimous_pair_count: int
    adjudicated_pair_count: int
    exact_agreement_rate: float
    reviewer_judgment_count: int
    contains_synthetic_labels: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_pair_count": self.candidate_pair_count,
            "unanimous_pair_count": self.unanimous_pair_count,
            "adjudicated_pair_count": self.adjudicated_pair_count,
            "exact_agreement_rate": self.exact_agreement_rate,
            "reviewer_judgment_count": self.reviewer_judgment_count,
            "contains_synthetic_labels": self.contains_synthetic_labels,
        }


@dataclass(frozen=True, slots=True)
class AdjudicatedHumanLabelSet:
    frozen_candidates: PinnedCandidateSet
    summary: AdjudicationSummary
    judgment_manifest_sha256: str


@dataclass(frozen=True, slots=True)
class HumanJudgmentSet:
    dataset_name: str
    dataset_version: str
    queries: tuple[LabelingQuery, ...]
    judgments: tuple[ReviewerJudgment, ...]
    adjudications: tuple[AdjudicationDecision, ...]

    def __post_init__(self) -> None:
        if not self.dataset_name or not self.dataset_version:
            raise ValueError("human judgment dataset metadata must not be empty")
        if not self.queries or not self.judgments:
            raise ValueError("human judgment datasets need queries and judgments")
        query_ids = [query.query_id for query in self.queries]
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("human judgment query IDs must be unique")

    def content_payload(self) -> dict[str, object]:
        return {
            "schema_version": HUMAN_LABEL_SCHEMA_VERSION,
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "label_source": RelevanceLabelSource.HUMAN.value,
            "queries": [query.to_dict() for query in self.queries],
            "judgments": [
                judgment.to_dict()
                for judgment in sorted(
                    self.judgments,
                    key=lambda item: (item.query_id, item.product_id, item.reviewer_id),
                )
            ],
            "adjudications": [
                decision.to_dict()
                for decision in sorted(
                    self.adjudications,
                    key=lambda item: (item.query_id, item.product_id),
                )
            ],
        }

    @property
    def content_sha256(self) -> str:
        return _content_sha256(self.content_payload())

    def adjudicate(self) -> AdjudicatedHumanLabelSet:
        query_by_id = {query.query_id: query for query in self.queries}
        expected_pairs = {
            (query.query_id, product_id)
            for query in self.queries
            for product_id in query.candidate_ids
        }
        grouped: dict[tuple[str, str], list[ReviewerJudgment]] = defaultdict(list)
        seen_reviews: set[tuple[str, str, str]] = set()
        for judgment in self.judgments:
            pair = (judgment.query_id, judgment.product_id)
            if pair not in expected_pairs:
                raise ValueError(f"judgment references an unknown candidate pair: {pair!r}")
            review_key = (*pair, judgment.reviewer_id)
            if review_key in seen_reviews:
                raise ValueError(f"reviewer submitted a duplicate judgment: {review_key!r}")
            seen_reviews.add(review_key)
            grouped[pair].append(judgment)

        missing_pairs = expected_pairs - set(grouped)
        if missing_pairs:
            example = sorted(missing_pairs)[0]
            raise ValueError(f"candidate pair has no human judgments: {example!r}")

        decisions: dict[tuple[str, str], AdjudicationDecision] = {}
        for adjudication in self.adjudications:
            pair = (adjudication.query_id, adjudication.product_id)
            if pair not in expected_pairs:
                raise ValueError(f"adjudication references an unknown candidate pair: {pair!r}")
            if pair in decisions:
                raise ValueError(f"candidate pair has duplicate adjudications: {pair!r}")
            decisions[pair] = adjudication

        final_labels: dict[str, dict[str, int]] = defaultdict(dict)
        unanimous_count = 0
        adjudicated_count = 0
        contains_synthetic = False
        for pair in sorted(expected_pairs):
            reviews = grouped[pair]
            reviewer_ids = {review.reviewer_id for review in reviews}
            if len(reviewer_ids) < 2:
                raise ValueError(f"candidate pair needs two distinct reviewers: {pair!r}")
            contains_synthetic = contains_synthetic or any(
                review.is_synthetic for review in reviews
            )
            grades = {review.grade for review in reviews}
            resolved_decision = decisions.get(pair)
            if len(grades) == 1:
                if resolved_decision is not None:
                    raise ValueError(f"unanimous candidate pair has an adjudication: {pair!r}")
                final_grade = next(iter(grades))
                unanimous_count += 1
            else:
                if resolved_decision is None:
                    raise ValueError(f"disagreed candidate pair needs adjudication: {pair!r}")
                if resolved_decision.adjudicator_id in reviewer_ids:
                    raise ValueError(
                        f"adjudicator must be independent for candidate pair: {pair!r}"
                    )
                final_grade = resolved_decision.grade
                adjudicated_count += 1
            final_labels[pair[0]][pair[1]] = final_grade

        unused_decisions = set(decisions) - {
            pair for pair, reviews in grouped.items() if len({item.grade for item in reviews}) > 1
        }
        if unused_decisions:
            raise ValueError("adjudications may only resolve reviewer disagreement")

        frozen_queries: list[FrozenCandidateQuery] = []
        for query_id in sorted(query_by_id):
            query = query_by_id[query_id]
            labels = final_labels[query_id]
            if not any(grade > 0 for grade in labels.values()):
                raise ValueError(f"query {query_id!r} has no relevant product after adjudication")
            frozen_queries.append(
                FrozenCandidateQuery(
                    query_id=query.query_id,
                    candidate_ids=query.candidate_ids,
                    relevance_labels=labels,
                    query_text=query.query_text,
                    category=query.category,
                    query_group_id=query.query_group_id,
                )
            )

        manifest_sha256 = self.content_sha256
        frozen = PinnedCandidateSet.create(
            self.dataset_version,
            frozen_queries,
            label_source=RelevanceLabelSource.HUMAN,
            adjudication_complete=True,
            contains_synthetic_labels=contains_synthetic,
            judgment_manifest_sha256=manifest_sha256,
        )
        pair_count = len(expected_pairs)
        summary = AdjudicationSummary(
            candidate_pair_count=pair_count,
            unanimous_pair_count=unanimous_count,
            adjudicated_pair_count=adjudicated_count,
            exact_agreement_rate=unanimous_count / pair_count,
            reviewer_judgment_count=len(self.judgments),
            contains_synthetic_labels=contains_synthetic,
        )
        return AdjudicatedHumanLabelSet(
            frozen_candidates=frozen,
            summary=summary,
            judgment_manifest_sha256=manifest_sha256,
        )

    @property
    def average_judgments_per_pair(self) -> float:
        pair_count = sum(len(query.candidate_ids) for query in self.queries)
        return len(self.judgments) / pair_count


def human_judgment_set_from_mapping(payload: Mapping[str, Any]) -> HumanJudgmentSet:
    if payload.get("schema_version") != HUMAN_LABEL_SCHEMA_VERSION:
        raise ValueError("unsupported human relevance-label schema")
    if payload.get("label_source") != RelevanceLabelSource.HUMAN.value:
        raise ValueError("human judgment files must declare label_source='human'")
    queries = payload.get("queries")
    judgments = payload.get("judgments")
    adjudications = payload.get("adjudications", [])
    if not isinstance(queries, list) or not isinstance(judgments, list):
        raise TypeError("queries and judgments must be lists")
    if not isinstance(adjudications, list):
        raise TypeError("adjudications must be a list")
    return HumanJudgmentSet(
        dataset_name=str(payload["dataset_name"]),
        dataset_version=str(payload["dataset_version"]),
        queries=tuple(LabelingQuery.from_mapping(item) for item in queries),
        judgments=tuple(ReviewerJudgment.from_mapping(item) for item in judgments),
        adjudications=tuple(AdjudicationDecision.from_mapping(item) for item in adjudications),
    )


def load_human_judgment_set(path: str | Path) -> HumanJudgmentSet:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("human judgment file root must be an object")
    return human_judgment_set_from_mapping(payload)


def write_human_judgment_set(dataset: HumanJudgmentSet, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(dataset.content_payload(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return target
