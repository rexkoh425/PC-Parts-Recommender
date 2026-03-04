"""Frozen-candidate contracts and standard retrieval/ranking metrics."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from pc_build_recommender.evaluation.contracts import MetricEstimate
from pc_build_recommender.evaluation.metrics import bootstrap_confidence_interval


class RelevanceLabelSource(StrEnum):
    """Evidence tier for relevance grades.

    Only fully adjudicated human judgments can become promotion evidence.  Silver,
    synthetic, and legacy/unverified grades remain useful for diagnostics.
    """

    HUMAN = "human"
    SILVER = "silver"
    SYNTHETIC = "synthetic"
    UNVERIFIED = "unverified"


def _stable_payload(version: str, queries: Sequence[FrozenCandidateQuery]) -> bytes:
    payload = {
        "version": version,
        "queries": [query.to_dict() for query in queries],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True, slots=True)
class FrozenCandidateQuery:
    """One judged query whose candidate universe must not change between models."""

    query_id: str
    candidate_ids: tuple[str, ...]
    relevance_labels: Mapping[str, int]
    query_text: str = ""
    category: str | None = None
    query_group_id: str | None = None

    def __post_init__(self) -> None:
        if not self.query_id:
            raise ValueError("query_id must not be empty")
        if not self.candidate_ids:
            raise ValueError("candidate_ids must not be empty")
        if len(self.candidate_ids) != len(set(self.candidate_ids)):
            raise ValueError("candidate_ids must be unique")
        unknown = set(self.relevance_labels) - set(self.candidate_ids)
        if unknown:
            raise ValueError(f"labels reference candidates outside the frozen set: {unknown}")
        labels: dict[str, int] = {}
        for product_id, grade in self.relevance_labels.items():
            if isinstance(grade, bool) or not isinstance(grade, int) or not 0 <= grade <= 4:
                raise ValueError("relevance grades must be integers between 0 and 4")
            labels[product_id] = grade
        if not any(grade > 0 for grade in labels.values()):
            raise ValueError("each query needs at least one relevant candidate")
        object.__setattr__(self, "candidate_ids", tuple(self.candidate_ids))
        object.__setattr__(self, "relevance_labels", labels)
        if self.category is not None:
            object.__setattr__(self, "category", self.category.casefold())
        if self.query_group_id is not None and not self.query_group_id:
            raise ValueError("query_group_id must not be empty when supplied")

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "query_id": self.query_id,
            "candidate_ids": list(self.candidate_ids),
            "relevance_labels": dict(sorted(self.relevance_labels.items())),
            "query_text": self.query_text,
            "category": self.category,
        }
        # Omitting ``None`` keeps legacy candidate checksums loadable.
        if self.query_group_id is not None:
            payload["query_group_id"] = self.query_group_id
        return payload


@dataclass(frozen=True, slots=True)
class PinnedCandidateSet:
    """Versioned, checksummed relevance set shared by every retrieval baseline."""

    version: str
    queries: tuple[FrozenCandidateQuery, ...]
    checksum: str
    label_source: RelevanceLabelSource = RelevanceLabelSource.UNVERIFIED
    adjudication_complete: bool = False
    contains_synthetic_labels: bool = False
    judgment_manifest_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.version:
            raise ValueError("version must not be empty")
        if not self.queries:
            raise ValueError("queries must not be empty")
        query_ids = [query.query_id for query in self.queries]
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("query_id values must be unique")
        expected = hashlib.sha256(_stable_payload(self.version, self.queries)).hexdigest()
        if self.checksum != expected:
            raise ValueError("frozen candidate checksum does not match its contents")
        object.__setattr__(self, "label_source", RelevanceLabelSource(self.label_source))
        if self.judgment_manifest_sha256 is not None and (
            len(self.judgment_manifest_sha256) != 64
            or any(
                character not in "0123456789abcdef" for character in self.judgment_manifest_sha256
            )
        ):
            raise ValueError("judgment_manifest_sha256 must be a lowercase SHA-256 digest")
        if self.adjudication_complete and self.label_source is not RelevanceLabelSource.HUMAN:
            raise ValueError("only human relevance labels can be declared adjudication-complete")

    @property
    def eligible_for_promotion(self) -> bool:
        return (
            self.label_source is RelevanceLabelSource.HUMAN
            and self.adjudication_complete
            and not self.contains_synthetic_labels
            and self.judgment_manifest_sha256 is not None
        )

    @property
    def promotion_block_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if self.label_source is not RelevanceLabelSource.HUMAN:
            reasons.append(f"label source is {self.label_source.value}, not human")
        if not self.adjudication_complete:
            reasons.append("human relevance judgments are not fully adjudicated")
        if self.contains_synthetic_labels:
            reasons.append("synthetic relevance labels are present")
        if self.judgment_manifest_sha256 is None:
            reasons.append("human judgment manifest hash is missing")
        return tuple(reasons)

    @property
    def evidence_checksum(self) -> str:
        payload = {
            "candidate_checksum": self.checksum,
            "label_source": self.label_source.value,
            "adjudication_complete": self.adjudication_complete,
            "contains_synthetic_labels": self.contains_synthetic_labels,
            "judgment_manifest_sha256": self.judgment_manifest_sha256,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    @classmethod
    def create(
        cls,
        version: str,
        queries: Sequence[FrozenCandidateQuery],
        *,
        label_source: RelevanceLabelSource = RelevanceLabelSource.UNVERIFIED,
        adjudication_complete: bool = False,
        contains_synthetic_labels: bool = False,
        judgment_manifest_sha256: str | None = None,
    ) -> PinnedCandidateSet:
        query_tuple = tuple(queries)
        checksum = hashlib.sha256(_stable_payload(version, query_tuple)).hexdigest()
        return cls(
            version=version,
            queries=query_tuple,
            checksum=checksum,
            label_source=label_source,
            adjudication_complete=adjudication_complete,
            contains_synthetic_labels=contains_synthetic_labels,
            judgment_manifest_sha256=judgment_manifest_sha256,
        )

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {
                    "version": self.version,
                    "checksum": self.checksum,
                    "evidence_checksum": self.evidence_checksum,
                    "evidence": {
                        "label_source": self.label_source.value,
                        "adjudication_complete": self.adjudication_complete,
                        "contains_synthetic_labels": self.contains_synthetic_labels,
                        "judgment_manifest_sha256": self.judgment_manifest_sha256,
                        "eligible_for_promotion": self.eligible_for_promotion,
                        "promotion_block_reasons": list(self.promotion_block_reasons),
                    },
                    "queries": [query.to_dict() for query in self.queries],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> PinnedCandidateSet:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        queries = tuple(
            FrozenCandidateQuery(
                query_id=item["query_id"],
                candidate_ids=tuple(item["candidate_ids"]),
                relevance_labels=item["relevance_labels"],
                query_text=item.get("query_text", ""),
                category=item.get("category"),
                query_group_id=item.get("query_group_id"),
            )
            for item in payload["queries"]
        )
        evidence = payload.get("evidence", {})
        result = cls(
            version=payload["version"],
            queries=queries,
            checksum=payload["checksum"],
            label_source=evidence.get("label_source", RelevanceLabelSource.UNVERIFIED),
            adjudication_complete=bool(evidence.get("adjudication_complete", False)),
            contains_synthetic_labels=bool(evidence.get("contains_synthetic_labels", False)),
            judgment_manifest_sha256=evidence.get("judgment_manifest_sha256"),
        )
        stored_evidence_checksum = payload.get("evidence_checksum")
        if (
            stored_evidence_checksum is not None
            and stored_evidence_checksum != result.evidence_checksum
        ):
            raise ValueError("frozen candidate evidence checksum does not match its contents")
        return result


@dataclass(frozen=True, slots=True)
class RetrievalEvaluation:
    dataset_version: str
    candidate_checksum: str
    query_count: int
    recall_at: Mapping[int, float]
    mean_reciprocal_rank: float
    ndcg_at_10: float
    metric_estimates: Mapping[str, MetricEstimate] = field(default_factory=dict)
    per_query: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    label_source: RelevanceLabelSource = RelevanceLabelSource.UNVERIFIED
    eligible_for_promotion: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_version": self.dataset_version,
            "candidate_checksum": self.candidate_checksum,
            "query_count": self.query_count,
            "recall_at": {str(key): value for key, value in self.recall_at.items()},
            "mean_reciprocal_rank": self.mean_reciprocal_rank,
            "ndcg_at_10": self.ndcg_at_10,
            "metric_estimates": {
                name: estimate.to_dict() for name, estimate in sorted(self.metric_estimates.items())
            },
            "per_query": {
                query_id: dict(sorted(metrics.items()))
                for query_id, metrics in sorted(self.per_query.items())
            },
            "label_source": self.label_source.value,
            "eligible_for_promotion": self.eligible_for_promotion,
        }


def _dcg(grades: Sequence[int], k: int) -> float:
    return float(
        sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(grades[:k], 1))
    )


def ndcg_at_k(
    relevance_labels: Mapping[str, int],
    ranked_product_ids: Sequence[str],
    *,
    k: int,
) -> float:
    """Calculate NDCG with the same exponential gains used by LambdaRank.

    The ranking must cover exactly the labeled candidate universe.  Keeping
    this check at the metric boundary prevents an apparently better score from
    being produced by silently dropping difficult candidates.
    """

    if k < 1:
        raise ValueError("NDCG cutoff must be positive")
    ranking = tuple(ranked_product_ids)
    if len(ranking) != len(set(ranking)):
        raise ValueError("NDCG ranking contains duplicate products")
    expected_ids = set(relevance_labels)
    actual_ids = set(ranking)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise ValueError(
            "NDCG ranking does not match the labeled candidate universe; "
            f"missing={missing}, extra={extra}"
        )
    grades = tuple(int(relevance_labels[product_id]) for product_id in ranking)
    ideal = tuple(sorted((int(grade) for grade in relevance_labels.values()), reverse=True))
    ideal_dcg = _dcg(ideal, k)
    return _dcg(grades, k) / ideal_dcg if ideal_dcg else 0.0


def evaluate_ranked_candidates(
    dataset: PinnedCandidateSet,
    ranked_product_ids: Mapping[str, Sequence[str]],
    *,
    recall_ks: Sequence[int] = (20, 50),
    confidence_level: float = 0.95,
    n_resamples: int = 1_000,
    seed: int = 20260722,
    bootstrap_groups: Mapping[str, Hashable] | None = None,
) -> RetrievalEvaluation:
    """Evaluate any retriever/ranker against exactly one frozen candidate set."""

    if not recall_ks or any(k < 1 for k in recall_ks):
        raise ValueError("recall cutoffs must be positive")
    recall_values: dict[int, list[float]] = {k: [] for k in recall_ks}
    reciprocal_ranks: list[float] = []
    ndcg_values: list[float] = []
    per_query: dict[str, dict[str, float]] = {}

    for query in dataset.queries:
        if query.query_id not in ranked_product_ids:
            raise ValueError(f"missing ranked results for query {query.query_id!r}")
        ranking = list(ranked_product_ids[query.query_id])
        if len(ranking) != len(set(ranking)):
            raise ValueError(f"ranking for {query.query_id!r} contains duplicate products")
        outside = set(ranking) - set(query.candidate_ids)
        if outside:
            raise ValueError(
                f"ranking for {query.query_id!r} changed the frozen candidate set: {outside}"
            )
        # Relevance judgments may use standard sparse-qrels semantics, where an
        # unlisted candidate is judged irrelevant.  Materialize those implicit
        # zero grades before calling the exact-universe metric so every frozen
        # candidate still contributes to the metric contract.
        complete_relevance_labels = {
            product_id: int(query.relevance_labels.get(product_id, 0))
            for product_id in query.candidate_ids
        }
        relevant = {
            item for item, grade in complete_relevance_labels.items() if grade > 0
        }
        for cutoff in recall_ks:
            recall = len(relevant.intersection(ranking[:cutoff])) / len(relevant)
            recall_values[cutoff].append(recall)
        first_relevant = next(
            (rank for rank, product_id in enumerate(ranking, 1) if product_id in relevant), None
        )
        reciprocal_ranks.append(0.0 if first_relevant is None else 1.0 / first_relevant)
        ndcg_values.append(ndcg_at_k(complete_relevance_labels, ranking, k=10))
        per_query[query.query_id] = {
            **{f"recall_at_{cutoff}": recall_values[cutoff][-1] for cutoff in recall_ks},
            "mean_reciprocal_rank": reciprocal_ranks[-1],
            "ndcg_at_10": ndcg_values[-1],
        }

    count = len(dataset.queries)
    if bootstrap_groups is not None:
        query_ids = {query.query_id for query in dataset.queries}
        if set(bootstrap_groups) != query_ids:
            raise ValueError("bootstrap_groups must cover exactly the evaluated queries")
        ordered_bootstrap_groups = [bootstrap_groups[query.query_id] for query in dataset.queries]
    else:
        ordered_bootstrap_groups = None
    metric_values: dict[str, list[float]] = {
        **{f"recall_at_{cutoff}": values for cutoff, values in recall_values.items()},
        "mean_reciprocal_rank": reciprocal_ranks,
        "ndcg_at_10": ndcg_values,
    }
    estimates: dict[str, MetricEstimate] = {}
    for offset, (name, values) in enumerate(sorted(metric_values.items())):
        value = sum(values) / count
        lower, upper = bootstrap_confidence_interval(
            values,
            groups=ordered_bootstrap_groups,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            seed=seed + offset,
        )
        estimates[name] = MetricEstimate(
            name=name,
            value=value,
            sample_count=count,
            ci_lower=min(lower, value),
            ci_upper=max(upper, value),
            confidence_level=confidence_level,
        )
    return RetrievalEvaluation(
        dataset_version=dataset.version,
        candidate_checksum=dataset.checksum,
        query_count=count,
        recall_at={cutoff: sum(values) / count for cutoff, values in recall_values.items()},
        mean_reciprocal_rank=sum(reciprocal_ranks) / count,
        ndcg_at_10=sum(ndcg_values) / count,
        metric_estimates=estimates,
        per_query=per_query,
        label_source=dataset.label_source,
        eligible_for_promotion=dataset.eligible_for_promotion,
    )
