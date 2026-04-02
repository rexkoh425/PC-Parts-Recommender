"""Frozen query splits and paired retrieval/ranking benchmark reports."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from pc_build_recommender.evaluation.manifest import json_sha256
from pc_build_recommender.evaluation.metrics import bootstrap_confidence_interval
from pc_build_recommender.evaluation.splits import deterministic_group_split

from .evaluation import (
    FrozenCandidateQuery,
    FrozenCandidateSet,
    RetrievalEvaluation,
    evaluate_ranked_candidates,
)

QUERY_SPLIT_SCHEMA_VERSION = "pc-build-recommender.frozen-query-group-split.v1"
COMPARISON_REPORT_SCHEMA_VERSION = "pc-build-recommender.ranking-comparison-report.v2"
ARTIFACT_BOUND_RANKING_SCHEMA_VERSION = "pc-build-recommender.artifact-bound-ranking.v1"
SILVER_DIAGNOSTIC_SCHEMA_VERSION = "pc-build-recommender.retrieval-silver-pilot.v1"


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _atomic_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialised = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(serialised)
            handle.write("\n")
            temporary_name = handle.name
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            temporary = Path(temporary_name)
            if temporary.exists():
                temporary.unlink()
    return path


@dataclass(frozen=True, slots=True)
class QueryGroupSplit:
    """Checksummed split assignments for query-intent leakage groups."""

    version: str
    dataset_checksum: str
    dataset_evidence_checksum: str
    label_source: str
    adjudication_complete: bool
    contains_synthetic_labels: bool
    judgment_manifest_sha256: str | None
    query_group_ids: Mapping[str, str]
    assignments: Mapping[str, str]
    weights: Mapping[str, float]
    seed: int
    checksum: str

    def __post_init__(self) -> None:
        if not self.version or not self.query_group_ids:
            raise ValueError("split version and query groups must not be empty")
        if self.label_source not in {"human", "silver", "synthetic", "unverified"}:
            raise ValueError("unsupported relevance label source in frozen split")
        if self.adjudication_complete and self.label_source != "human":
            raise ValueError("only human labels can be adjudication-complete")
        if self.judgment_manifest_sha256 is not None and (
            len(self.judgment_manifest_sha256) != 64
            or any(
                character not in "0123456789abcdef" for character in self.judgment_manifest_sha256
            )
        ):
            raise ValueError("judgment manifest must be a lowercase SHA-256 digest")
        if set(self.query_group_ids) != set(self.assignments):
            raise ValueError("split assignments must cover exactly the query-group mapping")
        if not self.weights or any(weight <= 0 for weight in self.weights.values()):
            raise ValueError("split weights must be positive")
        known_splits = set(self.weights)
        if set(self.assignments.values()) - known_splits:
            raise ValueError("split assignments contain an unknown split name")
        group_splits: dict[str, str] = {}
        for query_id, group_id in self.query_group_ids.items():
            split_name = self.assignments[query_id]
            previous = group_splits.setdefault(group_id, split_name)
            if previous != split_name:
                raise ValueError(
                    f"query group {group_id!r} leaks across {previous!r} and {split_name!r}"
                )
        if json_sha256(self.content_payload()) != self.checksum:
            raise ValueError("frozen query-group split checksum does not match its contents")

    def content_payload(self) -> dict[str, object]:
        return {
            "schema_version": QUERY_SPLIT_SCHEMA_VERSION,
            "version": self.version,
            "dataset_checksum": self.dataset_checksum,
            "dataset_evidence_checksum": self.dataset_evidence_checksum,
            "label_source": self.label_source,
            "adjudication_complete": self.adjudication_complete,
            "contains_synthetic_labels": self.contains_synthetic_labels,
            "judgment_manifest_sha256": self.judgment_manifest_sha256,
            "query_group_ids": dict(sorted(self.query_group_ids.items())),
            "assignments": dict(sorted(self.assignments.items())),
            "weights": dict(sorted(self.weights.items())),
            "seed": self.seed,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.content_payload(), "checksum": self.checksum}

    @classmethod
    def create(
        cls,
        dataset: FrozenCandidateSet,
        *,
        version: str,
        query_group_ids: Mapping[str, str] | None = None,
        weights: Mapping[str, float] | None = None,
        seed: int = 20260722,
    ) -> QueryGroupSplit:
        if query_group_ids is None:
            inferred = {query.query_id: query.query_group_id for query in dataset.queries}
            if any(group_id is None for group_id in inferred.values()):
                raise ValueError("every query needs a query_group_id before splitting")
            groups = {query_id: str(group_id) for query_id, group_id in inferred.items()}
        else:
            groups = {
                str(query_id): str(group_id) for query_id, group_id in query_group_ids.items()
            }
        expected_query_ids = {query.query_id for query in dataset.queries}
        if set(groups) != expected_query_ids:
            raise ValueError("query_group_ids must cover exactly the frozen dataset")
        if any(not group_id for group_id in groups.values()):
            raise ValueError("query group IDs must not be empty")
        group_split = deterministic_group_split(
            groups.values(),
            weights=weights,
            seed=seed,
        )
        assignments = {
            query_id: group_split.split_for(group_id) for query_id, group_id in groups.items()
        }
        split_weights = dict(group_split.weights)
        payload: dict[str, object] = {
            "schema_version": QUERY_SPLIT_SCHEMA_VERSION,
            "version": version,
            "dataset_checksum": dataset.checksum,
            "dataset_evidence_checksum": dataset.evidence_checksum,
            "label_source": dataset.label_source.value,
            "adjudication_complete": dataset.adjudication_complete,
            "contains_synthetic_labels": dataset.contains_synthetic_labels,
            "judgment_manifest_sha256": dataset.judgment_manifest_sha256,
            "query_group_ids": dict(sorted(groups.items())),
            "assignments": dict(sorted(assignments.items())),
            "weights": dict(sorted(split_weights.items())),
            "seed": seed,
        }
        return cls(
            version=version,
            dataset_checksum=dataset.checksum,
            dataset_evidence_checksum=dataset.evidence_checksum,
            label_source=dataset.label_source.value,
            adjudication_complete=dataset.adjudication_complete,
            contains_synthetic_labels=dataset.contains_synthetic_labels,
            judgment_manifest_sha256=dataset.judgment_manifest_sha256,
            query_group_ids=groups,
            assignments=assignments,
            weights=split_weights,
            seed=seed,
            checksum=json_sha256(payload),
        )

    def validate_dataset(self, dataset: FrozenCandidateSet) -> None:
        if dataset.checksum != self.dataset_checksum:
            raise ValueError("query split was created for a different frozen candidate set")
        if dataset.evidence_checksum != self.dataset_evidence_checksum:
            raise ValueError("query split relevance evidence has changed")
        if {query.query_id for query in dataset.queries} != set(self.assignments):
            raise ValueError("query split IDs do not match the frozen candidate set")

    def queries_for(
        self,
        dataset: FrozenCandidateSet,
        split_name: str,
    ) -> tuple[FrozenCandidateQuery, ...]:
        self.validate_dataset(dataset)
        if split_name not in self.weights:
            raise KeyError(f"unknown split name: {split_name!r}")
        queries = tuple(
            query for query in dataset.queries if self.assignments[query.query_id] == split_name
        )
        if not queries:
            raise ValueError(f"split {split_name!r} has no queries")
        return queries

    def subset(self, dataset: FrozenCandidateSet, split_name: str) -> FrozenCandidateSet:
        return FrozenCandidateSet.create(
            f"{dataset.version}:{split_name}",
            self.queries_for(dataset, split_name),
            label_source=dataset.label_source,
            adjudication_complete=dataset.adjudication_complete,
            contains_synthetic_labels=dataset.contains_synthetic_labels,
            judgment_manifest_sha256=dataset.judgment_manifest_sha256,
        )

    def save(self, path: str | Path) -> Path:
        return _atomic_json(Path(path), self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> QueryGroupSplit:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("query split root must be an object")
        if payload.get("schema_version") != QUERY_SPLIT_SCHEMA_VERSION:
            raise ValueError("unsupported frozen query-group split schema")
        return cls(
            version=str(payload["version"]),
            dataset_checksum=str(payload["dataset_checksum"]),
            dataset_evidence_checksum=str(payload["dataset_evidence_checksum"]),
            label_source=str(payload["label_source"]),
            adjudication_complete=bool(payload["adjudication_complete"]),
            contains_synthetic_labels=bool(payload["contains_synthetic_labels"]),
            judgment_manifest_sha256=(
                str(payload["judgment_manifest_sha256"])
                if payload.get("judgment_manifest_sha256") is not None
                else None
            ),
            query_group_ids={
                str(key): str(value) for key, value in payload["query_group_ids"].items()
            },
            assignments={str(key): str(value) for key, value in payload["assignments"].items()},
            weights={str(key): float(value) for key, value in payload["weights"].items()},
            seed=int(payload["seed"]),
            checksum=str(payload["checksum"]),
        )


@dataclass(frozen=True, slots=True)
class ArtifactBoundRankingEvidence:
    """Content binding between one evaluated ranking and exact model artifact bytes."""

    model_name: str
    ranker_version: str
    model_sha256: str
    metadata_sha256: str
    manifest_sha256: str
    ranker_metadata_payload: Mapping[str, Any]
    metadata_payload_sha256: str
    feature_version: str
    feature_names: tuple[str, ...]
    candidate_snapshot_sha256: str
    feature_snapshot_sha256: str
    score_snapshot_sha256: str
    ranking_sha256: str
    split_name: str | None
    split_checksum: str | None
    query_count: int
    row_count: int
    evidence_sha256: str

    def __post_init__(self) -> None:
        if not self.model_name or not self.ranker_version:
            raise ValueError("bound ranking model name and version must not be empty")
        for name, digest in (
            ("model_sha256", self.model_sha256),
            ("metadata_sha256", self.metadata_sha256),
            ("manifest_sha256", self.manifest_sha256),
            ("metadata_payload_sha256", self.metadata_payload_sha256),
            ("candidate_snapshot_sha256", self.candidate_snapshot_sha256),
            ("feature_snapshot_sha256", self.feature_snapshot_sha256),
            ("score_snapshot_sha256", self.score_snapshot_sha256),
            ("ranking_sha256", self.ranking_sha256),
        ):
            _require_sha256(digest, name)
        if self.split_checksum is not None:
            _require_sha256(self.split_checksum, "split_checksum")
        if (self.split_name is None) != (self.split_checksum is None):
            raise ValueError("bound ranking split name and checksum must be supplied together")
        if not self.feature_version or not self.feature_names:
            raise ValueError("bound ranking feature contract must not be empty")
        if len(self.feature_names) != len(set(self.feature_names)):
            raise ValueError("bound ranking feature names must be unique")
        if type(self.query_count) is not int or self.query_count < 1:
            raise ValueError("bound ranking query_count must be positive")
        if type(self.row_count) is not int or self.row_count < self.query_count:
            raise ValueError("bound ranking row_count must cover every query")
        metadata = dict(self.ranker_metadata_payload)
        if json_sha256(metadata) != self.metadata_payload_sha256:
            raise ValueError("bound ranking metadata payload hash does not match")
        if metadata.get("ranker_version") != self.ranker_version:
            raise ValueError("bound ranking version does not match metadata")
        if metadata.get("model_sha256") != self.model_sha256:
            raise ValueError("bound ranking model hash does not match metadata")
        if metadata.get("feature_version") != self.feature_version:
            raise ValueError("bound ranking feature version does not match metadata")
        if tuple(metadata.get("feature_names", ())) != self.feature_names:
            raise ValueError("bound ranking feature order does not match metadata")
        if self.evidence_sha256 and json_sha256(self.content_payload()) != self.evidence_sha256:
            raise ValueError("bound ranking evidence hash does not match its contents")

    def content_payload(self) -> dict[str, object]:
        return {
            "schema_version": ARTIFACT_BOUND_RANKING_SCHEMA_VERSION,
            "model_name": self.model_name,
            "ranker_version": self.ranker_version,
            "artifact_identity": {
                "model_sha256": self.model_sha256,
                "metadata_sha256": self.metadata_sha256,
                "manifest_sha256": self.manifest_sha256,
            },
            "ranker_metadata": dict(self.ranker_metadata_payload),
            "metadata_payload_sha256": self.metadata_payload_sha256,
            "feature_contract": {
                "version": self.feature_version,
                "names": list(self.feature_names),
                "snapshot_sha256": self.feature_snapshot_sha256,
            },
            "candidate_snapshot_sha256": self.candidate_snapshot_sha256,
            "score_snapshot_sha256": self.score_snapshot_sha256,
            "ranking_sha256": self.ranking_sha256,
            "split_name": self.split_name,
            "split_checksum": self.split_checksum,
            "query_count": self.query_count,
            "row_count": self.row_count,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.content_payload(), "evidence_sha256": self.evidence_sha256}

    @classmethod
    def create(
        cls,
        *,
        model_name: str,
        ranker_version: str,
        model_sha256: str,
        metadata_sha256: str,
        manifest_sha256: str,
        ranker_metadata_payload: Mapping[str, Any],
        feature_version: str,
        feature_names: Sequence[str],
        candidate_snapshot_sha256: str,
        feature_snapshot_sha256: str,
        score_snapshot_sha256: str,
        ranking_sha256: str,
        split_name: str | None,
        split_checksum: str | None,
        query_count: int,
        row_count: int,
    ) -> ArtifactBoundRankingEvidence:
        metadata = dict(ranker_metadata_payload)
        fields: dict[str, Any] = {
            "model_name": model_name,
            "ranker_version": ranker_version,
            "model_sha256": model_sha256,
            "metadata_sha256": metadata_sha256,
            "manifest_sha256": manifest_sha256,
            "ranker_metadata_payload": metadata,
            "metadata_payload_sha256": json_sha256(metadata),
            "feature_version": feature_version,
            "feature_names": tuple(feature_names),
            "candidate_snapshot_sha256": candidate_snapshot_sha256,
            "feature_snapshot_sha256": feature_snapshot_sha256,
            "score_snapshot_sha256": score_snapshot_sha256,
            "ranking_sha256": ranking_sha256,
            "split_name": split_name,
            "split_checksum": split_checksum,
            "query_count": query_count,
            "row_count": row_count,
        }
        provisional = cls(evidence_sha256="", **fields)
        return cls(evidence_sha256=json_sha256(provisional.content_payload()), **fields)


@dataclass(frozen=True, slots=True)
class PairedMetricComparison:
    metric_name: str
    baseline_model: str
    challenger_model: str
    baseline_value: float
    challenger_value: float
    absolute_delta: float
    relative_delta_percent: float | None
    sample_count: int
    ci_lower: float
    ci_upper: float
    confidence_level: float

    def to_dict(self) -> dict[str, object]:
        return {
            "metric_name": self.metric_name,
            "baseline_model": self.baseline_model,
            "challenger_model": self.challenger_model,
            "baseline_value": self.baseline_value,
            "challenger_value": self.challenger_value,
            "absolute_delta": self.absolute_delta,
            "relative_delta_percent": self.relative_delta_percent,
            "sample_count": self.sample_count,
            "paired_query_bootstrap_delta_ci": {
                "lower": self.ci_lower,
                "upper": self.ci_upper,
                "confidence_level": self.confidence_level,
            },
        }


@dataclass(frozen=True, slots=True)
class RankingComparisonReport:
    dataset_version: str
    candidate_checksum: str
    evidence_checksum: str
    judgment_manifest_sha256: str | None
    split_name: str | None
    split_checksum: str | None
    query_count: int
    query_group_count: int
    ranking_sha256: Mapping[str, str]
    artifact_bound_rankings: Mapping[str, ArtifactBoundRankingEvidence]
    model_evaluations: Mapping[str, RetrievalEvaluation]
    paired_comparisons: Mapping[str, Mapping[str, PairedMetricComparison]]
    label_source: str
    eligible_for_promotion: bool
    promotion_block_reasons: tuple[str, ...]
    confidence_level: float
    bootstrap_resamples: int
    bootstrap_seed: int
    report_sha256: str

    def content_payload(self) -> dict[str, object]:
        return {
            "schema_version": COMPARISON_REPORT_SCHEMA_VERSION,
            "dataset": {
                "version": self.dataset_version,
                "candidate_checksum": self.candidate_checksum,
                "evidence_checksum": self.evidence_checksum,
                "judgment_manifest_sha256": self.judgment_manifest_sha256,
                "split_name": self.split_name,
                "split_checksum": self.split_checksum,
                "query_count": self.query_count,
                "query_group_count": self.query_group_count,
                "label_source": self.label_source,
            },
            "eligible_for_promotion": self.eligible_for_promotion,
            "promotion_block_reasons": list(self.promotion_block_reasons),
            "evaluation_parameters": {
                "confidence_level": self.confidence_level,
                "bootstrap_resamples": self.bootstrap_resamples,
                "bootstrap_seed": self.bootstrap_seed,
                "paired_unit": ("query_group" if self.split_checksum is not None else "query"),
                "candidate_set_policy": "identical_complete_frozen_candidates",
            },
            "ranking_sha256": dict(sorted(self.ranking_sha256.items())),
            "artifact_bound_rankings": {
                name: evidence.to_dict()
                for name, evidence in sorted(self.artifact_bound_rankings.items())
            },
            "models": {
                name: evaluation.to_dict()
                for name, evaluation in sorted(self.model_evaluations.items())
            },
            "paired_comparisons": {
                pair_name: {
                    metric_name: comparison.to_dict()
                    for metric_name, comparison in sorted(metrics.items())
                }
                for pair_name, metrics in sorted(self.paired_comparisons.items())
            },
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.content_payload(), "report_sha256": self.report_sha256}

    def __post_init__(self) -> None:
        for model_name, evidence in self.artifact_bound_rankings.items():
            if model_name != evidence.model_name:
                raise ValueError("artifact-bound ranking key does not match its model name")
            if self.ranking_sha256.get(model_name) != evidence.ranking_sha256:
                raise ValueError("artifact-bound ranking hash does not match report output")
            if evidence.candidate_snapshot_sha256 != self.candidate_checksum:
                raise ValueError("artifact-bound candidate snapshot does not match report")
            if (
                evidence.split_name != self.split_name
                or evidence.split_checksum != self.split_checksum
            ):
                raise ValueError("artifact-bound ranking split does not match report")
            if evidence.query_count != self.query_count:
                raise ValueError("artifact-bound ranking query count does not match report")
        if self.report_sha256 and json_sha256(self.content_payload()) != self.report_sha256:
            raise ValueError("ranking comparison report hash does not match its contents")


@dataclass(frozen=True, slots=True)
class DiagnosticRankingArtifact:
    """Verified adapter for legacy/current silver diagnostic reports.

    This type deliberately has no conversion to ``RankingComparisonReport`` because
    silver query definitions lack frozen human adjudication and intent-group splits.
    """

    artifact_sha256: str
    dataset_version: str
    candidate_checksum: str
    query_count: int
    aggregate_model_metrics: Mapping[str, Mapping[str, object]]
    paired_baseline_comparisons: Mapping[str, object]
    reporting_block_reason: str
    eligible_for_promotion: bool = False


def load_diagnostic_ranking_artifact(path: str | Path) -> DiagnosticRankingArtifact:
    """Verify and consume a silver report while preserving its promotion block."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("diagnostic ranking artifact root must be an object")
    stored_hash = payload.get("artifact_sha256")
    unhashed = dict(payload)
    unhashed.pop("artifact_sha256", None)
    if stored_hash != json_sha256(unhashed):
        raise ValueError("diagnostic ranking artifact hash verification failed")
    if payload.get("schema_version") != SILVER_DIAGNOSTIC_SCHEMA_VERSION:
        raise ValueError("unsupported diagnostic ranking artifact schema")
    if payload.get("eligible_for_production_or_resume_metric_claims") is not False:
        raise ValueError("silver diagnostic artifact must be explicitly non-reportable")
    if payload.get("human_relevance_judgments") is not False:
        raise ValueError("silver diagnostic artifact cannot claim human judgments")
    if payload.get("training_performed") is not False:
        raise ValueError("silver diagnostic artifact cannot contain ranker training evidence")
    dataset = payload.get("dataset")
    models = payload.get("models")
    paired = payload.get("paired_baseline_comparisons")
    if not isinstance(dataset, Mapping) or not isinstance(models, Mapping):
        raise TypeError("diagnostic dataset and models must be objects")
    if not isinstance(paired, Mapping):
        raise TypeError("diagnostic paired comparisons must be an object")
    aggregate_metrics: dict[str, Mapping[str, object]] = {}
    for model_name, model_payload in models.items():
        if not isinstance(model_payload, Mapping):
            raise TypeError("diagnostic model payload must be an object")
        metrics = model_payload.get("metrics")
        if not isinstance(metrics, Mapping):
            raise TypeError("diagnostic model metrics must be an object")
        aggregate = metrics.get("aggregate")
        if not isinstance(aggregate, Mapping):
            raise TypeError("diagnostic aggregate metrics must be an object")
        aggregate_metrics[str(model_name)] = dict(aggregate)
    block_reason = payload.get("reporting_block_reason")
    if not isinstance(block_reason, str) or not block_reason:
        raise ValueError("diagnostic report must include a promotion block reason")
    return DiagnosticRankingArtifact(
        artifact_sha256=str(stored_hash),
        dataset_version=str(dataset["version"]),
        candidate_checksum=str(dataset["candidate_checksum"]),
        query_count=int(dataset["query_count"]),
        aggregate_model_metrics=aggregate_metrics,
        paired_baseline_comparisons=dict(paired),
        reporting_block_reason=block_reason,
        eligible_for_promotion=False,
    )


def _validate_complete_rankings(
    dataset: FrozenCandidateSet,
    rankings: Mapping[str, Sequence[str]],
    *,
    model_name: str,
) -> dict[str, list[str]]:
    expected_query_ids = {query.query_id for query in dataset.queries}
    missing = expected_query_ids - set(rankings)
    if missing:
        raise ValueError(f"model {model_name!r} is missing rankings for {sorted(missing)}")
    result: dict[str, list[str]] = {}
    for query in dataset.queries:
        ranking = list(rankings[query.query_id])
        if len(ranking) != len(set(ranking)):
            raise ValueError(f"model {model_name!r} returned duplicate product IDs")
        if set(ranking) != set(query.candidate_ids):
            raise ValueError(
                f"model {model_name!r} did not rank the identical frozen candidate set"
            )
        result[query.query_id] = ranking
    return result


def _paired_metrics(
    baseline_name: str,
    challenger_name: str,
    baseline: RetrievalEvaluation,
    challenger: RetrievalEvaluation,
    *,
    confidence_level: float,
    n_resamples: int,
    seed: int,
    query_group_ids: Mapping[str, str] | None,
) -> dict[str, PairedMetricComparison]:
    if set(baseline.per_query) != set(challenger.per_query):
        raise ValueError("paired models must contain identical query results")
    query_ids = sorted(baseline.per_query)
    metric_names = sorted(next(iter(baseline.per_query.values())))
    comparisons: dict[str, PairedMetricComparison] = {}
    for offset, metric_name in enumerate(metric_names):
        baseline_values = [baseline.per_query[query_id][metric_name] for query_id in query_ids]
        challenger_values = [challenger.per_query[query_id][metric_name] for query_id in query_ids]
        differences = [
            challenger_value - baseline_value
            for baseline_value, challenger_value in zip(
                baseline_values, challenger_values, strict=True
            )
        ]
        baseline_value = fmean(baseline_values)
        challenger_value = fmean(challenger_values)
        delta = challenger_value - baseline_value
        lower, upper = bootstrap_confidence_interval(
            differences,
            groups=(
                [query_group_ids[query_id] for query_id in query_ids]
                if query_group_ids is not None
                else None
            ),
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            seed=seed + offset,
        )
        comparisons[metric_name] = PairedMetricComparison(
            metric_name=metric_name,
            baseline_model=baseline_name,
            challenger_model=challenger_name,
            baseline_value=baseline_value,
            challenger_value=challenger_value,
            absolute_delta=delta,
            relative_delta_percent=(
                delta / baseline_value * 100.0 if baseline_value > 0.0 else None
            ),
            sample_count=len(query_ids),
            ci_lower=min(lower, delta),
            ci_upper=max(upper, delta),
            confidence_level=confidence_level,
        )
    return comparisons


def compare_ranked_models(
    dataset: FrozenCandidateSet,
    rankings_by_model: Mapping[str, Mapping[str, Sequence[str]]],
    *,
    artifact_bound_rankings: Mapping[str, ArtifactBoundRankingEvidence] | None = None,
    baseline_model: str = "bm25",
    reference_models: Sequence[str] | None = None,
    query_split: QueryGroupSplit | None = None,
    split_name: str | None = None,
    recall_ks: Sequence[int] = (20, 50),
    confidence_level: float = 0.95,
    n_resamples: int = 1_000,
    seed: int = 20260722,
) -> RankingComparisonReport:
    """Compare BM25, vector, RRF, and rankers on one frozen query/candidate set."""

    if len(rankings_by_model) < 2:
        raise ValueError("at least two ranked models are required for comparison")
    if baseline_model not in rankings_by_model:
        raise ValueError("baseline_model is not present in rankings_by_model")
    if query_split is not None:
        if split_name is None:
            raise ValueError("split_name is required with a frozen query split")
        evaluation_dataset = query_split.subset(dataset, split_name)
    else:
        if split_name is not None:
            raise ValueError("split_name requires a frozen query split")
        evaluation_dataset = dataset

    evaluations: dict[str, RetrievalEvaluation] = {}
    ranking_hashes: dict[str, str] = {}
    evaluation_group_ids = (
        {
            query.query_id: query_split.query_group_ids[query.query_id]
            for query in evaluation_dataset.queries
        }
        if query_split is not None
        else None
    )
    for offset, (model_name, rankings) in enumerate(sorted(rankings_by_model.items())):
        selected = {
            query.query_id: rankings[query.query_id]
            for query in evaluation_dataset.queries
            if query.query_id in rankings
        }
        complete = _validate_complete_rankings(
            evaluation_dataset,
            selected,
            model_name=model_name,
        )
        ranking_hashes[model_name] = json_sha256(complete)
        evaluations[model_name] = evaluate_ranked_candidates(
            evaluation_dataset,
            complete,
            recall_ks=recall_ks,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            seed=seed + offset * 100,
            bootstrap_groups=evaluation_group_ids,
        )

    bound_evidence = dict(artifact_bound_rankings or {})
    unknown_bound_models = set(bound_evidence) - set(evaluations)
    if unknown_bound_models:
        raise ValueError(
            f"artifact evidence refers to unknown models: {sorted(unknown_bound_models)}"
        )
    expected_row_count = sum(len(query.candidate_ids) for query in evaluation_dataset.queries)
    expected_split_checksum = query_split.checksum if query_split is not None else None
    for model_name, evidence in bound_evidence.items():
        if evidence.model_name != model_name:
            raise ValueError("artifact evidence key does not match its model name")
        if evidence.ranking_sha256 != ranking_hashes[model_name]:
            raise ValueError("artifact evidence does not match the evaluated ranking output")
        if evidence.candidate_snapshot_sha256 != evaluation_dataset.checksum:
            raise ValueError("artifact evidence does not match the evaluated candidate snapshot")
        if evidence.feature_snapshot_sha256 == evidence.candidate_snapshot_sha256:
            raise ValueError("feature and candidate snapshots must be independently hashed")
        if evidence.split_name != split_name or evidence.split_checksum != expected_split_checksum:
            raise ValueError("artifact evidence does not match the evaluated query split")
        if evidence.query_count != len(evaluation_dataset.queries):
            raise ValueError("artifact evidence query count does not match the evaluation")
        if evidence.row_count != expected_row_count:
            raise ValueError("artifact evidence row count does not match the evaluation")

    references = list(reference_models or (baseline_model,))
    if reference_models is None:
        references.extend(
            name for name in evaluations if "rrf" in name.casefold() and name not in references
        )
    unknown_references = set(references) - set(evaluations)
    if unknown_references:
        raise ValueError(f"unknown reference models: {sorted(unknown_references)}")
    paired: dict[str, Mapping[str, PairedMetricComparison]] = {}
    pair_offset = 0
    for reference in dict.fromkeys(references):
        for challenger in sorted(evaluations):
            if challenger == reference:
                continue
            pair_name = f"{challenger}_minus_{reference}"
            paired[pair_name] = _paired_metrics(
                reference,
                challenger,
                evaluations[reference],
                evaluations[challenger],
                confidence_level=confidence_level,
                n_resamples=n_resamples,
                seed=seed + 10_000 + pair_offset * 100,
                query_group_ids=evaluation_group_ids,
            )
            pair_offset += 1

    block_reasons = list(dataset.promotion_block_reasons)
    if query_split is None or split_name != "test":
        block_reasons.append("evaluation is not tied to the frozen test query-group split")
    eligible = not block_reasons
    report_fields: dict[str, Any] = {
        "dataset_version": evaluation_dataset.version,
        "candidate_checksum": evaluation_dataset.checksum,
        "evidence_checksum": evaluation_dataset.evidence_checksum,
        "judgment_manifest_sha256": evaluation_dataset.judgment_manifest_sha256,
        "split_name": split_name,
        "split_checksum": query_split.checksum if query_split is not None else None,
        "query_count": len(evaluation_dataset.queries),
        "query_group_count": (
            len(set(evaluation_group_ids.values()))
            if evaluation_group_ids is not None
            else len(evaluation_dataset.queries)
        ),
        "ranking_sha256": ranking_hashes,
        "artifact_bound_rankings": bound_evidence,
        "model_evaluations": evaluations,
        "paired_comparisons": paired,
        "label_source": dataset.label_source.value,
        "eligible_for_promotion": eligible,
        "promotion_block_reasons": tuple(block_reasons),
        "confidence_level": confidence_level,
        "bootstrap_resamples": n_resamples,
        "bootstrap_seed": seed,
    }
    provisional = RankingComparisonReport(report_sha256="", **report_fields)
    # Build the hash from the semantic payload without accepting the provisional hash.
    report_hash = json_sha256(provisional.content_payload())
    return RankingComparisonReport(report_sha256=report_hash, **report_fields)


def write_ranking_comparison_report(
    report: RankingComparisonReport,
    path: str | Path,
) -> Path:
    return _atomic_json(Path(path), report.to_dict())


def load_ranking_comparison_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("ranking comparison report root must be an object")
    stored_hash = payload.get("report_sha256")
    unhashed = dict(payload)
    unhashed.pop("report_sha256", None)
    if stored_hash != json_sha256(unhashed):
        raise ValueError("ranking comparison report hash verification failed")
    return payload
