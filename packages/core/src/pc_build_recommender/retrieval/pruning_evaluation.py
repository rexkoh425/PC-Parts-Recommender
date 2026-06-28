"""Claim-grade evaluation for sequential candidate-pruning stages.

The evaluator deliberately consumes frozen candidate/qrel contracts rather than
application pipeline objects.  This keeps measurement independent from serving
code and prevents a diagnostic candidate pool from being presented as a
full-corpus pruning result.
"""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
from collections import defaultdict
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from statistics import fmean
from typing import Any

from pc_build_recommender.evaluation.contracts import MetricEstimate
from pc_build_recommender.evaluation.manifest import sha256_json
from pc_build_recommender.evaluation.metrics import bootstrap_confidence_interval

from .benchmark import QueryGroupSplit
from .evaluation import FrozenCandidateQuery, FrozenCandidateSet, RelevanceLabelSource

PRUNING_STAGE_SCHEMA_VERSION = "pc-build-recommender.frozen-pruning-stage.v1"
PRUNING_TRACE_SCHEMA_VERSION = "pc-build-recommender.frozen-pruning-trace.v1"
PRUNING_POPULATION_SCHEMA_VERSION = "pc-build-recommender.pruning-population.v1"
PRUNING_PROVENANCE_SCHEMA_VERSION = "pc-build-recommender.pruning-provenance.v1"
PRUNING_REPORT_SCHEMA_VERSION = "pc-build-recommender.pruning-evaluation-report.v1"
PRUNING_CLAIM_DECISION_SCHEMA_VERSION = "pc-build-recommender.pruning-claim-decision.v1"


class CandidatePopulationScope(StrEnum):
    """Population represented by each query's frozen candidate IDs."""

    FROZEN_RETRIEVAL_POOL = "frozen_retrieval_pool"
    FULL_ELIGIBLE_CORPUS = "full_eligible_corpus"


class PruningStageKind(StrEnum):
    """Typed stage semantics prevent rank truncation from masquerading as filtering."""

    STRUCTURED_REQUIREMENTS = "structured_requirements"
    COMPATIBILITY = "compatibility"
    RETRIEVAL = "retrieval"
    RANKING_TRUNCATION = "ranking_truncation"
    OTHER_DIAGNOSTIC = "other_diagnostic"


class PruningPromotionError(ValueError):
    """Raised when a diagnostic pruning report is used as promotion evidence."""


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _require_sha256(value: str, field_name: str) -> None:
    if not _is_sha256(value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


def _validate_utc_timestamp(value: str) -> None:
    if not value:
        raise ValueError("evaluated_at_utc must not be empty")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("evaluated_at_utc must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError("evaluated_at_utc must have a UTC offset")


def _atomic_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialised = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
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
class FrozenPruningStage:
    """Content-addressed retained candidate sets emitted by one filter stage."""

    name: str
    kind: PruningStageKind
    version: str
    retained_candidate_ids: Mapping[str, tuple[str, ...]]
    removal_reasons: Mapping[str, Mapping[str, tuple[str, ...]]]
    checksum: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("pruning stage name must not be empty")
        object.__setattr__(self, "kind", PruningStageKind(self.kind))
        if not self.version:
            raise ValueError("pruning stage version must not be empty")
        if not self.retained_candidate_ids:
            raise ValueError("pruning stage must contain query outputs")
        normalised: dict[str, tuple[str, ...]] = {}
        for query_id, candidate_ids in self.retained_candidate_ids.items():
            if not isinstance(query_id, str) or not query_id:
                raise ValueError("pruning stage query IDs must not be empty")
            if isinstance(candidate_ids, (str, bytes)):
                raise TypeError("pruning stage candidate collections must be sequences of IDs")
            candidates = tuple(candidate_ids)
            if any(
                not isinstance(candidate_id, str) or not candidate_id for candidate_id in candidates
            ):
                raise ValueError("pruning stage candidate IDs must not be empty")
            if len(candidates) != len(set(candidates)):
                raise ValueError(f"pruning stage {self.name!r} contains duplicate candidates")
            normalised[str(query_id)] = tuple(sorted(candidates))
        object.__setattr__(self, "retained_candidate_ids", normalised)
        if set(self.removal_reasons) != set(normalised):
            raise ValueError("removal reasons must cover exactly the pruning stage queries")
        normalised_reasons: dict[str, dict[str, tuple[str, ...]]] = {}
        for query_id, candidate_reasons in self.removal_reasons.items():
            query_reasons: dict[str, tuple[str, ...]] = {}
            for candidate_id, reason_codes in candidate_reasons.items():
                if not isinstance(candidate_id, str) or not candidate_id:
                    raise ValueError("removed candidate IDs must be non-empty strings")
                if isinstance(reason_codes, (str, bytes)):
                    raise TypeError("removal reason codes must be sequences of strings")
                reasons = tuple(reason_codes)
                if not reasons or any(
                    not isinstance(reason, str) or not reason for reason in reasons
                ):
                    raise ValueError("every removed candidate needs non-empty reason codes")
                if len(reasons) != len(set(reasons)):
                    raise ValueError("removal reason codes must be unique per candidate")
                query_reasons[candidate_id] = tuple(sorted(reasons))
            normalised_reasons[query_id] = dict(sorted(query_reasons.items()))
        object.__setattr__(self, "removal_reasons", normalised_reasons)
        _require_sha256(self.checksum, "pruning stage checksum")
        if sha256_json(self.content_payload()) != self.checksum:
            raise ValueError("frozen pruning stage checksum does not match its contents")

    def content_payload(self) -> dict[str, object]:
        return {
            "schema_version": PRUNING_STAGE_SCHEMA_VERSION,
            "name": self.name,
            "kind": self.kind.value,
            "version": self.version,
            "retained_candidate_ids": {
                query_id: list(candidate_ids)
                for query_id, candidate_ids in sorted(self.retained_candidate_ids.items())
            },
            "removal_reasons": {
                query_id: {
                    candidate_id: list(reasons)
                    for candidate_id, reasons in sorted(candidate_reasons.items())
                }
                for query_id, candidate_reasons in sorted(self.removal_reasons.items())
            },
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.content_payload(), "checksum": self.checksum}

    @classmethod
    def create(
        cls,
        name: str,
        retained_candidate_ids: Mapping[str, Sequence[str]],
        *,
        kind: PruningStageKind,
        version: str,
        removal_reasons: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
    ) -> FrozenPruningStage:
        normalised: dict[str, tuple[str, ...]] = {}
        for query_id, candidate_ids in retained_candidate_ids.items():
            if not isinstance(query_id, str):
                raise TypeError("pruning stage query IDs must be strings")
            if isinstance(candidate_ids, (str, bytes)):
                raise TypeError("pruning stage candidate collections must be sequences of IDs")
            if any(not isinstance(candidate_id, str) for candidate_id in candidate_ids):
                raise TypeError("pruning stage candidate IDs must be strings")
            normalised[query_id] = tuple(sorted(candidate_ids))
        normalised_reasons: dict[str, dict[str, tuple[str, ...]]] = {}
        for query_id, candidate_reasons in (removal_reasons or {}).items():
            if not isinstance(query_id, str) or not isinstance(candidate_reasons, Mapping):
                raise TypeError("removal reasons must map query IDs to candidate mappings")
            normalised_reasons[query_id] = {}
            for candidate_id, reasons in candidate_reasons.items():
                if not isinstance(candidate_id, str):
                    raise TypeError("removed candidate IDs must be strings")
                if isinstance(reasons, (str, bytes)):
                    raise TypeError("removal reason codes must be sequences of strings")
                reason_values = tuple(reasons)
                if any(not isinstance(reason, str) for reason in reason_values):
                    raise TypeError("removal reason codes must be strings")
                normalised_reasons[query_id][candidate_id] = tuple(sorted(reason_values))
        for query_id in normalised:
            normalised_reasons.setdefault(query_id, {})
        payload: dict[str, object] = {
            "schema_version": PRUNING_STAGE_SCHEMA_VERSION,
            "name": name,
            "kind": PruningStageKind(kind).value,
            "version": version,
            "retained_candidate_ids": {
                query_id: list(candidate_ids)
                for query_id, candidate_ids in sorted(normalised.items())
            },
            "removal_reasons": {
                query_id: {
                    candidate_id: list(reasons)
                    for candidate_id, reasons in sorted(candidate_reasons.items())
                }
                for query_id, candidate_reasons in sorted(normalised_reasons.items())
            },
        }
        return cls(
            name=name,
            kind=PruningStageKind(kind),
            version=version,
            retained_candidate_ids=normalised,
            removal_reasons=normalised_reasons,
            checksum=sha256_json(payload),
        )


@dataclass(frozen=True, slots=True)
class CandidatePopulationDeclaration:
    """Frozen declaration of what the candidate IDs represent.

    ``FULL_ELIGIBLE_CORPUS`` is only valid when the caller explicitly attests that
    the candidate set exhausts the eligible catalog population for every query.
    Retrieval pools remain useful diagnostics but cannot support a corpus-level
    pruning claim.
    """

    scope: CandidatePopulationScope
    dataset_checksum: str
    dataset_evidence_checksum: str
    catalog_manifest_sha256: str
    construction_method: str
    complete_eligible_corpus: bool
    catalog_membership_sha256: str | None
    catalog_membership_verified: bool
    candidate_counts_by_query: Mapping[str, int]
    checksum: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", CandidatePopulationScope(self.scope))
        _require_sha256(self.dataset_checksum, "population dataset checksum")
        _require_sha256(self.dataset_evidence_checksum, "population evidence checksum")
        _require_sha256(self.catalog_manifest_sha256, "catalog manifest hash")
        if not self.construction_method:
            raise ValueError("population construction_method must not be empty")
        if not self.candidate_counts_by_query:
            raise ValueError("population candidate counts must not be empty")
        counts: dict[str, int] = {}
        for query_id, count in self.candidate_counts_by_query.items():
            if not isinstance(query_id, str):
                raise TypeError("population query IDs must be strings")
            if isinstance(count, bool) or not isinstance(count, int):
                raise TypeError("population candidate counts must be integers")
            counts[query_id] = count
        if any(not query_id for query_id in counts):
            raise ValueError("population query IDs must not be empty")
        if any(count < 1 for count in counts.values()):
            raise ValueError("population candidate counts must be positive")
        object.__setattr__(self, "candidate_counts_by_query", counts)
        if self.scope is CandidatePopulationScope.FULL_ELIGIBLE_CORPUS:
            if not self.complete_eligible_corpus:
                raise ValueError("full eligible corpus scope requires completeness attestation")
            if not self.catalog_membership_verified or self.catalog_membership_sha256 is None:
                raise ValueError(
                    "full eligible corpus scope requires verified catalog candidate membership"
                )
        elif self.complete_eligible_corpus:
            raise ValueError("a retrieval pool cannot attest full eligible-corpus completeness")
        elif self.catalog_membership_verified or self.catalog_membership_sha256 is not None:
            raise ValueError("retrieval-pool scope cannot claim verified catalog membership")
        if self.catalog_membership_sha256 is not None:
            _require_sha256(self.catalog_membership_sha256, "catalog membership hash")
        _require_sha256(self.checksum, "population declaration checksum")
        if sha256_json(self.content_payload()) != self.checksum:
            raise ValueError("population declaration checksum does not match its contents")

    @property
    def corpus_qualified(self) -> bool:
        return (
            self.scope is CandidatePopulationScope.FULL_ELIGIBLE_CORPUS
            and self.complete_eligible_corpus
            and self.catalog_membership_verified
            and self.catalog_membership_sha256 is not None
        )

    def content_payload(self) -> dict[str, object]:
        return {
            "schema_version": PRUNING_POPULATION_SCHEMA_VERSION,
            "scope": self.scope.value,
            "dataset_checksum": self.dataset_checksum,
            "dataset_evidence_checksum": self.dataset_evidence_checksum,
            "catalog_manifest_sha256": self.catalog_manifest_sha256,
            "construction_method": self.construction_method,
            "complete_eligible_corpus": self.complete_eligible_corpus,
            "catalog_membership_sha256": self.catalog_membership_sha256,
            "catalog_membership_verified": self.catalog_membership_verified,
            "candidate_counts_by_query": dict(sorted(self.candidate_counts_by_query.items())),
            "corpus_qualified": self.corpus_qualified,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.content_payload(), "checksum": self.checksum}

    @classmethod
    def create(
        cls,
        dataset: FrozenCandidateSet,
        *,
        scope: CandidatePopulationScope,
        catalog_manifest_sha256: str,
        construction_method: str,
        complete_eligible_corpus: bool,
        catalog_candidate_ids_by_query: Mapping[str, Sequence[str]] | None = None,
    ) -> CandidatePopulationDeclaration:
        counts = {query.query_id: len(query.candidate_ids) for query in dataset.queries}
        expected_membership = {
            query.query_id: sorted(query.candidate_ids) for query in dataset.queries
        }
        membership_hash: str | None = None
        membership_verified = False
        if CandidatePopulationScope(scope) is CandidatePopulationScope.FULL_ELIGIBLE_CORPUS:
            if catalog_candidate_ids_by_query is None:
                raise ValueError(
                    "full eligible corpus scope requires catalog_candidate_ids_by_query"
                )
            supplied_membership: dict[str, list[str]] = {}
            for query_id, candidate_ids in catalog_candidate_ids_by_query.items():
                if not isinstance(query_id, str) or not query_id:
                    raise TypeError("catalog membership query IDs must be non-empty strings")
                if isinstance(candidate_ids, (str, bytes)):
                    raise TypeError("catalog membership candidates must be sequences of IDs")
                candidates = tuple(candidate_ids)
                if any(
                    not isinstance(candidate_id, str) or not candidate_id
                    for candidate_id in candidates
                ):
                    raise TypeError("catalog membership candidate IDs must be non-empty strings")
                if len(candidates) != len(set(candidates)):
                    raise ValueError("catalog membership candidate IDs must be unique")
                supplied_membership[query_id] = sorted(candidates)
            if supplied_membership != expected_membership:
                raise ValueError(
                    "frozen candidates do not exactly match catalog-eligible query membership"
                )
            membership_hash = sha256_json(supplied_membership)
            membership_verified = True
        elif catalog_candidate_ids_by_query is not None:
            raise ValueError("retrieval-pool scope cannot accept full-corpus membership evidence")
        payload: dict[str, object] = {
            "schema_version": PRUNING_POPULATION_SCHEMA_VERSION,
            "scope": CandidatePopulationScope(scope).value,
            "dataset_checksum": dataset.checksum,
            "dataset_evidence_checksum": dataset.evidence_checksum,
            "catalog_manifest_sha256": catalog_manifest_sha256,
            "construction_method": construction_method,
            "complete_eligible_corpus": complete_eligible_corpus,
            "catalog_membership_sha256": membership_hash,
            "catalog_membership_verified": membership_verified,
            "candidate_counts_by_query": dict(sorted(counts.items())),
            "corpus_qualified": (
                CandidatePopulationScope(scope) is CandidatePopulationScope.FULL_ELIGIBLE_CORPUS
                and complete_eligible_corpus
                and membership_verified
                and membership_hash is not None
            ),
        }
        return cls(
            scope=CandidatePopulationScope(scope),
            dataset_checksum=dataset.checksum,
            dataset_evidence_checksum=dataset.evidence_checksum,
            catalog_manifest_sha256=catalog_manifest_sha256,
            construction_method=construction_method,
            complete_eligible_corpus=complete_eligible_corpus,
            catalog_membership_sha256=membership_hash,
            catalog_membership_verified=membership_verified,
            candidate_counts_by_query=counts,
            checksum=sha256_json(payload),
        )

    def validate_dataset(self, dataset: FrozenCandidateSet) -> None:
        if self.dataset_checksum != dataset.checksum:
            raise ValueError("population declaration targets a different frozen candidate set")
        if self.dataset_evidence_checksum != dataset.evidence_checksum:
            raise ValueError("population declaration targets different relevance evidence")
        expected = {query.query_id: len(query.candidate_ids) for query in dataset.queries}
        if dict(self.candidate_counts_by_query) != expected:
            raise ValueError("population counts do not match the frozen candidate set")
        if self.catalog_membership_verified:
            expected_membership_hash = sha256_json(
                {query.query_id: sorted(query.candidate_ids) for query in dataset.queries}
            )
            if self.catalog_membership_sha256 != expected_membership_hash:
                raise ValueError("catalog membership does not match frozen candidate IDs")


@dataclass(frozen=True, slots=True)
class PruningEvaluationProvenance:
    """Execution and configuration provenance committed into a pruning report."""

    run_id: str
    evaluated_at_utc: str
    pipeline_version: str
    compatibility_rule_version: str
    data_version: str
    code_revision: str
    catalog_manifest_sha256: str
    filter_configuration_sha256: str
    metadata: Mapping[str, object]
    checksum: str

    def __post_init__(self) -> None:
        required_versions = (
            self.run_id,
            self.pipeline_version,
            self.compatibility_rule_version,
            self.data_version,
            self.code_revision,
        )
        if any(not value for value in required_versions):
            raise ValueError("run and version provenance fields must not be empty")
        _validate_utc_timestamp(self.evaluated_at_utc)
        _require_sha256(self.catalog_manifest_sha256, "catalog manifest hash")
        _require_sha256(self.filter_configuration_sha256, "filter configuration hash")
        metadata = dict(self.metadata)
        # Canonical serialisation both validates the values and freezes their hash.
        sha256_json(metadata)
        object.__setattr__(self, "metadata", metadata)
        _require_sha256(self.checksum, "provenance checksum")
        if sha256_json(self.content_payload()) != self.checksum:
            raise ValueError("pruning provenance checksum does not match its contents")

    def content_payload(self) -> dict[str, object]:
        return {
            "schema_version": PRUNING_PROVENANCE_SCHEMA_VERSION,
            "run_id": self.run_id,
            "evaluated_at_utc": self.evaluated_at_utc,
            "pipeline_version": self.pipeline_version,
            "compatibility_rule_version": self.compatibility_rule_version,
            "data_version": self.data_version,
            "code_revision": self.code_revision,
            "catalog_manifest_sha256": self.catalog_manifest_sha256,
            "filter_configuration_sha256": self.filter_configuration_sha256,
            "metadata": dict(self.metadata),
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.content_payload(), "checksum": self.checksum}

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        evaluated_at_utc: str,
        pipeline_version: str,
        compatibility_rule_version: str,
        data_version: str,
        code_revision: str,
        catalog_manifest_sha256: str,
        filter_configuration_sha256: str,
        metadata: Mapping[str, object] | None = None,
    ) -> PruningEvaluationProvenance:
        values = dict(metadata or {})
        payload: dict[str, object] = {
            "schema_version": PRUNING_PROVENANCE_SCHEMA_VERSION,
            "run_id": run_id,
            "evaluated_at_utc": evaluated_at_utc,
            "pipeline_version": pipeline_version,
            "compatibility_rule_version": compatibility_rule_version,
            "data_version": data_version,
            "code_revision": code_revision,
            "catalog_manifest_sha256": catalog_manifest_sha256,
            "filter_configuration_sha256": filter_configuration_sha256,
            "metadata": values,
        }
        return cls(
            run_id=run_id,
            evaluated_at_utc=evaluated_at_utc,
            pipeline_version=pipeline_version,
            compatibility_rule_version=compatibility_rule_version,
            data_version=data_version,
            code_revision=code_revision,
            catalog_manifest_sha256=catalog_manifest_sha256,
            filter_configuration_sha256=filter_configuration_sha256,
            metadata=values,
            checksum=sha256_json(payload),
        )


@dataclass(frozen=True, slots=True)
class PruningStageEvaluation:
    """Counts and ratio estimates for one sequential pruning stage."""

    name: str
    kind: PruningStageKind
    version: str
    stage_snapshot_sha256: str
    candidate_count_before: int
    candidate_count_after: int
    relevant_count_before: int
    relevant_count_after: int
    metrics: Mapping[str, MetricEstimate]
    per_query: Mapping[str, PruningQueryStageCounts]

    def __post_init__(self) -> None:
        if not self.name or not self.version:
            raise ValueError("evaluated pruning stage name and version must not be empty")
        object.__setattr__(self, "kind", PruningStageKind(self.kind))
        _require_sha256(self.stage_snapshot_sha256, "stage snapshot hash")
        if not self.per_query:
            raise ValueError("evaluated pruning stage must contain per-query counts")
        if sum(item.candidate_before for item in self.per_query.values()) != (
            self.candidate_count_before
        ):
            raise ValueError("candidate before-count does not match per-query counts")
        if sum(item.candidate_after for item in self.per_query.values()) != (
            self.candidate_count_after
        ):
            raise ValueError("candidate after-count does not match per-query counts")
        if sum(item.relevant_before for item in self.per_query.values()) != (
            self.relevant_count_before
        ):
            raise ValueError("relevant before-count does not match per-query counts")
        if sum(item.relevant_after for item in self.per_query.values()) != (
            self.relevant_count_after
        ):
            raise ValueError("relevant after-count does not match per-query counts")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "version": self.version,
            "stage_snapshot_sha256": self.stage_snapshot_sha256,
            "counts": {
                "candidate_query_pairs": {
                    "before": self.candidate_count_before,
                    "after": self.candidate_count_after,
                    "pruned": self.candidate_count_before - self.candidate_count_after,
                },
                "judged_relevant_query_pairs": {
                    "before": self.relevant_count_before,
                    "after": self.relevant_count_after,
                    "lost": self.relevant_count_before - self.relevant_count_after,
                },
            },
            "metrics": {
                name: estimate.to_dict() for name, estimate in sorted(self.metrics.items())
            },
            "per_query": {
                query_id: counts.to_dict() for query_id, counts in sorted(self.per_query.items())
            },
        }


@dataclass(frozen=True, slots=True)
class PruningEvaluationReport:
    """Content-addressed, scope-qualified pruning evaluation report."""

    dataset_version: str
    candidate_checksum: str
    evidence_checksum: str
    judgment_manifest_sha256: str | None
    label_source: str
    adjudication_complete: bool
    contains_synthetic_labels: bool
    qrels_complete: bool
    split_name: str | None
    split_checksum: str | None
    evaluated_candidate_checksum: str
    query_count: int
    query_group_count: int
    population: CandidatePopulationDeclaration
    provenance: PruningEvaluationProvenance
    pruning_trace_sha256: str
    stages: tuple[PruningStageEvaluation, ...]
    evidence_eligible: bool
    evidence_block_reasons: tuple[str, ...]
    eligible_for_promotion: bool
    promotion_block_reasons: tuple[str, ...]
    confidence_level: float
    bootstrap_resamples: int
    bootstrap_seed: int
    report_sha256: str

    @property
    def corpus_claim_qualified(self) -> bool:
        return self.population.corpus_qualified

    @property
    def final_stage(self) -> PruningStageEvaluation:
        return self.stages[-1]

    def require_promotable(self) -> None:
        if sha256_json(self.content_payload()) != self.report_sha256:
            raise PruningPromotionError("pruning report changed after hashing")
        if not self.eligible_for_promotion:
            reasons = "; ".join(self.promotion_block_reasons)
            raise PruningPromotionError(f"pruning report is not promotable: {reasons}")

    def content_payload(self) -> dict[str, object]:
        return {
            "schema_version": PRUNING_REPORT_SCHEMA_VERSION,
            "dataset": {
                "version": self.dataset_version,
                "candidate_checksum": self.candidate_checksum,
                "evidence_checksum": self.evidence_checksum,
                "judgment_manifest_sha256": self.judgment_manifest_sha256,
                "label_source": self.label_source,
                "adjudication_complete": self.adjudication_complete,
                "contains_synthetic_labels": self.contains_synthetic_labels,
                "qrels_complete": self.qrels_complete,
                "split_name": self.split_name,
                "split_checksum": self.split_checksum,
                "evaluated_candidate_checksum": self.evaluated_candidate_checksum,
                "query_count": self.query_count,
                "query_group_count": self.query_group_count,
            },
            "population": self.population.to_dict(),
            "provenance": self.provenance.to_dict(),
            "claim_qualification": {
                "aggregation": "pooled_query_candidate_pairs",
                "population_scope": self.population.scope.value,
                "corpus_claim_qualified": self.corpus_claim_qualified,
                "relevance_basis": f"{self.label_source}_qrels",
                "relevance_threshold": "grade > 0",
                "evidence_eligible": self.evidence_eligible,
                "evidence_block_reasons": list(self.evidence_block_reasons),
                "judged_pool_claim_eligible": self.evidence_eligible,
                "corpus_claim_eligible": self.eligible_for_promotion,
                "eligible_for_promotion": self.eligible_for_promotion,
                "promotion_block_reasons": list(self.promotion_block_reasons),
            },
            "evaluation_parameters": {
                "confidence_level": self.confidence_level,
                "bootstrap_resamples": self.bootstrap_resamples,
                "bootstrap_seed": self.bootstrap_seed,
                "bootstrap_unit": ("query_group" if self.split_checksum is not None else "query"),
                "stage_policy": "sequential_monotonic_subsets",
            },
            "pruning_trace_sha256": self.pruning_trace_sha256,
            "stages": [stage.to_dict() for stage in self.stages],
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.content_payload(), "report_sha256": self.report_sha256}

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("pruning report must contain at least one stage")
        _require_sha256(self.candidate_checksum, "candidate checksum")
        _require_sha256(self.evidence_checksum, "evidence checksum")
        _require_sha256(self.evaluated_candidate_checksum, "evaluated candidate checksum")
        RelevanceLabelSource(self.label_source)
        if self.evidence_eligible != (not self.evidence_block_reasons):
            raise ValueError("evidence eligibility and block reasons are inconsistent")
        expected_promotion = self.evidence_eligible and self.population.corpus_qualified
        if self.eligible_for_promotion != expected_promotion:
            raise ValueError("promotion eligibility is inconsistent with evidence and scope")
        expected_queries = set(self.stages[0].per_query)
        if len(expected_queries) != self.query_count:
            raise ValueError("report query count does not match per-query stage counts")
        previous_after: dict[str, int] | None = None
        for stage in self.stages:
            if set(stage.per_query) != expected_queries:
                raise ValueError("every evaluated stage must cover the same queries")
            current_before = {
                query_id: counts.candidate_before for query_id, counts in stage.per_query.items()
            }
            if previous_after is not None and current_before != previous_after:
                raise ValueError("evaluated stages are not sequential")
            previous_after = {
                query_id: counts.candidate_after for query_id, counts in stage.per_query.items()
            }
        _require_sha256(self.pruning_trace_sha256, "pruning trace hash")
        if self.report_sha256:
            _require_sha256(self.report_sha256, "pruning report hash")
        if self.report_sha256 and sha256_json(self.content_payload()) != self.report_sha256:
            raise ValueError("pruning report hash does not match its contents")


@dataclass(frozen=True, slots=True)
class PruningClaimPolicy:
    """Predeclared gates for the 90%-pruning-at-97%-recall claim."""

    minimum_test_query_groups: int = 150
    minimum_fully_judged_candidate_pairs: int = 2_000
    minimum_confidence_level: float = 0.95
    minimum_bootstrap_resamples: int = 1_000
    minimum_pooled_pruning_fraction: float = 0.90
    minimum_pooled_retained_relevant_recall: float = 0.97
    minimum_pruning_ci_lower: float = 0.90
    minimum_recall_ci_lower: float = 0.97
    required_stage_kinds: tuple[PruningStageKind, ...] = (
        PruningStageKind.STRUCTURED_REQUIREMENTS,
        PruningStageKind.COMPATIBILITY,
    )
    require_corpus_qualification: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.minimum_test_query_groups, bool)
            or not isinstance(self.minimum_test_query_groups, int)
            or self.minimum_test_query_groups < 1
        ):
            raise ValueError("minimum_test_query_groups must be positive")
        if (
            isinstance(self.minimum_fully_judged_candidate_pairs, bool)
            or not isinstance(self.minimum_fully_judged_candidate_pairs, int)
            or self.minimum_fully_judged_candidate_pairs < 1
        ):
            raise ValueError("minimum_fully_judged_candidate_pairs must be positive")
        if (
            isinstance(self.minimum_confidence_level, bool)
            or not isinstance(self.minimum_confidence_level, (int, float))
            or not math.isfinite(self.minimum_confidence_level)
            or not 0.0 < self.minimum_confidence_level < 1.0
        ):
            raise ValueError("minimum_confidence_level must be between zero and one")
        if (
            isinstance(self.minimum_bootstrap_resamples, bool)
            or not isinstance(self.minimum_bootstrap_resamples, int)
            or self.minimum_bootstrap_resamples < 2
        ):
            raise ValueError("minimum_bootstrap_resamples must be at least two")
        bounded = (
            self.minimum_pooled_pruning_fraction,
            self.minimum_pooled_retained_relevant_recall,
            self.minimum_pruning_ci_lower,
            self.minimum_recall_ci_lower,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
            for value in bounded
        ):
            raise ValueError("pruning claim thresholds must be between zero and one")
        kinds = tuple(PruningStageKind(kind) for kind in self.required_stage_kinds)
        if not kinds:
            raise ValueError("required_stage_kinds must not be empty")
        if len(kinds) != len(set(kinds)):
            raise ValueError("required_stage_kinds must be unique")
        object.__setattr__(self, "required_stage_kinds", kinds)
        if not isinstance(self.require_corpus_qualification, bool):
            raise TypeError("require_corpus_qualification must be a boolean")

    def to_dict(self) -> dict[str, object]:
        return {
            "minimum_test_query_groups": self.minimum_test_query_groups,
            "minimum_fully_judged_candidate_pairs": (self.minimum_fully_judged_candidate_pairs),
            "minimum_confidence_level": self.minimum_confidence_level,
            "minimum_bootstrap_resamples": self.minimum_bootstrap_resamples,
            "minimum_pooled_pruning_fraction": self.minimum_pooled_pruning_fraction,
            "minimum_pooled_retained_relevant_recall": (
                self.minimum_pooled_retained_relevant_recall
            ),
            "minimum_pruning_ci_lower": self.minimum_pruning_ci_lower,
            "minimum_recall_ci_lower": self.minimum_recall_ci_lower,
            "required_stage_kinds": [kind.value for kind in self.required_stage_kinds],
            "require_corpus_qualification": self.require_corpus_qualification,
        }


@dataclass(frozen=True, slots=True)
class PruningClaimDecision:
    """Hashed pass/fail decision separate from basic evidence eligibility."""

    report_sha256: str
    passed: bool
    failures: tuple[str, ...]
    measured_values: Mapping[str, float | int | str | bool | None]
    policy: PruningClaimPolicy
    decision_sha256: str

    def content_payload(self) -> dict[str, object]:
        return {
            "schema_version": PRUNING_CLAIM_DECISION_SCHEMA_VERSION,
            "report_sha256": self.report_sha256,
            "decision": "claim_supported" if self.passed else "claim_not_supported",
            "passed": self.passed,
            "failures": list(self.failures),
            "measured_values": dict(self.measured_values),
            "policy": self.policy.to_dict(),
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.content_payload(), "decision_sha256": self.decision_sha256}

    def __post_init__(self) -> None:
        if self.passed and self.failures:
            raise ValueError("a passing pruning claim decision cannot contain failures")
        if not self.passed and not self.failures:
            raise ValueError("a failed pruning claim decision must explain why")
        if self.decision_sha256:
            _require_sha256(self.decision_sha256, "pruning claim decision hash")
        if self.decision_sha256 and sha256_json(self.content_payload()) != self.decision_sha256:
            raise ValueError("pruning claim decision hash does not match its contents")


@dataclass(frozen=True, slots=True)
class PruningQueryStageCounts:
    """Auditable counts used to derive one query's stage metrics."""

    candidate_before: int
    candidate_after: int
    candidate_initial: int
    relevant_before: int
    relevant_after: int
    relevant_initial: int

    def __post_init__(self) -> None:
        counts = (
            self.candidate_before,
            self.candidate_after,
            self.candidate_initial,
            self.relevant_before,
            self.relevant_after,
            self.relevant_initial,
        )
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0 for count in counts
        ):
            raise ValueError("query-stage counts must be non-negative integers")
        if not self.candidate_after <= self.candidate_before <= self.candidate_initial:
            raise ValueError("candidate counts must be monotonic")
        if not self.relevant_after <= self.relevant_before <= self.relevant_initial:
            raise ValueError("relevant counts must be monotonic")
        if self.relevant_initial > self.candidate_initial:
            raise ValueError("relevant count cannot exceed candidate count")
        if self.relevant_before > self.candidate_before:
            raise ValueError("relevant before-count cannot exceed candidate before-count")
        if self.relevant_after > self.candidate_after:
            raise ValueError("relevant after-count cannot exceed candidate after-count")

    def to_dict(self) -> dict[str, int]:
        return {
            "candidate_before": self.candidate_before,
            "candidate_after": self.candidate_after,
            "candidate_initial": self.candidate_initial,
            "relevant_before": self.relevant_before,
            "relevant_after": self.relevant_after,
            "relevant_initial": self.relevant_initial,
        }


def _ratio_interval(
    numerators: Sequence[int],
    denominators: Sequence[int],
    *,
    groups: Sequence[Hashable],
    confidence_level: float,
    n_resamples: int,
    seed: int,
) -> tuple[float, float]:
    if len(numerators) != len(denominators) or len(groups) != len(numerators):
        raise ValueError("ratio-bootstrap inputs must have equal lengths")
    grouped_indices: dict[Hashable, list[int]] = defaultdict(list)
    for index, group_id in enumerate(groups):
        grouped_indices[group_id].append(index)
    units = list(grouped_indices.values())
    generator = random.Random(seed)
    estimates: list[float] = []
    for _ in range(n_resamples):
        sampled: list[int] = []
        for _ in range(len(units)):
            sampled.extend(generator.choice(units))
        denominator = sum(denominators[index] for index in sampled)
        if denominator == 0:
            continue
        estimates.append(sum(numerators[index] for index in sampled) / denominator)
    if len(estimates) < 2:
        raise ValueError("ratio bootstrap did not produce enough defined estimates")
    estimates.sort()
    alpha = 1.0 - confidence_level

    def percentile(probability: float) -> float:
        position = (len(estimates) - 1) * probability
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return estimates[lower]
        fraction = position - lower
        return estimates[lower] * (1.0 - fraction) + estimates[upper] * fraction

    return percentile(alpha / 2.0), percentile(1.0 - alpha / 2.0)


def _pooled_estimate(
    name: str,
    numerators: Sequence[int],
    denominators: Sequence[int],
    *,
    groups: Sequence[Hashable],
    confidence_level: float,
    n_resamples: int,
    seed: int,
) -> MetricEstimate | None:
    numerator = sum(numerators)
    denominator = sum(denominators)
    if denominator == 0:
        return None
    value = numerator / denominator
    lower, upper = _ratio_interval(
        numerators,
        denominators,
        groups=groups,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        seed=seed,
    )
    return MetricEstimate(
        name=name,
        value=value,
        sample_count=len(numerators),
        ci_lower=min(lower, value),
        ci_upper=max(upper, value),
        confidence_level=confidence_level,
        numerator=numerator,
        denominator=denominator,
    )


def _macro_estimate(
    name: str,
    numerators: Sequence[int],
    denominators: Sequence[int],
    *,
    groups: Sequence[Hashable],
    confidence_level: float,
    n_resamples: int,
    seed: int,
) -> MetricEstimate | None:
    defined = [
        (numerator / denominator, group_id)
        for numerator, denominator, group_id in zip(numerators, denominators, groups, strict=True)
        if denominator > 0
    ]
    if not defined:
        return None
    values = [value for value, _ in defined]
    selected_groups = [group_id for _, group_id in defined]
    value = fmean(values)
    lower, upper = bootstrap_confidence_interval(
        values,
        groups=selected_groups,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        seed=seed,
    )
    return MetricEstimate(
        name=name,
        value=value,
        sample_count=len(values),
        ci_lower=min(lower, value),
        ci_upper=max(upper, value),
        confidence_level=confidence_level,
    )


def _stage_metrics(
    counts: Sequence[PruningQueryStageCounts],
    *,
    groups: Sequence[Hashable],
    confidence_level: float,
    n_resamples: int,
    seed: int,
) -> dict[str, MetricEstimate]:
    metric_inputs = {
        "incremental_pooled_pruning_fraction": (
            [item.candidate_before - item.candidate_after for item in counts],
            [item.candidate_before for item in counts],
            _pooled_estimate,
        ),
        "cumulative_pooled_pruning_fraction": (
            [item.candidate_initial - item.candidate_after for item in counts],
            [item.candidate_initial for item in counts],
            _pooled_estimate,
        ),
        "cumulative_pooled_retained_relevant_recall": (
            [item.relevant_after for item in counts],
            [item.relevant_initial for item in counts],
            _pooled_estimate,
        ),
        "incremental_pooled_retained_relevant_recall": (
            [item.relevant_after for item in counts],
            [item.relevant_before for item in counts],
            _pooled_estimate,
        ),
        "incremental_macro_pruning_fraction": (
            [item.candidate_before - item.candidate_after for item in counts],
            [item.candidate_before for item in counts],
            _macro_estimate,
        ),
        "cumulative_macro_pruning_fraction": (
            [item.candidate_initial - item.candidate_after for item in counts],
            [item.candidate_initial for item in counts],
            _macro_estimate,
        ),
        "cumulative_macro_retained_relevant_recall": (
            [item.relevant_after for item in counts],
            [item.relevant_initial for item in counts],
            _macro_estimate,
        ),
    }
    estimates: dict[str, MetricEstimate] = {}
    for offset, (name, (numerators, denominators, evaluator)) in enumerate(metric_inputs.items()):
        estimate = evaluator(
            name,
            numerators,
            denominators,
            groups=groups,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            seed=seed + offset,
        )
        if estimate is not None:
            estimates[name] = estimate
    return estimates


def _validate_stages(
    dataset: FrozenCandidateSet,
    stages: Sequence[FrozenPruningStage],
) -> None:
    if not stages:
        raise ValueError("at least one pruning stage is required")
    names = [stage.name for stage in stages]
    if len(names) != len(set(names)):
        raise ValueError("pruning stage names must be unique")
    expected_query_ids = {query.query_id for query in dataset.queries}
    previous = {query.query_id: set(query.candidate_ids) for query in dataset.queries}
    for stage in stages:
        if sha256_json(stage.content_payload()) != stage.checksum:
            raise ValueError(f"pruning stage {stage.name!r} changed after it was frozen")
        if set(stage.retained_candidate_ids) != expected_query_ids:
            raise ValueError(f"pruning stage {stage.name!r} must cover exactly the frozen queries")
        for query_id, candidate_ids in stage.retained_candidate_ids.items():
            retained = set(candidate_ids)
            outside = retained - previous[query_id]
            if outside:
                raise ValueError(
                    f"pruning stage {stage.name!r} is not a monotonic subset for "
                    f"query {query_id!r}: {sorted(outside)}"
                )
            removed = previous[query_id] - retained
            documented = set(stage.removal_reasons[query_id])
            if documented != removed:
                raise ValueError(
                    f"pruning stage {stage.name!r} removal reasons do not exactly cover "
                    f"removed candidates for query {query_id!r}"
                )
            previous[query_id] = retained


def _selected_queries(
    dataset: FrozenCandidateSet,
    *,
    query_split: QueryGroupSplit | None,
    split_name: str | None,
) -> tuple[tuple[FrozenCandidateQuery, ...], FrozenCandidateSet, Mapping[str, Hashable]]:
    if query_split is None:
        if split_name is not None:
            raise ValueError("split_name requires a frozen query-group split")
        return (
            dataset.queries,
            dataset,
            {query.query_id: query.query_id for query in dataset.queries},
        )
    if split_name is None:
        raise ValueError("split_name is required with a frozen query-group split")
    if sha256_json(query_split.content_payload()) != query_split.checksum:
        raise ValueError("frozen query-group split changed after hashing")
    query_split.validate_dataset(dataset)
    queries = query_split.queries_for(dataset, split_name)
    subset = query_split.subset(dataset, split_name)
    return (
        queries,
        subset,
        {query.query_id: query_split.query_group_ids[query.query_id] for query in queries},
    )


def evaluate_candidate_pruning(
    dataset: FrozenCandidateSet,
    stages: Sequence[FrozenPruningStage],
    *,
    population: CandidatePopulationDeclaration,
    provenance: PruningEvaluationProvenance,
    query_split: QueryGroupSplit | None = None,
    split_name: str | None = None,
    confidence_level: float = 0.95,
    n_resamples: int = 1_000,
    seed: int = 20260722,
) -> PruningEvaluationReport:
    """Evaluate monotonic filter stages against frozen relevance judgments.

    Pooled metrics weight query-candidate pairs equally.  Macro metrics weight
    queries equally.  Confidence intervals resample complete query-intent groups
    when a frozen split is supplied, preventing paraphrase leakage.
    """

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")
    if n_resamples < 2:
        raise ValueError("n_resamples must be at least two")
    refrozen_dataset = FrozenCandidateSet.create(
        dataset.version,
        dataset.queries,
        label_source=dataset.label_source,
        adjudication_complete=dataset.adjudication_complete,
        contains_synthetic_labels=dataset.contains_synthetic_labels,
        judgment_manifest_sha256=dataset.judgment_manifest_sha256,
    )
    if refrozen_dataset.checksum != dataset.checksum:
        raise ValueError("frozen candidate or relevance content changed after hashing")
    if refrozen_dataset.evidence_checksum != dataset.evidence_checksum:
        raise ValueError("frozen relevance evidence changed after hashing")
    if sha256_json(population.content_payload()) != population.checksum:
        raise ValueError("population declaration changed after hashing")
    if sha256_json(provenance.content_payload()) != provenance.checksum:
        raise ValueError("pruning provenance changed after hashing")
    population.validate_dataset(dataset)
    if provenance.catalog_manifest_sha256 != population.catalog_manifest_sha256:
        raise ValueError("population and run provenance catalog manifests do not match")
    _validate_stages(dataset, stages)
    queries, evaluation_dataset, bootstrap_groups = _selected_queries(
        dataset,
        query_split=query_split,
        split_name=split_name,
    )

    previous_by_query = {query.query_id: set(query.candidate_ids) for query in queries}
    initial_by_query = {query.query_id: set(query.candidate_ids) for query in queries}
    labels_by_query = {
        query.query_id: {
            candidate_id for candidate_id, grade in query.relevance_labels.items() if grade > 0
        }
        for query in queries
    }
    group_ids = [bootstrap_groups[query.query_id] for query in queries]
    stage_evaluations: list[PruningStageEvaluation] = []
    for stage_index, stage in enumerate(stages):
        query_counts: dict[str, PruningQueryStageCounts] = {}
        for query in queries:
            query_id = query.query_id
            retained = set(stage.retained_candidate_ids[query_id])
            previous = previous_by_query[query_id]
            initial = initial_by_query[query_id]
            relevant = labels_by_query[query_id]
            query_counts[query_id] = PruningQueryStageCounts(
                candidate_before=len(previous),
                candidate_after=len(retained),
                candidate_initial=len(initial),
                relevant_before=len(previous.intersection(relevant)),
                relevant_after=len(retained.intersection(relevant)),
                relevant_initial=len(relevant),
            )
            previous_by_query[query_id] = retained
        metrics = _stage_metrics(
            list(query_counts.values()),
            groups=group_ids,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            seed=seed + stage_index * 100,
        )
        stage_evaluations.append(
            PruningStageEvaluation(
                name=stage.name,
                kind=stage.kind,
                version=stage.version,
                stage_snapshot_sha256=stage.checksum,
                candidate_count_before=sum(item.candidate_before for item in query_counts.values()),
                candidate_count_after=sum(item.candidate_after for item in query_counts.values()),
                relevant_count_before=sum(item.relevant_before for item in query_counts.values()),
                relevant_count_after=sum(item.relevant_after for item in query_counts.values()),
                metrics=metrics,
                per_query=query_counts,
            )
        )

    qrels_complete = all(
        set(query.relevance_labels) == set(query.candidate_ids) for query in dataset.queries
    )
    evidence_block_reasons = list(dataset.promotion_block_reasons)
    if not qrels_complete:
        evidence_block_reasons.append(
            "relevance judgments do not cover every frozen candidate pair"
        )
    if query_split is None or split_name != "test":
        evidence_block_reasons.append("evaluation is not tied to the frozen test query-group split")
    evidence_eligible = not evidence_block_reasons
    block_reasons = list(evidence_block_reasons)
    if not population.corpus_qualified:
        block_reasons.append(
            "candidate population is a retrieval pool, not the full eligible corpus"
        )
    eligible = not block_reasons
    trace_hash = sha256_json(
        {
            "schema_version": PRUNING_TRACE_SCHEMA_VERSION,
            "stages": [stage.to_dict() for stage in stages],
        }
    )
    report_fields: dict[str, Any] = {
        "dataset_version": dataset.version,
        "candidate_checksum": dataset.checksum,
        "evidence_checksum": dataset.evidence_checksum,
        "judgment_manifest_sha256": dataset.judgment_manifest_sha256,
        "label_source": dataset.label_source.value,
        "adjudication_complete": dataset.adjudication_complete,
        "contains_synthetic_labels": dataset.contains_synthetic_labels,
        "qrels_complete": qrels_complete,
        "split_name": split_name,
        "split_checksum": query_split.checksum if query_split is not None else None,
        "evaluated_candidate_checksum": evaluation_dataset.checksum,
        "query_count": len(queries),
        "query_group_count": len(set(group_ids)),
        "population": population,
        "provenance": provenance,
        "pruning_trace_sha256": trace_hash,
        "stages": tuple(stage_evaluations),
        "evidence_eligible": evidence_eligible,
        "evidence_block_reasons": tuple(evidence_block_reasons),
        "eligible_for_promotion": eligible,
        "promotion_block_reasons": tuple(block_reasons),
        "confidence_level": confidence_level,
        "bootstrap_resamples": n_resamples,
        "bootstrap_seed": seed,
    }
    provisional = PruningEvaluationReport(report_sha256="", **report_fields)
    report_hash = sha256_json(provisional.content_payload())
    return PruningEvaluationReport(report_sha256=report_hash, **report_fields)


def evaluate_pruning_claim(
    report: PruningEvaluationReport,
    *,
    policy: PruningClaimPolicy | None = None,
) -> PruningClaimDecision:
    """Apply scope, sample-size, point-estimate, and uncertainty gates."""

    if sha256_json(report.content_payload()) != report.report_sha256:
        raise ValueError("pruning report changed after hashing")
    _validate_loaded_pruning_report(report.to_dict())
    selected_policy = policy or PruningClaimPolicy()
    failures: list[str] = []
    if not report.evidence_eligible:
        failures.extend(report.evidence_block_reasons)
    if selected_policy.require_corpus_qualification and not report.corpus_claim_qualified:
        failures.append("the report does not qualify a full eligible-corpus claim")
    if report.query_group_count < selected_policy.minimum_test_query_groups:
        failures.append(
            "frozen test query-group count is below the claim minimum: "
            f"{report.query_group_count} < {selected_policy.minimum_test_query_groups}"
        )
    if report.confidence_level < selected_policy.minimum_confidence_level:
        failures.append(
            "report confidence level is below the claim minimum: "
            f"{report.confidence_level:.6f} < {selected_policy.minimum_confidence_level:.6f}"
        )
    if report.bootstrap_resamples < selected_policy.minimum_bootstrap_resamples:
        failures.append(
            "bootstrap resample count is below the claim minimum: "
            f"{report.bootstrap_resamples} < {selected_policy.minimum_bootstrap_resamples}"
        )
    judged_pairs = report.stages[0].candidate_count_before
    if judged_pairs < selected_policy.minimum_fully_judged_candidate_pairs:
        failures.append(
            "fully judged candidate-pair count is below the claim minimum: "
            f"{judged_pairs} < {selected_policy.minimum_fully_judged_candidate_pairs}"
        )
    stages_by_kind: dict[PruningStageKind, list[tuple[int, PruningStageEvaluation]]] = defaultdict(
        list
    )
    for index, stage in enumerate(report.stages):
        stages_by_kind[stage.kind].append((index, stage))
    observed_stage_sequence = tuple(stage.kind for stage in report.stages)
    if observed_stage_sequence != selected_policy.required_stage_kinds:
        failures.append(
            "claim stage sequence must exactly match the policy-required filtering stages"
        )
    missing_stages = set(selected_policy.required_stage_kinds) - set(stages_by_kind)
    if missing_stages:
        failures.append(
            f"required pruning stages are missing: {sorted(kind.value for kind in missing_stages)}"
        )
    duplicate_required = [
        kind for kind in selected_policy.required_stage_kinds if len(stages_by_kind[kind]) > 1
    ]
    if duplicate_required:
        failures.append(
            "required pruning stages are duplicated: "
            f"{sorted(kind.value for kind in duplicate_required)}"
        )
    ordered_required_positions = [
        stages_by_kind[kind][0][0]
        for kind in selected_policy.required_stage_kinds
        if len(stages_by_kind[kind]) == 1
    ]
    if len(ordered_required_positions) == len(selected_policy.required_stage_kinds) and (
        ordered_required_positions != sorted(ordered_required_positions)
    ):
        failures.append("required pruning stages are not in the declared policy order")
    claim_stage = report.final_stage
    if ordered_required_positions:
        claim_stage = report.stages[ordered_required_positions[-1]]
        if ordered_required_positions[-1] != len(report.stages) - 1:
            failures.append("the terminal claim stage must be the final evaluated stage")
    compatibility_stages = stages_by_kind[PruningStageKind.COMPATIBILITY]
    if len(compatibility_stages) == 1 and (
        compatibility_stages[0][1].version != report.provenance.compatibility_rule_version
    ):
        failures.append(
            "compatibility stage version does not match provenance compatibility rule version"
        )

    claim_metrics = claim_stage.metrics
    pruning = claim_metrics.get("cumulative_pooled_pruning_fraction")
    recall = claim_metrics.get("cumulative_pooled_retained_relevant_recall")
    measured: dict[str, float | int | str | bool | None] = {
        "test_query_count": report.query_count,
        "test_query_group_count": report.query_group_count,
        "fully_judged_candidate_pairs": judged_pairs,
        "confidence_level": report.confidence_level,
        "bootstrap_resamples": report.bootstrap_resamples,
        "population_scope": report.population.scope.value,
        "corpus_claim_qualified": report.corpus_claim_qualified,
        "label_source": report.label_source,
        "split_name": report.split_name,
        "claim_stage_name": claim_stage.name,
        "claim_stage_kind": claim_stage.kind.value,
        "claim_stage_version": claim_stage.version,
        "pooled_pruning_fraction": pruning.value if pruning is not None else None,
        "pooled_pruning_ci_lower": pruning.ci_lower if pruning is not None else None,
        "pooled_retained_relevant_recall": recall.value if recall is not None else None,
        "pooled_retained_relevant_recall_ci_lower": (
            recall.ci_lower if recall is not None else None
        ),
    }
    if pruning is None:
        failures.append("cumulative pooled pruning fraction is missing")
    else:
        if pruning.value < selected_policy.minimum_pooled_pruning_fraction:
            failures.append("pooled pruning fraction is below the claim target")
        if pruning.ci_lower is None or pruning.ci_lower < selected_policy.minimum_pruning_ci_lower:
            failures.append("pooled pruning confidence lower bound is below the claim target")
    if recall is None:
        failures.append("cumulative pooled retained-relevant recall is missing")
    else:
        if recall.value < selected_policy.minimum_pooled_retained_relevant_recall:
            failures.append("pooled retained-relevant recall is below the claim target")
        if recall.ci_lower is None or recall.ci_lower < selected_policy.minimum_recall_ci_lower:
            failures.append("pooled recall confidence lower bound is below the claim target")

    unique_failures = tuple(dict.fromkeys(failures))
    provisional = PruningClaimDecision(
        report_sha256=report.report_sha256,
        passed=not unique_failures,
        failures=unique_failures,
        measured_values=measured,
        policy=selected_policy,
        decision_sha256="",
    )
    return PruningClaimDecision(
        report_sha256=report.report_sha256,
        passed=not unique_failures,
        failures=unique_failures,
        measured_values=measured,
        policy=selected_policy,
        decision_sha256=sha256_json(provisional.content_payload()),
    )


def write_pruning_evaluation_report(
    report: PruningEvaluationReport,
    path: str | Path,
) -> Path:
    """Atomically write a report after re-verifying its semantic hash."""

    if sha256_json(report.content_payload()) != report.report_sha256:
        raise ValueError("pruning report hash does not match its contents")
    _validate_loaded_pruning_report(report.to_dict())
    return _atomic_json(Path(path), report.to_dict())


def write_frozen_pruning_trace(
    stages: Sequence[FrozenPruningStage],
    path: str | Path,
) -> Path:
    """Persist the complete stage outputs needed to reproduce aggregate counts."""

    if not stages:
        raise ValueError("at least one pruning stage is required")
    for stage in stages:
        if sha256_json(stage.content_payload()) != stage.checksum:
            raise ValueError(f"pruning stage {stage.name!r} changed after it was frozen")
    content: dict[str, object] = {
        "schema_version": PRUNING_TRACE_SCHEMA_VERSION,
        "stages": [stage.to_dict() for stage in stages],
    }
    return _atomic_json(
        Path(path),
        {**content, "trace_sha256": sha256_json(content)},
    )


def load_frozen_pruning_trace(path: str | Path) -> tuple[FrozenPruningStage, ...]:
    """Load and independently verify every stage and the ordered trace hash."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("frozen pruning trace root must be an object")
    if payload.get("schema_version") != PRUNING_TRACE_SCHEMA_VERSION:
        raise ValueError("unsupported frozen pruning trace schema")
    raw_stages = payload.get("stages")
    if not isinstance(raw_stages, list) or not raw_stages:
        raise TypeError("frozen pruning trace stages must be a non-empty list")
    content = {
        "schema_version": PRUNING_TRACE_SCHEMA_VERSION,
        "stages": raw_stages,
    }
    if payload.get("trace_sha256") != sha256_json(content):
        raise ValueError("frozen pruning trace hash verification failed")
    stages: list[FrozenPruningStage] = []
    for raw_stage in raw_stages:
        if not isinstance(raw_stage, dict):
            raise TypeError("frozen pruning stage must be an object")
        if raw_stage.get("schema_version") != PRUNING_STAGE_SCHEMA_VERSION:
            raise ValueError("unsupported frozen pruning stage schema")
        raw_candidates = raw_stage.get("retained_candidate_ids")
        if not isinstance(raw_candidates, dict):
            raise TypeError("retained_candidate_ids must be an object")
        candidates: dict[str, tuple[str, ...]] = {}
        for query_id, candidate_ids in raw_candidates.items():
            if not isinstance(query_id, str) or not isinstance(candidate_ids, list):
                raise TypeError("trace candidate outputs must map strings to lists")
            if any(not isinstance(candidate_id, str) for candidate_id in candidate_ids):
                raise TypeError("trace candidate IDs must be strings")
            candidates[query_id] = tuple(candidate_ids)
        raw_removal_reasons = raw_stage.get("removal_reasons")
        if not isinstance(raw_removal_reasons, dict):
            raise TypeError("removal_reasons must be an object")
        removal_reasons: dict[str, dict[str, tuple[str, ...]]] = {}
        for query_id, raw_candidate_reasons in raw_removal_reasons.items():
            if not isinstance(query_id, str) or not isinstance(raw_candidate_reasons, dict):
                raise TypeError("trace removal reasons must map query IDs to objects")
            removal_reasons[query_id] = {}
            for candidate_id, raw_reasons in raw_candidate_reasons.items():
                if not isinstance(candidate_id, str) or not isinstance(raw_reasons, list):
                    raise TypeError("trace removed candidates must map to reason-code lists")
                if any(not isinstance(reason, str) for reason in raw_reasons):
                    raise TypeError("trace removal reason codes must be strings")
                removal_reasons[query_id][candidate_id] = tuple(raw_reasons)
        name = raw_stage.get("name")
        kind = raw_stage.get("kind")
        version = raw_stage.get("version")
        checksum = raw_stage.get("checksum")
        if (
            not isinstance(name, str)
            or not isinstance(kind, str)
            or not isinstance(version, str)
            or not isinstance(checksum, str)
        ):
            raise TypeError("trace stage name, kind, version, and checksum must be strings")
        stages.append(
            FrozenPruningStage(
                name=name,
                kind=PruningStageKind(kind),
                version=version,
                retained_candidate_ids=candidates,
                removal_reasons=removal_reasons,
                checksum=checksum,
            )
        )
    return tuple(stages)


def load_pruning_evaluation_report(path: str | Path) -> dict[str, Any]:
    """Load a pruning report and reject schema or content-hash drift."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("pruning evaluation report root must be an object")
    if payload.get("schema_version") != PRUNING_REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported pruning evaluation report schema")
    stored_hash = payload.get("report_sha256")
    unhashed = dict(payload)
    unhashed.pop("report_sha256", None)
    if stored_hash != sha256_json(unhashed):
        raise ValueError("pruning evaluation report hash verification failed")
    _validate_loaded_pruning_report(payload)
    return payload


def _strict_count(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{field_name} must be a non-negative integer")
    return value


def _require_dict(value: object, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be an object")
    return value


def _validate_loaded_metric(
    name: str,
    value: object,
    *,
    expected_confidence_level: float,
) -> None:
    metric = _require_dict(value, f"metric {name!r}")
    if metric.get("name") != name:
        raise ValueError(f"metric {name!r} has a mismatched name")
    measured = metric.get("value")
    if isinstance(measured, bool) or not isinstance(measured, (int, float)):
        raise TypeError(f"metric {name!r} value must be numeric")
    numeric_value = float(measured)
    if not math.isfinite(numeric_value) or not 0.0 <= numeric_value <= 1.0:
        raise ValueError(f"metric {name!r} must be a finite ratio")
    _strict_count(metric.get("sample_count"), f"metric {name!r} sample_count")
    interval = _require_dict(
        metric.get("confidence_interval"),
        f"metric {name!r} confidence_interval",
    )
    bounds: list[float] = []
    for key in ("lower", "upper", "confidence_level"):
        raw = interval.get(key)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise TypeError(f"metric {name!r} interval {key} must be numeric")
        bounds.append(float(raw))
    lower, upper, confidence = bounds
    if not 0.0 <= lower <= numeric_value <= upper <= 1.0:
        raise ValueError(f"metric {name!r} interval must contain its ratio")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"metric {name!r} confidence level is invalid")
    if not math.isclose(confidence, expected_confidence_level, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"metric {name!r} confidence level differs from the report protocol")
    numerator = metric.get("numerator")
    denominator = metric.get("denominator")
    if (numerator is None) != (denominator is None):
        raise ValueError(f"metric {name!r} count fields must be supplied together")
    if numerator is not None and denominator is not None:
        numerator_count = _strict_count(numerator, f"metric {name!r} numerator")
        denominator_count = _strict_count(denominator, f"metric {name!r} denominator")
        if denominator_count == 0 or numerator_count > denominator_count:
            raise ValueError(f"metric {name!r} ratio counts are invalid")
        if not math.isclose(
            numeric_value,
            numerator_count / denominator_count,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"metric {name!r} value does not match its counts")


def _validate_loaded_pruning_report(payload: Mapping[str, Any]) -> None:
    dataset = _require_dict(payload.get("dataset"), "dataset")
    for name in (
        "candidate_checksum",
        "evidence_checksum",
        "evaluated_candidate_checksum",
    ):
        value = dataset.get(name)
        if not isinstance(value, str):
            raise TypeError(f"dataset {name} must be a string")
        _require_sha256(value, f"dataset {name}")
    judgment_hash = dataset.get("judgment_manifest_sha256")
    if judgment_hash is not None:
        if not isinstance(judgment_hash, str):
            raise TypeError("judgment manifest hash must be a string or null")
        _require_sha256(judgment_hash, "judgment manifest hash")
    split_hash = dataset.get("split_checksum")
    if split_hash is not None:
        if not isinstance(split_hash, str):
            raise TypeError("split checksum must be a string or null")
        _require_sha256(split_hash, "split checksum")
    raw_label_source = dataset.get("label_source")
    if not isinstance(raw_label_source, str):
        raise TypeError("dataset label_source must be a string")
    label_source = RelevanceLabelSource(raw_label_source)
    for name in ("adjudication_complete", "contains_synthetic_labels", "qrels_complete"):
        if not isinstance(dataset.get(name), bool):
            raise TypeError(f"dataset {name} must be a boolean")
    _strict_count(dataset.get("query_count"), "dataset query_count")
    _strict_count(dataset.get("query_group_count"), "dataset query_group_count")

    parameters = _require_dict(payload.get("evaluation_parameters"), "evaluation_parameters")
    raw_confidence = parameters.get("confidence_level")
    if isinstance(raw_confidence, bool) or not isinstance(raw_confidence, (int, float)):
        raise TypeError("evaluation confidence_level must be numeric")
    confidence_level = float(raw_confidence)
    if not math.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("evaluation confidence_level must be between zero and one")
    bootstrap_resamples = _strict_count(
        parameters.get("bootstrap_resamples"), "evaluation bootstrap_resamples"
    )
    if bootstrap_resamples < 2:
        raise ValueError("evaluation bootstrap_resamples must be at least two")
    _strict_count(parameters.get("bootstrap_seed"), "evaluation bootstrap_seed")

    for field_name in ("population", "provenance"):
        nested = _require_dict(payload.get(field_name), field_name)
        checksum = nested.get("checksum")
        if not isinstance(checksum, str):
            raise TypeError(f"{field_name} checksum must be a string")
        unhashed = dict(nested)
        unhashed.pop("checksum", None)
        if checksum != sha256_json(unhashed):
            raise ValueError(f"{field_name} nested hash verification failed")

    population = _require_dict(payload.get("population"), "population")
    if population.get("schema_version") != PRUNING_POPULATION_SCHEMA_VERSION:
        raise ValueError("unsupported pruning population schema")
    raw_scope = population.get("scope")
    if not isinstance(raw_scope, str):
        raise TypeError("population scope must be a string")
    scope = CandidatePopulationScope(raw_scope)
    completeness = population.get("complete_eligible_corpus")
    corpus_qualified = population.get("corpus_qualified")
    membership_verified = population.get("catalog_membership_verified")
    membership_hash = population.get("catalog_membership_sha256")
    if (
        not isinstance(completeness, bool)
        or not isinstance(corpus_qualified, bool)
        or not isinstance(membership_verified, bool)
    ):
        raise TypeError("population qualification fields must be booleans")
    if membership_hash is not None:
        if not isinstance(membership_hash, str):
            raise TypeError("catalog membership hash must be a string or null")
        _require_sha256(membership_hash, "catalog membership hash")
    expected_corpus_qualified = (
        scope is CandidatePopulationScope.FULL_ELIGIBLE_CORPUS
        and completeness
        and membership_verified
        and membership_hash is not None
    )
    if corpus_qualified is not expected_corpus_qualified:
        raise ValueError("population corpus qualification is inconsistent")
    raw_population_counts = _require_dict(
        population.get("candidate_counts_by_query"),
        "population candidate_counts_by_query",
    )
    for query_id, count in raw_population_counts.items():
        if not query_id:
            raise ValueError("population query IDs must not be empty")
        _strict_count(count, f"population count for {query_id!r}")

    provenance = _require_dict(payload.get("provenance"), "provenance")
    if provenance.get("schema_version") != PRUNING_PROVENANCE_SCHEMA_VERSION:
        raise ValueError("unsupported pruning provenance schema")
    for name in (
        "run_id",
        "evaluated_at_utc",
        "pipeline_version",
        "compatibility_rule_version",
        "data_version",
        "code_revision",
    ):
        value = provenance.get(name)
        if not isinstance(value, str) or not value:
            raise TypeError(f"provenance {name} must be a non-empty string")
    _validate_utc_timestamp(str(provenance["evaluated_at_utc"]))
    for name in ("catalog_manifest_sha256", "filter_configuration_sha256"):
        value = provenance.get(name)
        if not isinstance(value, str):
            raise TypeError(f"provenance {name} must be a string")
        _require_sha256(value, f"provenance {name}")

    trace_hash = payload.get("pruning_trace_sha256")
    if not isinstance(trace_hash, str):
        raise TypeError("pruning_trace_sha256 must be a string")
    _require_sha256(trace_hash, "pruning trace hash")
    raw_stages = payload.get("stages")
    if not isinstance(raw_stages, list) or not raw_stages:
        raise TypeError("pruning report stages must be a non-empty list")
    observed_names: set[str] = set()
    previous_candidates_by_query: dict[str, int] | None = None
    initial_candidates_by_query: dict[str, int] | None = None
    for stage_index, raw_stage in enumerate(raw_stages):
        stage = _require_dict(raw_stage, f"stages[{stage_index}]")
        stage_name = stage.get("name")
        if not isinstance(stage_name, str) or not stage_name or stage_name in observed_names:
            raise ValueError("pruning report stage names must be unique non-empty strings")
        observed_names.add(stage_name)
        kind = stage.get("kind")
        version = stage.get("version")
        if not isinstance(kind, str) or not isinstance(version, str) or not version:
            raise TypeError("pruning report stage kind and version must be strings")
        PruningStageKind(kind)
        snapshot_hash = stage.get("stage_snapshot_sha256")
        if not isinstance(snapshot_hash, str):
            raise TypeError("stage snapshot hash must be a string")
        _require_sha256(snapshot_hash, "stage snapshot hash")
        counts = _require_dict(stage.get("counts"), f"stage {stage_name!r} counts")
        candidate_counts = _require_dict(
            counts.get("candidate_query_pairs"),
            f"stage {stage_name!r} candidate counts",
        )
        relevant_counts = _require_dict(
            counts.get("judged_relevant_query_pairs"),
            f"stage {stage_name!r} relevant counts",
        )
        candidate_before = _strict_count(candidate_counts.get("before"), "candidate before")
        candidate_after = _strict_count(candidate_counts.get("after"), "candidate after")
        candidate_pruned = _strict_count(candidate_counts.get("pruned"), "candidate pruned")
        relevant_before = _strict_count(relevant_counts.get("before"), "relevant before")
        relevant_after = _strict_count(relevant_counts.get("after"), "relevant after")
        relevant_lost = _strict_count(relevant_counts.get("lost"), "relevant lost")
        if candidate_before - candidate_after != candidate_pruned:
            raise ValueError("aggregate candidate pruning counts are inconsistent")
        if relevant_before - relevant_after != relevant_lost:
            raise ValueError("aggregate relevant retention counts are inconsistent")
        per_query = _require_dict(stage.get("per_query"), f"stage {stage_name!r} per_query")
        if not per_query:
            raise ValueError("stage per_query counts must not be empty")
        sum_candidate_before = 0
        sum_candidate_after = 0
        sum_relevant_before = 0
        sum_relevant_after = 0
        stage_candidate_before: dict[str, int] = {}
        stage_candidate_after: dict[str, int] = {}
        stage_candidate_initial: dict[str, int] = {}
        stage_rows: list[dict[str, int]] = []
        for query_id, raw_query_counts in per_query.items():
            if not query_id:
                raise ValueError("per-query count IDs must not be empty")
            query_counts = _require_dict(raw_query_counts, f"per_query {query_id!r}")
            values = {
                key: _strict_count(query_counts.get(key), f"per_query {query_id!r} {key}")
                for key in (
                    "candidate_before",
                    "candidate_after",
                    "candidate_initial",
                    "relevant_before",
                    "relevant_after",
                    "relevant_initial",
                )
            }
            if not (
                values["candidate_after"]
                <= values["candidate_before"]
                <= values["candidate_initial"]
            ):
                raise ValueError("per-query candidate counts are not monotonic")
            if not (
                values["relevant_after"] <= values["relevant_before"] <= values["relevant_initial"]
            ):
                raise ValueError("per-query relevant counts are not monotonic")
            if (
                values["relevant_initial"] > values["candidate_initial"]
                or values["relevant_before"] > values["candidate_before"]
                or values["relevant_after"] > values["candidate_after"]
            ):
                raise ValueError("per-query relevant counts cannot exceed candidate counts")
            sum_candidate_before += values["candidate_before"]
            sum_candidate_after += values["candidate_after"]
            sum_relevant_before += values["relevant_before"]
            sum_relevant_after += values["relevant_after"]
            stage_candidate_before[query_id] = values["candidate_before"]
            stage_candidate_after[query_id] = values["candidate_after"]
            stage_candidate_initial[query_id] = values["candidate_initial"]
            stage_rows.append(values)
        if (
            sum_candidate_before != candidate_before
            or sum_candidate_after != candidate_after
            or sum_relevant_before != relevant_before
            or sum_relevant_after != relevant_after
        ):
            raise ValueError("aggregate counts do not equal the per-query counts")
        if previous_candidates_by_query is not None and (
            stage_candidate_before != previous_candidates_by_query
        ):
            raise ValueError("sequential stage before-counts do not match prior after-counts")
        if initial_candidates_by_query is None:
            initial_candidates_by_query = stage_candidate_initial
        elif stage_candidate_initial != initial_candidates_by_query:
            raise ValueError("initial candidate counts changed between pruning stages")
        previous_candidates_by_query = stage_candidate_after
        metrics = _require_dict(stage.get("metrics"), f"stage {stage_name!r} metrics")
        metric_inputs: dict[str, tuple[list[int], list[int], bool]] = {
            "incremental_pooled_pruning_fraction": (
                [row["candidate_before"] - row["candidate_after"] for row in stage_rows],
                [row["candidate_before"] for row in stage_rows],
                True,
            ),
            "cumulative_pooled_pruning_fraction": (
                [row["candidate_initial"] - row["candidate_after"] for row in stage_rows],
                [row["candidate_initial"] for row in stage_rows],
                True,
            ),
            "cumulative_pooled_retained_relevant_recall": (
                [row["relevant_after"] for row in stage_rows],
                [row["relevant_initial"] for row in stage_rows],
                True,
            ),
            "incremental_pooled_retained_relevant_recall": (
                [row["relevant_after"] for row in stage_rows],
                [row["relevant_before"] for row in stage_rows],
                True,
            ),
            "incremental_macro_pruning_fraction": (
                [row["candidate_before"] - row["candidate_after"] for row in stage_rows],
                [row["candidate_before"] for row in stage_rows],
                False,
            ),
            "cumulative_macro_pruning_fraction": (
                [row["candidate_initial"] - row["candidate_after"] for row in stage_rows],
                [row["candidate_initial"] for row in stage_rows],
                False,
            ),
            "cumulative_macro_retained_relevant_recall": (
                [row["relevant_after"] for row in stage_rows],
                [row["relevant_initial"] for row in stage_rows],
                False,
            ),
        }
        expected_metrics = {
            name for name, (_, denominators, _) in metric_inputs.items() if sum(denominators) > 0
        }
        if set(metrics) != expected_metrics:
            raise ValueError("stage metrics do not match the defined pruning metric set")
        for metric_name, metric in metrics.items():
            _validate_loaded_metric(
                metric_name,
                metric,
                expected_confidence_level=confidence_level,
            )
            metric_payload = _require_dict(metric, f"metric {metric_name!r}")
            numerators, denominators, pooled = metric_inputs[metric_name]
            defined = [
                (numerator, denominator)
                for numerator, denominator in zip(numerators, denominators, strict=True)
                if denominator > 0
            ]
            if pooled:
                expected_numerator = sum(numerators)
                expected_denominator = sum(denominators)
                expected_value = expected_numerator / expected_denominator
                expected_sample_count = len(numerators)
                if (
                    metric_payload.get("numerator") != expected_numerator
                    or metric_payload.get("denominator") != expected_denominator
                ):
                    raise ValueError(f"metric {metric_name!r} counts do not match per-query data")
            else:
                expected_value = fmean(
                    numerator / denominator for numerator, denominator in defined
                )
                expected_sample_count = len(defined)
                if (
                    metric_payload.get("numerator") is not None
                    or metric_payload.get("denominator") is not None
                ):
                    raise ValueError(f"macro metric {metric_name!r} cannot carry pooled counts")
            if metric_payload.get("sample_count") != expected_sample_count:
                raise ValueError(f"metric {metric_name!r} sample count is inconsistent")
            measured_value = metric_payload.get("value")
            if not isinstance(measured_value, (int, float)) or not math.isclose(
                float(measured_value), expected_value, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(f"metric {metric_name!r} does not match per-query counts")

    query_count = _strict_count(dataset.get("query_count"), "dataset query_count")
    query_group_count = _strict_count(dataset.get("query_group_count"), "dataset query_group_count")
    if not 1 <= query_group_count <= query_count:
        raise ValueError("query_group_count must be between one and query_count")
    if initial_candidates_by_query is None or len(initial_candidates_by_query) != query_count:
        raise ValueError("stage query coverage does not match the dataset query count")
    if population.get("dataset_checksum") != dataset.get("candidate_checksum"):
        raise ValueError("population and report candidate checksums do not match")
    if population.get("dataset_evidence_checksum") != dataset.get("evidence_checksum"):
        raise ValueError("population and report evidence checksums do not match")
    if provenance.get("catalog_manifest_sha256") != population.get("catalog_manifest_sha256"):
        raise ValueError("population and provenance catalog manifests do not match")
    for query_id, initial_count in initial_candidates_by_query.items():
        if raw_population_counts.get(query_id) != initial_count:
            raise ValueError("population candidate counts do not match evaluated stage inputs")
    claim = _require_dict(payload.get("claim_qualification"), "claim_qualification")
    for name in (
        "corpus_claim_qualified",
        "evidence_eligible",
        "judged_pool_claim_eligible",
        "corpus_claim_eligible",
        "eligible_for_promotion",
    ):
        if not isinstance(claim.get(name), bool):
            raise TypeError(f"claim qualification {name} must be a boolean")
    if claim.get("corpus_claim_qualified") is not corpus_qualified:
        raise ValueError("report and population corpus qualification disagree")
    if claim.get("judged_pool_claim_eligible") is not claim.get("evidence_eligible"):
        raise ValueError("judged-pool eligibility must match evidence eligibility")
    expected_corpus_eligible = bool(claim.get("evidence_eligible")) and corpus_qualified
    if claim.get("corpus_claim_eligible") is not expected_corpus_eligible:
        raise ValueError("corpus claim eligibility is inconsistent")
    if claim.get("eligible_for_promotion") is not expected_corpus_eligible:
        raise ValueError("promotion eligibility is inconsistent")
    if claim.get("relevance_basis") != f"{label_source.value}_qrels":
        raise ValueError("claim relevance basis does not match the dataset label source")
    if claim.get("relevance_threshold") != "grade > 0":
        raise ValueError("unsupported pruning relevance threshold")
    block_reasons = claim.get("promotion_block_reasons")
    evidence_reasons = claim.get("evidence_block_reasons")
    if not isinstance(block_reasons, list) or not isinstance(evidence_reasons, list):
        raise TypeError("claim block reasons must be lists")
    if any(not isinstance(reason, str) or not reason for reason in block_reasons):
        raise TypeError("promotion block reasons must be non-empty strings")
    if any(not isinstance(reason, str) or not reason for reason in evidence_reasons):
        raise TypeError("evidence block reasons must be non-empty strings")
    expected_evidence_eligible = (
        label_source is RelevanceLabelSource.HUMAN
        and dataset.get("adjudication_complete") is True
        and dataset.get("contains_synthetic_labels") is False
        and dataset.get("qrels_complete") is True
        and judgment_hash is not None
        and dataset.get("split_name") == "test"
        and isinstance(dataset.get("split_checksum"), str)
    )
    if claim.get("evidence_eligible") is not expected_evidence_eligible:
        raise ValueError("evidence eligibility is inconsistent with frozen provenance")
