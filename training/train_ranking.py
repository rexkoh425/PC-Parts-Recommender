"""Train LambdaMART from query-grouped, graded candidates with strict evidence gates.

Input is JSONL with one query per row::

    {"context": {"query_id": "q1", ...},
     "candidates": [{"product_id": "gpu-1", "category": "gpu",
                     "relevance_grade": 4, ...}]}

Candidate fields follow :class:`ScoredCandidate`.  Silver or synthetic labels may be
used only for an explicitly enabled pipeline diagnostic and always produce a
non-promotable artifact.  The retrieval silver pilot's frozen-candidate document alone
is intentionally insufficient: ranking features must be snapshotted with every grade.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pc_build_recommender.evaluation.manifest import json_sha256
from pc_build_recommender.evaluation.splits import deterministic_group_split
from pc_build_recommender.ranking import (
    LabeledRankingQuery,
    LambdaMARTRanker,
    RankerMetadata,
    ScoredCandidate,
    RankingContext,
    relative_ndcg_improvement,
)
from pc_build_recommender.retrieval import (
    PinnedCandidateSet,
    QueryGroupSplit,
    HumanJudgmentSet,
    RelevanceLabelSource,
    load_human_judgment_set,
    ndcg_at_k,
)
from training._common import (
    estimate_materialized_file_memory_mib,
    print_json,
    read_json_lines,
    require_host_memory_headroom,
    sha256_file,
    sha256_text,
    utc_now_iso,
    write_json,
)
from training.materialize_ranking_snapshot import verify_labeled_ranking_snapshot
from training.mlflow_tracking import (
    OptionalMLflowRun,
    add_mlflow_arguments,
    promotion_blocker_tags,
    tracking_config_from_args,
)


def _json_object(value: str) -> dict[str, Any]:
    decoded = json.loads(value)
    if not isinstance(decoded, dict):
        raise argparse.ArgumentTypeError("expected a JSON object")
    return decoded


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object")
    return {str(key): item for key, item in value.items()}


@dataclass(frozen=True, slots=True)
class VerifiedRankingEvidence:
    """Verified lineage used by both the trainer and the promotion report."""

    label_source: RelevanceLabelSource
    adjudication_complete: bool
    contains_synthetic_labels: bool
    judgment_manifest_sha256: str | None
    minimum_independent_reviewers: int
    qrels_path: Path | None = None
    qrels_file_sha256: str | None = None
    qrels_checksum: str | None = None
    qrels_evidence_checksum: str | None = None
    human_judgments_path: Path | None = None
    human_judgments_file_sha256: str | None = None
    frozen_query_split_path: Path | None = None
    frozen_query_split_file_sha256: str | None = None
    frozen_query_split: QueryGroupSplit | None = None
    dataset_manifest_path: Path | None = None
    dataset_manifest_file_sha256: str | None = None
    dataset_manifest_sha256: str | None = None
    prelabel_snapshot_sha256: str | None = None
    feature_contract_sha256: str | None = None
    annotation_release_sha256: str | None = None


class _RankingScorePredictor(Protocol):
    def predict(
        self,
        context: RankingContext,
        candidates: Sequence[ScoredCandidate],
    ) -> Any:
        """Return one finite ranking score per candidate."""


def _minimum_independent_reviewers(dataset: HumanJudgmentSet) -> int:
    reviewers_by_pair: dict[tuple[str, str], set[str]] = {}
    for judgment in dataset.judgments:
        reviewers_by_pair.setdefault((judgment.query_id, judgment.product_id), set()).add(
            judgment.reviewer_id
        )
    return min((len(reviewers) for reviewers in reviewers_by_pair.values()), default=0)


def _load_queries(path: Path) -> tuple[LabeledRankingQuery, ...]:
    queries: list[LabeledRankingQuery] = []
    for row_number, row in enumerate(read_json_lines(path), start=1):
        context_payload = _mapping(row.get("context"), field=f"row {row_number} context")
        raw_candidates = row.get("candidates")
        if not isinstance(raw_candidates, list):
            raise ValueError(f"row {row_number} candidates must be a JSON array")
        candidates: list[ScoredCandidate] = []
        grades: list[int] = []
        for candidate_number, raw_candidate in enumerate(raw_candidates, start=1):
            payload = _mapping(
                raw_candidate,
                field=f"row {row_number} candidate {candidate_number}",
            )
            try:
                raw_grade = payload.pop("relevance_grade")
            except KeyError as error:
                raise ValueError(
                    f"row {row_number} candidate {candidate_number} lacks relevance_grade"
                ) from error
            if type(raw_grade) is not int or not 0 <= raw_grade <= 4:
                raise ValueError(
                    f"row {row_number} candidate {candidate_number} relevance_grade "
                    "must be an integer from 0 to 4"
                )
            grade = raw_grade
            candidates.append(ScoredCandidate(**payload))
            grades.append(grade)
        queries.append(
            LabeledRankingQuery.create(
                RankingContext(**context_payload),
                candidates,
                grades,
            )
        )
    query_ids = [query.context.query_id for query in queries]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("query IDs must be unique in the ranking dataset")
    if len(queries) < 3:
        raise ValueError("ranking data needs at least three queries for grouped splits")
    return tuple(queries)


def _validate_feature_snapshot_against_qrels(
    queries: Sequence[LabeledRankingQuery],
    qrels: PinnedCandidateSet,
) -> None:
    """Reject feature rows whose candidate universe or grades differ from qrels."""

    actual_by_id = {query.context.query_id: query for query in queries}
    expected_by_id = {query.query_id: query for query in qrels.queries}
    if set(actual_by_id) != set(expected_by_id):
        missing = sorted(set(expected_by_id) - set(actual_by_id))
        extra = sorted(set(actual_by_id) - set(expected_by_id))
        raise ValueError(
            "ranking feature snapshot query IDs do not exactly match qrels; "
            f"missing={missing}, extra={extra}"
        )
    for query_id, expected in expected_by_id.items():
        actual = actual_by_id[query_id]
        actual_grades = {
            candidate.product_id: grade
            for candidate, grade in zip(actual.candidates, actual.relevance_grades, strict=True)
        }
        if set(actual_grades) != set(expected.candidate_ids):
            missing = sorted(set(expected.candidate_ids) - set(actual_grades))
            extra = sorted(set(actual_grades) - set(expected.candidate_ids))
            raise ValueError(
                f"ranking feature snapshot candidates differ from qrels for {query_id!r}; "
                f"missing={missing}, extra={extra}"
            )
        if actual_grades != dict(expected.relevance_labels):
            raise ValueError(
                f"ranking feature snapshot grades differ from adjudicated qrels for {query_id!r}"
            )
        if expected.category is not None and any(
            candidate.category.casefold() != expected.category for candidate in actual.candidates
        ):
            raise ValueError(
                f"ranking feature snapshot category differs from qrels for {query_id!r}"
            )


def _load_verified_human_evidence(
    *,
    ranking_path: Path,
    dataset_manifest_path: Path,
    human_judgments_path: Path,
    qrels_path: Path,
    frozen_query_split_path: Path,
    candidate_set_version: str,
    queries: Sequence[LabeledRankingQuery],
) -> VerifiedRankingEvidence:
    for path in (human_judgments_path, qrels_path, frozen_query_split_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    human_judgments = load_human_judgment_set(human_judgments_path)
    adjudicated = human_judgments.adjudicate()
    qrels = PinnedCandidateSet.load(qrels_path)
    expected_qrels = adjudicated.frozen_candidates
    if (
        qrels.version != expected_qrels.version
        or qrels.checksum != expected_qrels.checksum
        or qrels.evidence_checksum != expected_qrels.evidence_checksum
    ):
        raise ValueError(
            "frozen qrels do not match the independently reviewed and adjudicated judgments"
        )
    if qrels.judgment_manifest_sha256 != human_judgments.content_sha256:
        raise ValueError("qrels judgment manifest hash does not match the human judgments")
    if candidate_set_version != qrels.version:
        raise ValueError(
            "candidate-set version does not match qrels: "
            f"{candidate_set_version!r} != {qrels.version!r}"
        )

    frozen_query_split = QueryGroupSplit.load(frozen_query_split_path)
    frozen_query_split.validate_dataset(qrels)
    expected_query_groups = {query.query_id: query.query_group_id for query in qrels.queries}
    if any(group_id is None for group_id in expected_query_groups.values()):
        raise ValueError("every human qrels query must have a query-group ID")
    if dict(frozen_query_split.query_group_ids) != expected_query_groups:
        raise ValueError("frozen split query groups do not match the human qrels")
    _validate_feature_snapshot_against_qrels(queries, qrels)
    verified_snapshot = verify_labeled_ranking_snapshot(
        ranking_path=ranking_path,
        manifest_path=dataset_manifest_path,
        human_judgments_path=human_judgments_path,
        qrels_path=qrels_path,
        query_split_path=frozen_query_split_path,
    )
    return VerifiedRankingEvidence(
        label_source=qrels.label_source,
        adjudication_complete=qrels.adjudication_complete,
        contains_synthetic_labels=qrels.contains_synthetic_labels,
        judgment_manifest_sha256=qrels.judgment_manifest_sha256,
        minimum_independent_reviewers=_minimum_independent_reviewers(human_judgments),
        qrels_path=qrels_path.resolve(),
        qrels_file_sha256=sha256_file(qrels_path),
        qrels_checksum=qrels.checksum,
        qrels_evidence_checksum=qrels.evidence_checksum,
        human_judgments_path=human_judgments_path.resolve(),
        human_judgments_file_sha256=sha256_file(human_judgments_path),
        frozen_query_split_path=frozen_query_split_path.resolve(),
        frozen_query_split_file_sha256=sha256_file(frozen_query_split_path),
        frozen_query_split=frozen_query_split,
        dataset_manifest_path=verified_snapshot.manifest_path,
        dataset_manifest_file_sha256=verified_snapshot.manifest_file_sha256,
        dataset_manifest_sha256=verified_snapshot.manifest_sha256,
        prelabel_snapshot_sha256=verified_snapshot.prelabel_snapshot_sha256,
        feature_contract_sha256=verified_snapshot.feature_contract_sha256,
        annotation_release_sha256=verified_snapshot.annotation_release_sha256,
    )


def _split_queries(
    queries: Sequence[LabeledRankingQuery], *, seed: int
) -> dict[str, tuple[LabeledRankingQuery, ...]]:
    query_ids = [query.context.query_id for query in queries]
    split = deterministic_group_split(
        query_ids,
        weights={"train": 0.6, "validation": 0.2, "test": 0.2},
        seed=seed,
    )
    result: dict[str, list[LabeledRankingQuery]] = {name: [] for name in split.weights}
    for query in queries:
        result[split.split_for(query.context.query_id)].append(query)
    if len({grade for query in result["train"] for grade in query.relevance_grades}) < 2:
        raise ValueError("ranking training split needs at least two relevance grades")
    return {name: tuple(rows) for name, rows in result.items()}


def _split_queries_from_frozen(
    queries: Sequence[LabeledRankingQuery],
    frozen_query_split: QueryGroupSplit,
) -> dict[str, tuple[LabeledRankingQuery, ...]]:
    required_names = {"train", "validation", "test"}
    if set(frozen_query_split.weights) != required_names:
        raise ValueError("frozen query split must contain exactly train, validation, and test")
    query_ids = {query.context.query_id for query in queries}
    if query_ids != set(frozen_query_split.assignments):
        raise ValueError("ranking feature snapshot does not exactly match frozen split query IDs")
    result: dict[str, list[LabeledRankingQuery]] = {name: [] for name in required_names}
    for query in queries:
        result[frozen_query_split.assignments[query.context.query_id]].append(query)
    for split_name, rows in result.items():
        if not rows:
            raise ValueError(f"frozen {split_name} query split must not be empty")
    if len({grade for query in result["train"] for grade in query.relevance_grades}) < 2:
        raise ValueError("ranking training split needs at least two relevance grades")
    return {name: tuple(rows) for name, rows in result.items()}


def _mean_ndcg(
    ranker: _RankingScorePredictor,
    queries: Sequence[LabeledRankingQuery],
) -> tuple[float, float]:
    learned: list[float] = []
    bm25: list[float] = []
    for query in queries:
        labels = {
            candidate.product_id: grade
            for candidate, grade in zip(
                query.candidates,
                query.relevance_grades,
                strict=True,
            )
        }
        learned_scores = ranker.predict(query.context, query.candidates)
        learned_ranking = tuple(
            candidate.product_id
            for candidate, _score in sorted(
                zip(query.candidates, learned_scores, strict=True),
                key=lambda item: (-float(item[1]), item[0].product_id),
            )
        )
        bm25_ranking = tuple(
            candidate.product_id
            for candidate in sorted(
                query.candidates,
                key=lambda item: (
                    -float(item.retrieval_scores.get("bm25_score", 0.0)),
                    item.product_id,
                ),
            )
        )
        learned.append(ndcg_at_k(labels, learned_ranking, k=10))
        bm25.append(ndcg_at_k(labels, bm25_ranking, k=10))
    return sum(learned) / len(learned), sum(bm25) / len(bm25)


def _promotion_blockers(
    *,
    ranker_metadata: RankerMetadata,
    minimum_independent_reviewers: int,
    query_count: int,
    row_count: int,
    relative_improvement: float | None,
) -> list[str]:
    blockers = list(ranker_metadata.promotion_block_reasons)
    if not ranker_metadata.promotion_eligible and not blockers:
        blockers.append("ranker metadata is not promotion-eligible")
    if minimum_independent_reviewers < 2:
        blockers.append("fewer than two independent human relevance reviewers")
    if query_count < 150:
        blockers.append("fewer than the target 150 independently graded queries")
    if row_count < 2000:
        blockers.append("fewer than the target 2,000 graded query-product rows")
    if relative_improvement is None or relative_improvement < 15.0:
        blockers.append("held-out NDCG@10 improvement has not reached the 15% target")
    return blockers


def _training_publication_intent_sha256(
    *,
    source_sha256: str,
    training_data_version: str,
    candidate_set_version: str,
    seed: int,
    early_stopping_rounds: int,
    ranker: LambdaMARTRanker,
    evidence: VerifiedRankingEvidence,
) -> str:
    """Bind an idempotent publication to exact inputs and training configuration."""

    split = evidence.frozen_query_split
    return json_sha256(
        {
            "schema_version": "pc-build-recommender.ranking-training-publication-intent.v1",
            "feature_snapshot_sha256": source_sha256,
            "training_data_version": training_data_version,
            "candidate_set_version": candidate_set_version,
            "ranker_version": ranker.metadata.ranker_version,
            "ranking_basis": ranker.metadata.ranking_basis,
            "feature_version": ranker.metadata.feature_version,
            "feature_names": list(ranker.metadata.feature_names),
            "parameters": ranker.metadata.parameters,
            "seed": seed,
            "early_stopping_rounds": early_stopping_rounds,
            "label_source": evidence.label_source.value,
            "adjudication_complete": evidence.adjudication_complete,
            "contains_synthetic_labels": evidence.contains_synthetic_labels,
            "judgment_manifest_sha256": evidence.judgment_manifest_sha256,
            "human_judgments_file_sha256": evidence.human_judgments_file_sha256,
            "qrels_file_sha256": evidence.qrels_file_sha256,
            "qrels_checksum": evidence.qrels_checksum,
            "qrels_evidence_checksum": evidence.qrels_evidence_checksum,
            "frozen_query_split_file_sha256": evidence.frozen_query_split_file_sha256,
            "frozen_query_split_checksum": split.checksum if split is not None else None,
            "dataset_manifest_file_sha256": evidence.dataset_manifest_file_sha256,
            "dataset_manifest_sha256": evidence.dataset_manifest_sha256,
            "prelabel_snapshot_sha256": evidence.prelabel_snapshot_sha256,
            "feature_contract_sha256": evidence.feature_contract_sha256,
            "annotation_release_sha256": evidence.annotation_release_sha256,
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="query-grouped JSONL")
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--ranker-version", default="ltr-v1")
    parser.add_argument("--training-data-version")
    parser.add_argument("--candidate-set-version", required=True)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--early-stopping-rounds", type=int, default=30)
    parser.add_argument("--parameters", type=_json_object, default={})
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument(
        "--max-host-used-gb",
        type=float,
        default=55.0,
        help="refuse training when conservative projected host RAM reaches this cap",
    )
    parser.add_argument(
        "--minimum-free-memory-mb",
        type=float,
        default=1024.0,
        help="minimum host RAM that must remain after the conservative allocation",
    )
    parser.add_argument(
        "--materialization-memory-expansion-factor",
        type=float,
        default=12.0,
        help="conservative in-memory multiplier for JSON/typed/feature representations",
    )
    parser.add_argument(
        "--materialization-runtime-memory-mb",
        type=float,
        default=512.0,
        help="fixed learner/runtime allowance added to materialized input estimates",
    )
    parser.add_argument(
        "--label-provenance",
        choices=("human", "silver", "synthetic"),
        required=True,
    )
    parser.add_argument(
        "--human-judgments",
        type=Path,
        help="source HumanJudgmentSet JSON proving reviewers and adjudications",
    )
    parser.add_argument(
        "--qrels",
        type=Path,
        help="frozen, checksummed qrels generated from the human judgments",
    )
    parser.add_argument(
        "--frozen-query-split",
        type=Path,
        help="checksummed train/validation/test query-intent split for the qrels",
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        help="materialized labeled-snapshot manifest proving pre-label feature lineage",
    )
    parser.add_argument(
        "--allow-non-human-labels",
        action="store_true",
        help="allow a permanently non-promotable silver/synthetic pipeline diagnostic",
    )
    add_mlflow_arguments(parser, default_experiment="pcbr-learning-to-rank")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.label_provenance != "human" and not args.allow_non_human_labels:
        raise ValueError(
            "non-human relevance labels require --allow-non-human-labels and remain "
            "permanently non-promotable"
        )

    lineage_paths = (
        args.human_judgments,
        args.qrels,
        args.frozen_query_split,
        args.dataset_manifest,
    )
    materialized_input_paths = [args.input]
    if args.label_provenance == "human":
        if any(path is None for path in lineage_paths):
            raise ValueError(
                "human relevance labels require --human-judgments, --qrels, and "
                "--frozen-query-split, and --dataset-manifest"
            )
        assert args.human_judgments is not None
        assert args.qrels is not None
        assert args.frozen_query_split is not None
        assert args.dataset_manifest is not None
        materialized_input_paths.extend(
            (
                args.human_judgments,
                args.qrels,
                args.frozen_query_split,
                args.dataset_manifest,
            )
        )
    estimated_materialization_mib = estimate_materialized_file_memory_mib(
        materialized_input_paths,
        expansion_factor=args.materialization_memory_expansion_factor,
        runtime_allowance_mib=args.materialization_runtime_memory_mb,
    )
    host_memory_preflight = require_host_memory_headroom(
        max_used_gib=args.max_host_used_gb,
        estimated_additional_mib=estimated_materialization_mib,
        minimum_free_mib=args.minimum_free_memory_mb,
    )
    queries = _load_queries(args.input)
    if args.label_provenance == "human":
        evidence = _load_verified_human_evidence(
            ranking_path=args.input,
            dataset_manifest_path=args.dataset_manifest,
            human_judgments_path=args.human_judgments,
            qrels_path=args.qrels,
            frozen_query_split_path=args.frozen_query_split,
            candidate_set_version=args.candidate_set_version,
            queries=queries,
        )
        assert evidence.frozen_query_split is not None
        split = _split_queries_from_frozen(queries, evidence.frozen_query_split)
    else:
        if any(path is not None for path in lineage_paths):
            raise ValueError(
                "human judgments, qrels, frozen query split, and dataset manifest "
                "are valid only with --label-provenance human"
            )
        label_source = RelevanceLabelSource(args.label_provenance)
        evidence = VerifiedRankingEvidence(
            label_source=label_source,
            adjudication_complete=False,
            contains_synthetic_labels=label_source is RelevanceLabelSource.SYNTHETIC,
            judgment_manifest_sha256=None,
            minimum_independent_reviewers=0,
        )
        split = _split_queries(queries, seed=args.seed)
    source_sha256 = sha256_file(args.input)
    training_data_version = args.training_data_version or f"sha256:{source_sha256}"
    parameters = dict(args.parameters)
    parameters["random_state"] = args.seed
    parameters["device_type"] = args.device
    ranker = LambdaMARTRanker(
        parameters=parameters,
        ranker_version=args.ranker_version,
    ).fit(
        split["train"],
        validation_queries=split["validation"],
        training_data_version=training_data_version,
        candidate_set_version=args.candidate_set_version,
        early_stopping_rounds=args.early_stopping_rounds,
        training_label_source=evidence.label_source,
        training_adjudication_complete=evidence.adjudication_complete,
        contains_synthetic_labels=evidence.contains_synthetic_labels,
        training_judgment_manifest_sha256=evidence.judgment_manifest_sha256,
        training_dataset_manifest_sha256=evidence.dataset_manifest_sha256,
        training_prelabel_snapshot_sha256=evidence.prelabel_snapshot_sha256,
        training_feature_contract_sha256=evidence.feature_contract_sha256,
        frozen_query_split=evidence.frozen_query_split,
    )
    publication_intent_sha256 = _training_publication_intent_sha256(
        source_sha256=source_sha256,
        training_data_version=training_data_version,
        candidate_set_version=args.candidate_set_version,
        seed=args.seed,
        early_stopping_rounds=args.early_stopping_rounds,
        ranker=ranker,
        evidence=evidence,
    )
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path, metadata_path = ranker.publish_bundle(
        args.artifact_dir / "ranker-artifact",
        publication_intent_sha256=publication_intent_sha256,
    )
    # Publication can adopt an already-committed same-intent artifact after a
    # process crash or concurrent run.  Measure only after publication so the
    # report always describes the exact booster whose hashes it records.
    learned_ndcg, bm25_ndcg = _mean_ndcg(ranker, split["test"])
    improvement = relative_ndcg_improvement(learned_ndcg, bm25_ndcg) if bm25_ndcg > 0 else None
    row_count = sum(len(query.candidates) for query in queries)
    blockers = _promotion_blockers(
        ranker_metadata=ranker.metadata,
        minimum_independent_reviewers=evidence.minimum_independent_reviewers,
        query_count=len(queries),
        row_count=row_count,
        relative_improvement=improvement,
    )
    training_evidence_payload: dict[str, Any] = {
        "schema_version": "pc-build-recommender.ranking-training-evidence.v3",
        "source_sha256": source_sha256,
        "publication_intent_sha256": publication_intent_sha256,
        "training_data_version": training_data_version,
        "candidate_set_version": args.candidate_set_version,
        "leakage_unit": (
            "query_group_id" if evidence.frozen_query_split is not None else "query_id"
        ),
        "requested_label_provenance": args.label_provenance,
        "verified_label_source": evidence.label_source.value,
        "adjudication_complete": evidence.adjudication_complete,
        "contains_synthetic_labels": evidence.contains_synthetic_labels,
        "minimum_independent_reviewers_per_pair": (evidence.minimum_independent_reviewers),
        "judgment_manifest_sha256": evidence.judgment_manifest_sha256,
        "qrels_manifest_sha256": evidence.qrels_file_sha256,
        "dataset_manifest": (
            {
                "path": str(evidence.dataset_manifest_path),
                "file_sha256": evidence.dataset_manifest_file_sha256,
                "manifest_sha256": evidence.dataset_manifest_sha256,
                "prelabel_snapshot_sha256": evidence.prelabel_snapshot_sha256,
                "feature_contract_sha256": evidence.feature_contract_sha256,
                "annotation_release_sha256": evidence.annotation_release_sha256,
            }
            if evidence.dataset_manifest_path is not None
            else None
        ),
        "human_judgments": (
            {
                "path": str(evidence.human_judgments_path),
                "file_sha256": evidence.human_judgments_file_sha256,
                "content_sha256": evidence.judgment_manifest_sha256,
            }
            if evidence.human_judgments_path is not None
            else None
        ),
        "qrels": (
            {
                "path": str(evidence.qrels_path),
                "file_sha256": evidence.qrels_file_sha256,
                "candidate_checksum": evidence.qrels_checksum,
                "evidence_checksum": evidence.qrels_evidence_checksum,
            }
            if evidence.qrels_path is not None
            else None
        ),
        "frozen_query_split": (
            {
                "path": str(evidence.frozen_query_split_path),
                "file_sha256": evidence.frozen_query_split_file_sha256,
                "checksum": evidence.frozen_query_split.checksum,
                "version": evidence.frozen_query_split.version,
                "dataset_checksum": evidence.frozen_query_split.dataset_checksum,
                "dataset_evidence_checksum": (
                    evidence.frozen_query_split.dataset_evidence_checksum
                ),
                "seed": evidence.frozen_query_split.seed,
                "weights": dict(evidence.frozen_query_split.weights),
            }
            if evidence.frozen_query_split is not None
            else None
        ),
        "training_query_hashes": sorted(
            sha256_text(query.context.query_id) for query in split["train"]
        ),
        "validation_query_hashes": sorted(
            sha256_text(query.context.query_id) for query in split["validation"]
        ),
        "test_query_hashes": sorted(sha256_text(query.context.query_id) for query in split["test"]),
        "ranker_metadata": ranker.metadata.to_dict(),
    }
    write_json(args.artifact_dir / "training_evidence.json", training_evidence_payload)
    report: dict[str, Any] = {
        "created_at": utc_now_iso(),
        "task": "component_learning_to_rank",
        "input": {
            "path": str(args.input.resolve()),
            "sha256": source_sha256,
            "training_data_version": training_data_version,
            "candidate_set_version": args.candidate_set_version,
            "queries": len(queries),
            "graded_rows": row_count,
            "requested_label_provenance": args.label_provenance,
            "verified_label_source": evidence.label_source.value,
            "adjudication_complete": evidence.adjudication_complete,
            "minimum_independent_reviewers_per_pair": (evidence.minimum_independent_reviewers),
            "judgment_manifest_sha256": evidence.judgment_manifest_sha256,
            "qrels_file_sha256": evidence.qrels_file_sha256,
            "qrels_manifest_sha256": evidence.qrels_file_sha256,
            "dataset_manifest_sha256": evidence.dataset_manifest_sha256,
            "prelabel_snapshot_sha256": evidence.prelabel_snapshot_sha256,
            "feature_contract_sha256": evidence.feature_contract_sha256,
            "annotation_release_sha256": evidence.annotation_release_sha256,
            "query_split_checksum": (
                evidence.frozen_query_split.checksum
                if evidence.frozen_query_split is not None
                else None
            ),
        },
        "split": {
            name: {
                "queries": len(rows),
                "graded_rows": sum(len(query.candidates) for query in rows),
            }
            for name, rows in split.items()
        },
        "artifact": {
            "model_path": str(model_path.resolve()),
            "metadata_path": str(metadata_path.resolve()),
            "publication_intent_sha256": publication_intent_sha256,
        },
        "ranker": ranker.metadata.to_dict(),
        "device": {"requested": args.device, "actual": args.device, "fallback_reason": None},
        "resources": {
            "materialized_input_file_bytes": sum(
                path.stat().st_size for path in materialized_input_paths
            ),
            "materialization_memory_expansion_factor": args.materialization_memory_expansion_factor,
            "materialization_runtime_memory_mb": args.materialization_runtime_memory_mb,
            "estimated_materialization_memory_mb": estimated_materialization_mib,
            "host_memory_preflight": host_memory_preflight.to_dict(),
        },
        "heldout_test": {
            "ndcg_at_10": learned_ndcg,
            "bm25_ndcg_at_10": bm25_ndcg,
            "relative_improvement_percent": improvement,
        },
        "promotion": {"eligible": not blockers, "block_reasons": blockers},
    }
    report_path = args.report or args.artifact_dir / "training_report.json"
    write_json(report_path, report)

    with OptionalMLflowRun(tracking_config_from_args(args)) as tracking:
        tracking.log_params(
            {
                "task": report["task"],
                "dataset": report["input"],
                "split": report["split"],
                "ranker_version": ranker.metadata.ranker_version,
                "feature_version": ranker.metadata.feature_version,
                "model_type": ranker.metadata.model_type,
                "parameters": ranker.metadata.parameters,
                "device": report["device"],
            }
        )
        tracking.log_metrics(
            {
                "test.ndcg_at_10": learned_ndcg,
                "test.bm25_ndcg_at_10": bm25_ndcg,
                "test.relative_improvement_percent": improvement,
                **ranker.metadata.metrics,
            }
        )
        tracking.log_tags(
            {
                "task": report["task"],
                "model.version": ranker.metadata.ranker_version,
                "feature.version": ranker.metadata.feature_version,
                "label.provenance": ranker.metadata.training_label_source,
                "device.requested": args.device,
                "device.actual": args.device,
                **promotion_blocker_tags(blockers),
            }
        )
        tracking.log_dict(report["promotion"], "evidence/promotion-gate.json")
        tracking.log_dict(training_evidence_payload, "evidence/training-evidence.json")
        report["mlflow_tracking"] = tracking.describe()
        write_json(report_path, report)
        tracking.log_dict(report, "evidence/training-report.json")
        # Log only the committed bundle. Crash-orphaned hidden staging
        # directories are deliberately outside both serving and MLflow evidence.
        tracking.log_native_artifacts(model_path.parent, artifact_path="ranking-model")

    print_json(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
