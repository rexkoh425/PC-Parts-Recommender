"""Write-once governance for the frozen ranking test cohort.

Training may bind a frozen split, but it must not repeatedly inspect test metrics.
This module makes the final test evaluation an explicitly preregistered operation:

* one immutable intent owns one frozen test cohort;
* every first access is recorded before labels are parsed or scores are produced;
* exact retries are idempotent, while a different intent for the cohort fails closed;
* evaluation outputs are published as one content-bound, no-replace directory.

The ledger is a trusted local filesystem control, not a cryptographic signature.  Its
unkeyed hashes detect accidental or out-of-band drift under a governed artifact root.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pc_build_recommender.evaluation.manifest import sha256_json
from pc_build_recommender.ranking import (
    MINIMUM_PRODUCTION_BOOTSTRAP_RESAMPLES,
    LambdaMARTRanker,
    RankerPromotionPolicy,
    assert_production_ranker_promotion_policy,
    production_evaluation_payload,
    production_ledger_identity_payload,
    verify_ranking_evaluation_bundle,
)
from pc_build_recommender.ranking.evaluation_bundle import (
    RANKING_EVALUATION_BUNDLE_SCHEMA_VERSION,
    RANKING_LEDGER_ACCESS_SCHEMA_VERSION,
    RANKING_LEDGER_COMPLETION_SCHEMA_VERSION,
    RANKING_LEDGER_REGISTRATION_SCHEMA_VERSION,
)
from pc_build_recommender.retrieval import (
    FrozenCandidateSet,
    FrozenQueryGroupSplit,
    load_human_judgment_set,
)
from training._common import sha256_file
from training.materialize_ranking_snapshot import (
    VerifiedLabeledRankingSnapshot,
    verify_labeled_ranking_snapshot,
)

INTENT_SCHEMA_VERSION = "pc-build-recommender.ranking-evaluation-intent.v1"
LEDGER_REGISTRATION_SCHEMA_VERSION = RANKING_LEDGER_REGISTRATION_SCHEMA_VERSION
LEDGER_ACCESS_SCHEMA_VERSION = RANKING_LEDGER_ACCESS_SCHEMA_VERSION
LEDGER_COMPLETION_SCHEMA_VERSION = RANKING_LEDGER_COMPLETION_SCHEMA_VERSION
EVALUATION_BUNDLE_SCHEMA_VERSION = RANKING_EVALUATION_BUNDLE_SCHEMA_VERSION
MAX_BOOTSTRAP_RESAMPLES = 10_000
EVALUATOR_MODULE = "training.evaluate_ranking"
DEPENDENCY_LOCK_NAME = "uv.lock"


class RankingEvaluationGateError(ValueError):
    """Raised when a frozen-test governance invariant cannot be proven."""


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _governed_runtime_paths() -> tuple[Path, Path]:
    repository_root = Path(__file__).resolve().parents[1]
    evaluator_source = Path(__file__).resolve().with_name("evaluate_ranking.py")
    dependency_lock = repository_root / DEPENDENCY_LOCK_NAME
    if not evaluator_source.is_file():
        raise RankingEvaluationGateError(
            f"governed evaluator source is unavailable: {evaluator_source}"
        )
    if not dependency_lock.is_file():
        raise RankingEvaluationGateError(
            f"governed dependency lock is unavailable: {dependency_lock}"
        )
    return evaluator_source, dependency_lock


def _object(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RankingEvaluationGateError(f"{name} must be an object")
    return {str(key): item for key, item in value.items()}


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise RankingEvaluationGateError(f"{name} must be a non-empty string")
    return value


def _digest(value: object, *, name: str) -> str:
    digest = _string(value, name=name)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise RankingEvaluationGateError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(dict(payload), allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _load_json_object(path: Path, *, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RankingEvaluationGateError(f"{name} is not valid JSON") from error
    return _object(payload, name=name)


def _with_self_hash(payload: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    result = dict(payload)
    result[field] = sha256_json(result)
    return result


def _verify_self_hash(payload: Mapping[str, Any], *, field: str, name: str) -> str:
    stored = _digest(payload.get(field), name=f"{name} {field}")
    unhashed = dict(payload)
    unhashed.pop(field, None)
    if sha256_json(unhashed) != stored:
        raise RankingEvaluationGateError(f"{name} self-hash mismatch")
    return stored


def _write_once_json(path: Path, payload: Mapping[str, Any]) -> Path:
    """Create one immutable JSON file; exact retries adopt the existing bytes."""

    expected = _json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if not path.is_file() or path.read_bytes() != expected:
            raise RankingEvaluationGateError(
                f"write-once artifact already exists with different bytes: {path}"
            ) from None
        return path
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(expected)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        # A partial exclusive file is deliberately retained.  Removing it would turn
        # an ambiguous audit event into a silent retry.
        raise
    return path


@dataclass(frozen=True, slots=True)
class VerifiedHumanRankingLineage:
    """Independently reconstructed human qrels and split evidence."""

    snapshot: VerifiedLabeledRankingSnapshot
    qrels: FrozenCandidateSet
    query_split: FrozenQueryGroupSplit
    minimum_independent_reviewers: int
    ranking_file_sha256: str
    human_judgments_file_sha256: str
    qrels_file_sha256: str
    query_split_file_sha256: str


@dataclass(frozen=True, slots=True)
class UnopenedRankingTestCohort:
    """Raw file and manifest bindings created without parsing test judgments."""

    snapshot_manifest_sha256: str
    snapshot_manifest_file_sha256: str
    ranking_file_sha256: str
    human_judgments_file_sha256: str
    qrels_file_sha256: str
    qrels_version: str
    qrels_checksum: str
    qrels_evidence_checksum: str
    judgment_manifest_sha256: str
    query_split_file_sha256: str
    query_split_checksum: str
    prelabel_snapshot_sha256: str
    feature_contract_sha256: str


def bind_unopened_ranking_test_cohort(
    *,
    ranking_path: Path,
    manifest_path: Path,
    human_judgments_path: Path,
    qrels_path: Path,
    query_split_path: Path,
    ranker: LambdaMARTRanker,
) -> UnopenedRankingTestCohort:
    """Bind raw test files for preregistration without loading labels or qrels rows."""

    manifest = _load_json_object(manifest_path, name="labeled ranking snapshot manifest")
    if manifest.get("schema_version") != (
        "pc-build-recommender.ranking-labeled-snapshot-manifest.v1"
    ):
        raise RankingEvaluationGateError("unsupported labeled ranking snapshot manifest")
    manifest_sha256 = _verify_self_hash(
        manifest,
        field="manifest_sha256",
        name="labeled ranking snapshot manifest",
    )
    files = _object(manifest.get("files"), name="labeled snapshot files")
    if set(files) != {"ranking.jsonl"}:
        raise RankingEvaluationGateError(
            "labeled snapshot must bind exactly one ranking JSONL file"
        )
    ranking_evidence = _object(files["ranking.jsonl"], name="ranking file evidence")
    ranking_file_sha256 = sha256_file(ranking_path)
    if (
        ranking_evidence.get("size_bytes") != ranking_path.stat().st_size
        or ranking_evidence.get("sha256") != ranking_file_sha256
    ):
        raise RankingEvaluationGateError("ranking file does not match its snapshot manifest")

    annotation = _object(manifest.get("annotation_release"), name="annotation release")
    annotation_files = _object(annotation.get("files"), name="annotation release files")
    supplied = {
        "human-judgments.json": human_judgments_path,
        "qrels.json": qrels_path,
        "query-split.json": query_split_path,
    }
    file_hashes: dict[str, str] = {}
    for name, path in supplied.items():
        evidence = _object(annotation_files.get(name), name=f"annotation file {name}")
        actual_hash = sha256_file(path)
        if (
            evidence.get("size_bytes") != path.stat().st_size
            or evidence.get("sha256") != actual_hash
        ):
            raise RankingEvaluationGateError(
                f"annotation file does not match the labeled snapshot manifest: {name}"
            )
        file_hashes[name] = actual_hash

    qrels_manifest = _object(manifest.get("qrels"), name="qrels binding")
    split_manifest = _object(manifest.get("query_split"), name="query split binding")
    prelabel = _object(manifest.get("prelabel"), name="pre-label binding")
    binding = UnopenedRankingTestCohort(
        snapshot_manifest_sha256=manifest_sha256,
        snapshot_manifest_file_sha256=sha256_file(manifest_path),
        ranking_file_sha256=ranking_file_sha256,
        human_judgments_file_sha256=file_hashes["human-judgments.json"],
        qrels_file_sha256=file_hashes["qrels.json"],
        qrels_version=_string(qrels_manifest.get("version"), name="qrels version"),
        qrels_checksum=_digest(qrels_manifest.get("checksum"), name="qrels checksum"),
        qrels_evidence_checksum=_digest(
            qrels_manifest.get("evidence_checksum"),
            name="qrels evidence checksum",
        ),
        judgment_manifest_sha256=_digest(
            qrels_manifest.get("judgment_manifest_sha256"),
            name="judgment manifest sha256",
        ),
        query_split_file_sha256=file_hashes["query-split.json"],
        query_split_checksum=_digest(
            split_manifest.get("checksum"),
            name="query split checksum",
        ),
        prelabel_snapshot_sha256=_digest(
            prelabel.get("snapshot_sha256"),
            name="pre-label snapshot sha256",
        ),
        feature_contract_sha256=_digest(
            prelabel.get("feature_contract_sha256"),
            name="feature contract sha256",
        ),
    )
    metadata = ranker.metadata
    if metadata.candidate_set_version != binding.qrels_version:
        raise RankingEvaluationGateError("ranker was trained against a different candidate set")
    if metadata.query_group_split_checksum != binding.query_split_checksum:
        raise RankingEvaluationGateError("ranker was trained against a different frozen split")
    if metadata.training_judgment_manifest_sha256 != binding.judgment_manifest_sha256:
        raise RankingEvaluationGateError("ranker was trained against different human judgments")
    if (
        metadata.training_dataset_manifest_sha256 != binding.snapshot_manifest_sha256
        or metadata.training_prelabel_snapshot_sha256 != binding.prelabel_snapshot_sha256
        or metadata.training_feature_contract_sha256 != binding.feature_contract_sha256
    ):
        raise RankingEvaluationGateError(
            "ranker was trained against a different pre-label feature snapshot"
        )
    return binding


def verify_human_ranking_lineage(
    *,
    ranking_path: Path,
    manifest_path: Path,
    human_judgments_path: Path,
    qrels_path: Path,
    query_split_path: Path,
) -> VerifiedHumanRankingLineage:
    """Re-adjudicate humans and prove the supplied qrels/split are derived from them."""

    snapshot = verify_labeled_ranking_snapshot(
        ranking_path=ranking_path,
        manifest_path=manifest_path,
        human_judgments_path=human_judgments_path,
        qrels_path=qrels_path,
        query_split_path=query_split_path,
    )
    human = load_human_judgment_set(human_judgments_path)
    reviewers: dict[tuple[str, str], set[str]] = defaultdict(set)
    for judgment in human.judgments:
        reviewers[(judgment.query_id, judgment.product_id)].add(judgment.reviewer_id)
    expected_pairs = {
        (query.query_id, product_id)
        for query in human.queries
        for product_id in query.candidate_ids
    }
    minimum_reviewers = min(
        (len(reviewers.get(pair, set())) for pair in expected_pairs),
        default=0,
    )
    if set(reviewers) != expected_pairs or minimum_reviewers < 2:
        raise RankingEvaluationGateError(
            "every relevance pair requires two independent human reviewers"
        )

    adjudicated = human.adjudicate()
    expected_qrels = adjudicated.frozen_candidates
    qrels = FrozenCandidateSet.load(qrels_path)
    if (
        qrels.version != expected_qrels.version
        or qrels.checksum != expected_qrels.checksum
        or qrels.evidence_checksum != expected_qrels.evidence_checksum
        or qrels.judgment_manifest_sha256 != human.content_sha256
    ):
        raise RankingEvaluationGateError(
            "qrels do not match independently re-adjudicated human judgments"
        )
    if not qrels.eligible_for_promotion:
        raise RankingEvaluationGateError(
            "sealed ranking evaluation requires adjudicated, non-synthetic human qrels"
        )

    query_split = FrozenQueryGroupSplit.load(query_split_path)
    query_split.validate_dataset(qrels)
    if set(query_split.weights) != {"train", "validation", "test"}:
        raise RankingEvaluationGateError(
            "frozen query split must contain exactly train, validation, and test"
        )
    expected_groups = {query.query_id: query.query_group_id for query in qrels.queries}
    if any(group_id is None for group_id in expected_groups.values()):
        raise RankingEvaluationGateError("every human qrels query needs a query-group ID")
    if dict(query_split.query_group_ids) != expected_groups:
        raise RankingEvaluationGateError("frozen split query groups do not match human qrels")

    return VerifiedHumanRankingLineage(
        snapshot=snapshot,
        qrels=qrels,
        query_split=query_split,
        minimum_independent_reviewers=minimum_reviewers,
        ranking_file_sha256=sha256_file(ranking_path),
        human_judgments_file_sha256=sha256_file(human_judgments_path),
        qrels_file_sha256=sha256_file(qrels_path),
        query_split_file_sha256=sha256_file(query_split_path),
    )


def assert_ranker_matches_lineage(
    ranker: LambdaMARTRanker,
    lineage: VerifiedHumanRankingLineage,
) -> None:
    """Reject a model trained against any other labels, features, candidates, or split."""

    metadata = ranker.metadata
    snapshot = lineage.snapshot
    if metadata.candidate_set_version != lineage.qrels.version:
        raise RankingEvaluationGateError("ranker was trained against a different candidate set")
    if metadata.query_group_split_checksum != lineage.query_split.checksum:
        raise RankingEvaluationGateError("ranker was trained against a different frozen split")
    if metadata.training_judgment_manifest_sha256 != lineage.qrels.judgment_manifest_sha256:
        raise RankingEvaluationGateError("ranker was trained against different human judgments")
    if (
        metadata.training_dataset_manifest_sha256 != snapshot.manifest_sha256
        or metadata.training_prelabel_snapshot_sha256 != snapshot.prelabel_snapshot_sha256
        or metadata.training_feature_contract_sha256 != snapshot.feature_contract_sha256
    ):
        raise RankingEvaluationGateError(
            "ranker was trained against a different pre-label feature snapshot"
        )


@dataclass(frozen=True, slots=True)
class RankingEvaluationIntent:
    """Deterministic preregistration for one model on one frozen test cohort."""

    cohort_sha256: str
    snapshot_manifest_sha256: str
    snapshot_manifest_file_sha256: str
    ranking_file_sha256: str
    human_judgments_file_sha256: str
    qrels_file_sha256: str
    qrels_checksum: str
    qrels_evidence_checksum: str
    query_split_file_sha256: str
    query_split_checksum: str
    ranker_model_sha256: str
    ranker_metadata_sha256: str
    ranker_manifest_sha256: str
    evaluator_source_sha256: str
    dependency_lock_sha256: str
    challenger_model: str
    n_resamples: int
    bootstrap_seed: int
    policy: RankerPromotionPolicy
    intent_sha256: str

    def cohort_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pc-build-recommender.ranking-test-cohort.v1",
            "snapshot_manifest_sha256": self.snapshot_manifest_sha256,
            "snapshot_manifest_file_sha256": self.snapshot_manifest_file_sha256,
            "ranking_file_sha256": self.ranking_file_sha256,
            "human_judgments_file_sha256": self.human_judgments_file_sha256,
            "qrels_file_sha256": self.qrels_file_sha256,
            "qrels_checksum": self.qrels_checksum,
            "qrels_evidence_checksum": self.qrels_evidence_checksum,
            "query_split_file_sha256": self.query_split_file_sha256,
            "query_split_checksum": self.query_split_checksum,
            "split_name": "test",
        }

    def content_payload(self) -> dict[str, object]:
        return {
            "schema_version": INTENT_SCHEMA_VERSION,
            "cohort_sha256": self.cohort_sha256,
            "cohort": self.cohort_payload(),
            "ranker_artifact": {
                "model_sha256": self.ranker_model_sha256,
                "metadata_sha256": self.ranker_metadata_sha256,
                "manifest_sha256": self.ranker_manifest_sha256,
            },
            "evaluation_runtime": {
                "evaluator_module": EVALUATOR_MODULE,
                "evaluator_source_sha256": self.evaluator_source_sha256,
                "dependency_lock_name": DEPENDENCY_LOCK_NAME,
                "dependency_lock_sha256": self.dependency_lock_sha256,
            },
            "challenger_model": self.challenger_model,
            "evaluation_parameters": {
                "n_resamples": self.n_resamples,
                "bootstrap_seed": self.bootstrap_seed,
                "promotion_policy": self.policy.to_dict(),
            },
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.content_payload(), "intent_sha256": self.intent_sha256}

    def __post_init__(self) -> None:
        for name, digest in (
            ("cohort_sha256", self.cohort_sha256),
            ("snapshot_manifest_sha256", self.snapshot_manifest_sha256),
            ("snapshot_manifest_file_sha256", self.snapshot_manifest_file_sha256),
            ("ranking_file_sha256", self.ranking_file_sha256),
            ("human_judgments_file_sha256", self.human_judgments_file_sha256),
            ("qrels_file_sha256", self.qrels_file_sha256),
            ("qrels_checksum", self.qrels_checksum),
            ("qrels_evidence_checksum", self.qrels_evidence_checksum),
            ("query_split_file_sha256", self.query_split_file_sha256),
            ("query_split_checksum", self.query_split_checksum),
            ("ranker_model_sha256", self.ranker_model_sha256),
            ("ranker_metadata_sha256", self.ranker_metadata_sha256),
            ("ranker_manifest_sha256", self.ranker_manifest_sha256),
            ("evaluator_source_sha256", self.evaluator_source_sha256),
            ("dependency_lock_sha256", self.dependency_lock_sha256),
            ("intent_sha256", self.intent_sha256),
        ):
            _digest(digest, name=name)
        if not self.challenger_model:
            raise RankingEvaluationGateError("challenger_model must not be empty")
        if (
            type(self.n_resamples) is not int
            or not MINIMUM_PRODUCTION_BOOTSTRAP_RESAMPLES
            <= self.n_resamples
            <= MAX_BOOTSTRAP_RESAMPLES
        ):
            raise RankingEvaluationGateError(
                "production n_resamples must be between "
                f"{MINIMUM_PRODUCTION_BOOTSTRAP_RESAMPLES} and "
                f"{MAX_BOOTSTRAP_RESAMPLES}"
            )
        if type(self.bootstrap_seed) is not int:
            raise RankingEvaluationGateError("bootstrap_seed must be an integer")
        if self.challenger_model in {"bm25", "rrf_hybrid"}:
            raise RankingEvaluationGateError(
                "challenger_model must not shadow a frozen reference baseline"
            )
        try:
            assert_production_ranker_promotion_policy(self.policy)
        except ValueError as error:
            raise RankingEvaluationGateError(str(error)) from error
        if sha256_json(self.cohort_payload()) != self.cohort_sha256:
            raise RankingEvaluationGateError("evaluation cohort hash mismatch")
        if sha256_json(self.content_payload()) != self.intent_sha256:
            raise RankingEvaluationGateError("evaluation intent self-hash mismatch")

    @classmethod
    def create(
        cls,
        *,
        cohort: UnopenedRankingTestCohort,
        ranker: LambdaMARTRanker,
        challenger_model: str,
        n_resamples: int,
        bootstrap_seed: int,
        policy: RankerPromotionPolicy,
    ) -> RankingEvaluationIntent:
        identity = ranker.artifact_identity
        evaluator_source, dependency_lock = _governed_runtime_paths()
        fields: dict[str, Any] = {
            "snapshot_manifest_sha256": cohort.snapshot_manifest_sha256,
            "snapshot_manifest_file_sha256": cohort.snapshot_manifest_file_sha256,
            "ranking_file_sha256": cohort.ranking_file_sha256,
            "human_judgments_file_sha256": cohort.human_judgments_file_sha256,
            "qrels_file_sha256": cohort.qrels_file_sha256,
            "qrels_checksum": cohort.qrels_checksum,
            "qrels_evidence_checksum": cohort.qrels_evidence_checksum,
            "query_split_file_sha256": cohort.query_split_file_sha256,
            "query_split_checksum": cohort.query_split_checksum,
            "ranker_model_sha256": identity.model_sha256,
            "ranker_metadata_sha256": identity.metadata_sha256,
            "ranker_manifest_sha256": identity.manifest_sha256,
            "evaluator_source_sha256": sha256_file(evaluator_source),
            "dependency_lock_sha256": sha256_file(dependency_lock),
            "challenger_model": challenger_model,
            "n_resamples": n_resamples,
            "bootstrap_seed": bootstrap_seed,
            "policy": policy,
        }
        cohort_payload = {
            "schema_version": "pc-build-recommender.ranking-test-cohort.v1",
            "snapshot_manifest_sha256": fields["snapshot_manifest_sha256"],
            "snapshot_manifest_file_sha256": fields["snapshot_manifest_file_sha256"],
            "ranking_file_sha256": fields["ranking_file_sha256"],
            "human_judgments_file_sha256": fields["human_judgments_file_sha256"],
            "qrels_file_sha256": fields["qrels_file_sha256"],
            "qrels_checksum": fields["qrels_checksum"],
            "qrels_evidence_checksum": fields["qrels_evidence_checksum"],
            "query_split_file_sha256": fields["query_split_file_sha256"],
            "query_split_checksum": fields["query_split_checksum"],
            "split_name": "test",
        }
        cohort_sha256 = sha256_json(cohort_payload)
        content_payload = {
            "schema_version": INTENT_SCHEMA_VERSION,
            "cohort_sha256": cohort_sha256,
            "cohort": cohort_payload,
            "ranker_artifact": {
                "model_sha256": fields["ranker_model_sha256"],
                "metadata_sha256": fields["ranker_metadata_sha256"],
                "manifest_sha256": fields["ranker_manifest_sha256"],
            },
            "evaluation_runtime": {
                "evaluator_module": EVALUATOR_MODULE,
                "evaluator_source_sha256": fields["evaluator_source_sha256"],
                "dependency_lock_name": DEPENDENCY_LOCK_NAME,
                "dependency_lock_sha256": fields["dependency_lock_sha256"],
            },
            "challenger_model": challenger_model,
            "evaluation_parameters": {
                "n_resamples": n_resamples,
                "bootstrap_seed": bootstrap_seed,
                "promotion_policy": policy.to_dict(),
            },
        }
        return cls(
            cohort_sha256=cohort_sha256,
            intent_sha256=sha256_json(content_payload),
            **fields,
        )

    @classmethod
    def load(cls, path: Path) -> RankingEvaluationIntent:
        payload = _load_json_object(path, name="ranking evaluation intent")
        if payload.get("schema_version") != INTENT_SCHEMA_VERSION:
            raise RankingEvaluationGateError("unsupported ranking evaluation intent schema")
        cohort = _object(payload.get("cohort"), name="evaluation cohort")
        artifact = _object(payload.get("ranker_artifact"), name="ranker artifact")
        runtime = _object(payload.get("evaluation_runtime"), name="evaluation runtime")
        if (
            runtime.get("evaluator_module") != EVALUATOR_MODULE
            or runtime.get("dependency_lock_name") != DEPENDENCY_LOCK_NAME
        ):
            raise RankingEvaluationGateError(
                "evaluation runtime does not name the governed evaluator and dependency lock"
            )
        parameters = _object(
            payload.get("evaluation_parameters"),
            name="evaluation parameters",
        )
        raw_resamples = parameters.get("n_resamples")
        raw_seed = parameters.get("bootstrap_seed")
        if type(raw_resamples) is not int or type(raw_seed) is not int:
            raise RankingEvaluationGateError(
                "evaluation resamples and bootstrap seed must be integers"
            )
        policy = RankerPromotionPolicy.from_dict(
            _object(parameters.get("promotion_policy"), name="promotion policy")
        )
        return cls(
            cohort_sha256=_digest(payload.get("cohort_sha256"), name="cohort_sha256"),
            snapshot_manifest_sha256=_digest(
                cohort.get("snapshot_manifest_sha256"),
                name="snapshot_manifest_sha256",
            ),
            snapshot_manifest_file_sha256=_digest(
                cohort.get("snapshot_manifest_file_sha256"),
                name="snapshot_manifest_file_sha256",
            ),
            ranking_file_sha256=_digest(
                cohort.get("ranking_file_sha256"), name="ranking_file_sha256"
            ),
            human_judgments_file_sha256=_digest(
                cohort.get("human_judgments_file_sha256"),
                name="human_judgments_file_sha256",
            ),
            qrels_file_sha256=_digest(
                cohort.get("qrels_file_sha256"), name="qrels_file_sha256"
            ),
            qrels_checksum=_digest(cohort.get("qrels_checksum"), name="qrels_checksum"),
            qrels_evidence_checksum=_digest(
                cohort.get("qrels_evidence_checksum"), name="qrels_evidence_checksum"
            ),
            query_split_file_sha256=_digest(
                cohort.get("query_split_file_sha256"),
                name="query_split_file_sha256",
            ),
            query_split_checksum=_digest(
                cohort.get("query_split_checksum"), name="query_split_checksum"
            ),
            ranker_model_sha256=_digest(
                artifact.get("model_sha256"), name="ranker model_sha256"
            ),
            ranker_metadata_sha256=_digest(
                artifact.get("metadata_sha256"), name="ranker metadata_sha256"
            ),
            ranker_manifest_sha256=_digest(
                artifact.get("manifest_sha256"), name="ranker manifest_sha256"
            ),
            evaluator_source_sha256=_digest(
                runtime.get("evaluator_source_sha256"),
                name="evaluator source sha256",
            ),
            dependency_lock_sha256=_digest(
                runtime.get("dependency_lock_sha256"),
                name="dependency lock sha256",
            ),
            challenger_model=_string(
                payload.get("challenger_model"), name="challenger_model"
            ),
            n_resamples=raw_resamples,
            bootstrap_seed=raw_seed,
            policy=policy,
            intent_sha256=_digest(payload.get("intent_sha256"), name="intent_sha256"),
        )

    def assert_runtime_bindings(self) -> None:
        evaluator_source, dependency_lock = _governed_runtime_paths()
        if sha256_file(evaluator_source) != self.evaluator_source_sha256:
            raise RankingEvaluationGateError(
                "evaluator source does not match the preregistered intent"
            )
        if sha256_file(dependency_lock) != self.dependency_lock_sha256:
            raise RankingEvaluationGateError(
                "dependency lock does not match the preregistered intent"
            )

    def assert_file_bindings(
        self,
        *,
        ranking_path: Path,
        manifest_path: Path,
        human_judgments_path: Path,
        qrels_path: Path,
        query_split_path: Path,
        ranker: LambdaMARTRanker,
    ) -> None:
        actual = {
            "ranking_file_sha256": sha256_file(ranking_path),
            "snapshot_manifest_file_sha256": sha256_file(manifest_path),
            "human_judgments_file_sha256": sha256_file(human_judgments_path),
            "qrels_file_sha256": sha256_file(qrels_path),
            "query_split_file_sha256": sha256_file(query_split_path),
        }
        expected = {
            "ranking_file_sha256": self.ranking_file_sha256,
            "snapshot_manifest_file_sha256": self.snapshot_manifest_file_sha256,
            "human_judgments_file_sha256": self.human_judgments_file_sha256,
            "qrels_file_sha256": self.qrels_file_sha256,
            "query_split_file_sha256": self.query_split_file_sha256,
        }
        if actual != expected:
            raise RankingEvaluationGateError(
                "evaluation inputs do not match the preregistered file hashes"
            )
        identity = ranker.artifact_identity
        if (
            identity.model_sha256 != self.ranker_model_sha256
            or identity.metadata_sha256 != self.ranker_metadata_sha256
            or identity.manifest_sha256 != self.ranker_manifest_sha256
        ):
            raise RankingEvaluationGateError(
                "ranker artifact does not match the preregistered intent"
            )

    def assert_lineage(self, lineage: VerifiedHumanRankingLineage) -> None:
        if (
            lineage.snapshot.manifest_sha256 != self.snapshot_manifest_sha256
            or lineage.qrels.checksum != self.qrels_checksum
            or lineage.qrels.evidence_checksum != self.qrels_evidence_checksum
            or lineage.query_split.checksum != self.query_split_checksum
        ):
            raise RankingEvaluationGateError(
                "verified human lineage does not match the preregistered cohort"
            )


class RankingEvaluationLedger:
    """Append-only filesystem records for one-shot ranking test access."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def _registration_path(self, cohort_sha256: str) -> Path:
        return self.root / "registrations" / f"{cohort_sha256}.json"

    def _access_path(self, intent_sha256: str) -> Path:
        return self.root / "accesses" / f"{intent_sha256}.json"

    def _completion_path(self, intent_sha256: str) -> Path:
        return self.root / "completions" / f"{intent_sha256}.json"

    @staticmethod
    def _load_event(path: Path, *, schema: str, name: str) -> dict[str, Any]:
        payload = _load_json_object(path, name=name)
        if payload.get("schema_version") != schema:
            raise RankingEvaluationGateError(f"{name} has an unsupported schema")
        _verify_self_hash(payload, field="event_sha256", name=name)
        return payload

    def preregister(self, intent: RankingEvaluationIntent) -> Path:
        path = self._registration_path(intent.cohort_sha256)
        if path.exists():
            existing = self._load_event(
                path,
                schema=LEDGER_REGISTRATION_SCHEMA_VERSION,
                name="ranking test registration",
            )
            if existing.get("intent_sha256") != intent.intent_sha256:
                raise RankingEvaluationGateError(
                    "frozen test cohort is already registered to a different intent"
                )
            return path
        event = _with_self_hash(
            {
                "schema_version": LEDGER_REGISTRATION_SCHEMA_VERSION,
                "cohort_sha256": intent.cohort_sha256,
                "intent_sha256": intent.intent_sha256,
                "ranker_model_sha256": intent.ranker_model_sha256,
                "registered_at_utc": _utc_now_iso(),
            },
            field="event_sha256",
        )
        try:
            return _write_once_json(path, event)
        except RankingEvaluationGateError:
            if not path.is_file():
                raise
            existing = self._load_event(
                path,
                schema=LEDGER_REGISTRATION_SCHEMA_VERSION,
                name="ranking test registration",
            )
            if existing.get("intent_sha256") != intent.intent_sha256:
                raise RankingEvaluationGateError(
                    "frozen test cohort is already registered to a different intent"
                ) from None
            return path

    def assert_registered(self, intent: RankingEvaluationIntent) -> None:
        path = self._registration_path(intent.cohort_sha256)
        if not path.is_file():
            raise RankingEvaluationGateError("ranking evaluation intent is not preregistered")
        event = self._load_event(
            path,
            schema=LEDGER_REGISTRATION_SCHEMA_VERSION,
            name="ranking test registration",
        )
        if (
            event.get("cohort_sha256") != intent.cohort_sha256
            or event.get("intent_sha256") != intent.intent_sha256
            or event.get("ranker_model_sha256") != intent.ranker_model_sha256
        ):
            raise RankingEvaluationGateError(
                "ranking test registration does not match the supplied intent"
            )

    def claim_access(self, intent: RankingEvaluationIntent) -> Path:
        """Record the test look before parsing human labels or producing scores."""

        self.assert_registered(intent)
        path = self._access_path(intent.intent_sha256)
        if path.exists():
            event = self._load_event(
                path,
                schema=LEDGER_ACCESS_SCHEMA_VERSION,
                name="ranking test access",
            )
            if (
                event.get("cohort_sha256") != intent.cohort_sha256
                or event.get("intent_sha256") != intent.intent_sha256
            ):
                raise RankingEvaluationGateError("ranking test access ledger mismatch")
            return path
        event = _with_self_hash(
            {
                "schema_version": LEDGER_ACCESS_SCHEMA_VERSION,
                "cohort_sha256": intent.cohort_sha256,
                "intent_sha256": intent.intent_sha256,
                "first_accessed_at_utc": _utc_now_iso(),
            },
            field="event_sha256",
        )
        try:
            return _write_once_json(path, event)
        except RankingEvaluationGateError:
            if not path.is_file():
                raise
            existing = self._load_event(
                path,
                schema=LEDGER_ACCESS_SCHEMA_VERSION,
                name="ranking test access",
            )
            if (
                existing.get("cohort_sha256") != intent.cohort_sha256
                or existing.get("intent_sha256") != intent.intent_sha256
            ):
                raise RankingEvaluationGateError("ranking test access ledger mismatch") from None
            return path

    def record_completion(
        self,
        intent: RankingEvaluationIntent,
        *,
        comparison_report_sha256: str,
        promotion_decision_sha256: str,
    ) -> Path:
        self.assert_registered(intent)
        registration_path = self._registration_path(intent.cohort_sha256)
        access_path = self._access_path(intent.intent_sha256)
        if not access_path.is_file():
            raise RankingEvaluationGateError(
                "ranking test completion requires a prior access record"
            )
        registration = self._load_event(
            registration_path,
            schema=LEDGER_REGISTRATION_SCHEMA_VERSION,
            name="ranking test registration",
        )
        access = self._load_event(
            access_path,
            schema=LEDGER_ACCESS_SCHEMA_VERSION,
            name="ranking test access",
        )
        registration_event_sha256 = _digest(
            registration.get("event_sha256"), name="registration event sha256"
        )
        access_event_sha256 = _digest(
            access.get("event_sha256"), name="access event sha256"
        )
        evaluation_payload_sha256 = sha256_json(
            production_evaluation_payload(
                intent_sha256=intent.intent_sha256,
                registration_event_sha256=registration_event_sha256,
                access_event_sha256=access_event_sha256,
                comparison_report_sha256=comparison_report_sha256,
                promotion_decision_sha256=promotion_decision_sha256,
            )
        )
        path = self._completion_path(intent.intent_sha256)
        identity = {
            "cohort_sha256": intent.cohort_sha256,
            "intent_sha256": intent.intent_sha256,
            "registration_event_sha256": registration_event_sha256,
            "access_event_sha256": access_event_sha256,
            "evaluation_payload_sha256": evaluation_payload_sha256,
            "comparison_report_sha256": comparison_report_sha256,
            "promotion_decision_sha256": promotion_decision_sha256,
        }
        if path.exists():
            event = self._load_event(
                path,
                schema=LEDGER_COMPLETION_SCHEMA_VERSION,
                name="ranking test completion",
            )
            if any(event.get(key) != value for key, value in identity.items()):
                raise RankingEvaluationGateError("ranking test completion ledger mismatch")
            return path
        event = _with_self_hash(
            {
                "schema_version": LEDGER_COMPLETION_SCHEMA_VERSION,
                **identity,
                "completed_at_utc": _utc_now_iso(),
            },
            field="event_sha256",
        )
        try:
            return _write_once_json(path, event)
        except RankingEvaluationGateError:
            if not path.is_file():
                raise
            existing = self._load_event(
                path,
                schema=LEDGER_COMPLETION_SCHEMA_VERSION,
                name="ranking test completion",
            )
            if any(existing.get(key) != value for key, value in identity.items()):
                raise RankingEvaluationGateError(
                    "ranking test completion ledger mismatch"
                ) from None
            return path

    def evidence_paths(self, intent: RankingEvaluationIntent) -> tuple[Path, Path, Path]:
        """Return and revalidate the complete ledger chain for bundle publication."""

        registration_path = self._registration_path(intent.cohort_sha256)
        access_path = self._access_path(intent.intent_sha256)
        completion_path = self._completion_path(intent.intent_sha256)
        self.assert_registered(intent)
        if not access_path.is_file() or not completion_path.is_file():
            raise RankingEvaluationGateError(
                "ranking evaluation ledger is incomplete and cannot be published"
            )
        self._load_event(
            access_path,
            schema=LEDGER_ACCESS_SCHEMA_VERSION,
            name="ranking test access",
        )
        self._load_event(
            completion_path,
            schema=LEDGER_COMPLETION_SCHEMA_VERSION,
            name="ranking test completion",
        )
        return registration_path, access_path, completion_path


def preregister_evaluation_intent(
    intent: RankingEvaluationIntent,
    *,
    intent_root: Path,
    ledger: RankingEvaluationLedger,
) -> Path:
    """Persist deterministic intent bytes and claim its cohort exactly once."""

    path = intent_root.resolve() / f"{intent.intent_sha256}.json"
    _write_once_json(path, intent.to_dict())
    ledger.preregister(intent)
    return path


@dataclass(frozen=True, slots=True)
class PublishedRankingEvaluation:
    output_dir: Path
    bundle_manifest_sha256: str
    evaluation_payload_sha256: str
    ledger_identity_sha256: str


def _verify_existing_bundle(target: Path, expected: Mapping[str, bytes]) -> None:
    if not target.is_dir():
        raise RankingEvaluationGateError(
            f"sealed evaluation target is not a directory: {target}"
        )
    actual_names = {path.name for path in target.iterdir() if path.is_file()}
    if actual_names != set(expected):
        raise RankingEvaluationGateError(
            "existing sealed evaluation contains missing or unexpected files"
        )
    for name, payload in expected.items():
        if (target / name).read_bytes() != payload:
            raise RankingEvaluationGateError(
                f"existing sealed evaluation file differs: {name}"
            )


def publish_ranking_evaluation(
    *,
    output_root: Path,
    intent: RankingEvaluationIntent,
    registration_path: Path,
    access_path: Path,
    completion_path: Path,
    comparison_report: Mapping[str, Any],
    promotion_decision: Mapping[str, Any],
) -> PublishedRankingEvaluation:
    """Commit a completed evaluation bundle without replacing prior evidence."""

    report_bytes = _json_bytes(comparison_report)
    decision_bytes = _json_bytes(promotion_decision)
    intent_bytes = _json_bytes(intent.to_dict())
    registration = _load_json_object(
        registration_path, name="ranking test registration"
    )
    access = _load_json_object(access_path, name="ranking test access")
    completion = _load_json_object(completion_path, name="ranking test completion")
    for payload, schema, name in (
        (
            registration,
            LEDGER_REGISTRATION_SCHEMA_VERSION,
            "ranking test registration",
        ),
        (access, LEDGER_ACCESS_SCHEMA_VERSION, "ranking test access"),
        (
            completion,
            LEDGER_COMPLETION_SCHEMA_VERSION,
            "ranking test completion",
        ),
    ):
        if payload.get("schema_version") != schema:
            raise RankingEvaluationGateError(f"{name} has an unsupported schema")
        _verify_self_hash(payload, field="event_sha256", name=name)
        if (
            payload.get("cohort_sha256") != intent.cohort_sha256
            or payload.get("intent_sha256") != intent.intent_sha256
        ):
            raise RankingEvaluationGateError(f"{name} belongs to a different intent")
    registration_event_sha256 = _digest(
        registration.get("event_sha256"), name="registration event sha256"
    )
    access_event_sha256 = _digest(
        access.get("event_sha256"), name="access event sha256"
    )
    completion_event_sha256 = _digest(
        completion.get("event_sha256"), name="completion event sha256"
    )
    report_sha256 = _digest(
        comparison_report.get("report_sha256"), name="comparison report sha256"
    )
    report_content = dict(comparison_report)
    report_content.pop("report_sha256", None)
    if sha256_json(report_content) != report_sha256:
        raise RankingEvaluationGateError("ranking comparison report self-hash mismatch")
    decision_sha256 = _digest(
        promotion_decision.get("decision_sha256"), name="promotion decision sha256"
    )
    decision_content = dict(promotion_decision)
    decision_content.pop("decision_sha256", None)
    if sha256_json(decision_content) != decision_sha256:
        raise RankingEvaluationGateError("ranker promotion decision self-hash mismatch")
    evaluation_payload_sha256 = sha256_json(
        production_evaluation_payload(
            intent_sha256=intent.intent_sha256,
            registration_event_sha256=registration_event_sha256,
            access_event_sha256=access_event_sha256,
            comparison_report_sha256=report_sha256,
            promotion_decision_sha256=decision_sha256,
        )
    )
    if (
        completion.get("registration_event_sha256") != registration_event_sha256
        or completion.get("access_event_sha256") != access_event_sha256
        or completion.get("comparison_report_sha256") != report_sha256
        or completion.get("promotion_decision_sha256") != decision_sha256
        or completion.get("evaluation_payload_sha256") != evaluation_payload_sha256
    ):
        raise RankingEvaluationGateError(
            "ranking completion was not recorded for these exact evaluation outputs"
        )
    ledger = production_ledger_identity_payload(
        registration_event_sha256=registration_event_sha256,
        access_event_sha256=access_event_sha256,
        completion_event_sha256=completion_event_sha256,
    )
    ledger_identity_sha256 = sha256_json(ledger)
    file_payloads = {
        "evaluation-intent.json": intent_bytes,
        "ranking-test-registration.json": _json_bytes(registration),
        "ranking-test-access.json": _json_bytes(access),
        "ranking-test-completion.json": _json_bytes(completion),
        "ranking-comparison.json": report_bytes,
        "ranker-promotion-decision.json": decision_bytes,
    }
    manifest: dict[str, Any] = {
        "schema_version": EVALUATION_BUNDLE_SCHEMA_VERSION,
        "intent_sha256": intent.intent_sha256,
        "cohort_sha256": intent.cohort_sha256,
        "evaluation_payload_sha256": evaluation_payload_sha256,
        "ledger_identity_sha256": ledger_identity_sha256,
        "ledger": ledger,
        "files": {
            name: {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
            for name, payload in sorted(file_payloads.items())
        },
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    manifest_bytes = _json_bytes(manifest)
    expected = {**file_payloads, "manifest.json": manifest_bytes}

    root = output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / intent.intent_sha256
    if target.exists():
        _verify_existing_bundle(target, expected)
        verified = verify_ranking_evaluation_bundle(target / "manifest.json")
        return PublishedRankingEvaluation(
            output_dir=target,
            bundle_manifest_sha256=verified.manifest_sha256,
            evaluation_payload_sha256=verified.evaluation_payload_sha256,
            ledger_identity_sha256=verified.ledger_identity_sha256,
        )

    temporary = Path(tempfile.mkdtemp(prefix=f".{intent.intent_sha256}.", dir=root))
    try:
        for name, payload in expected.items():
            path = temporary / name
            with path.open("wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        try:
            os.rename(temporary, target)
        except OSError:
            if not target.exists():
                raise
            _verify_existing_bundle(target, expected)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
    verified = verify_ranking_evaluation_bundle(target / "manifest.json")
    return PublishedRankingEvaluation(
        output_dir=target,
        bundle_manifest_sha256=verified.manifest_sha256,
        evaluation_payload_sha256=verified.evaluation_payload_sha256,
        ledger_identity_sha256=verified.ledger_identity_sha256,
    )
