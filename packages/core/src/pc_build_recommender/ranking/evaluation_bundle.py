"""Verification for a sealed, one-shot production ranking evaluation bundle."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pc_build_recommender.evaluation.manifest import sha256_file, sha256_json
from pc_build_recommender.retrieval.benchmark import COMPARISON_REPORT_SCHEMA_VERSION

from .promotion import (
    RankerPromotionDecision,
    RankerPromotionPolicy,
    assert_production_ranker_promotion_policy,
)

RANKING_EVALUATION_INTENT_SCHEMA_VERSION = (
    "pc-build-recommender.ranking-evaluation-intent.v1"
)
RANKING_LEDGER_REGISTRATION_SCHEMA_VERSION = (
    "pc-build-recommender.ranking-test-registration.v1"
)
RANKING_LEDGER_ACCESS_SCHEMA_VERSION = "pc-build-recommender.ranking-test-access.v1"
RANKING_LEDGER_COMPLETION_SCHEMA_VERSION = (
    "pc-build-recommender.ranking-test-completion.v2"
)
RANKING_EVALUATION_PAYLOAD_SCHEMA_VERSION = (
    "pc-build-recommender.ranking-evaluation-payload.v1"
)
RANKING_EVALUATION_LEDGER_IDENTITY_SCHEMA_VERSION = (
    "pc-build-recommender.ranking-evaluation-ledger-identity.v1"
)
RANKING_EVALUATION_BUNDLE_SCHEMA_VERSION = (
    "pc-build-recommender.ranking-evaluation-bundle.v2"
)
MINIMUM_PRODUCTION_BOOTSTRAP_RESAMPLES = 2
MAXIMUM_PRODUCTION_BOOTSTRAP_RESAMPLES = 10_000
MAXIMUM_RANKING_BUNDLE_FILE_BYTES = 64 * 1024 * 1024

_BUNDLE_FILE_NAMES = frozenset(
    {
        "evaluation-intent.json",
        "ranking-test-registration.json",
        "ranking-test-access.json",
        "ranking-test-completion.json",
        "ranking-comparison.json",
        "ranker-promotion-decision.json",
    }
)


class RankingEvaluationBundleError(ValueError):
    """Raised when sealed ranking evaluation evidence is incomplete or inconsistent."""


def _reject_nonfinite(value: str) -> None:
    raise RankingEvaluationBundleError(f"non-finite JSON value is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RankingEvaluationBundleError(
                f"duplicate JSON object key is forbidden: {key!r}"
            )
        result[key] = value
    return result


def _load_json_object(path: Path, *, name: str) -> dict[str, Any]:
    try:
        with path.open("rb") as stream:
            raw = stream.read(MAXIMUM_RANKING_BUNDLE_FILE_BYTES + 1)
    except OSError as error:
        raise RankingEvaluationBundleError(f"unable to read {name}: {error}") from error
    if len(raw) > MAXIMUM_RANKING_BUNDLE_FILE_BYTES:
        raise RankingEvaluationBundleError(f"{name} exceeds the safety limit")
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RankingEvaluationBundleError(f"{name} is not valid UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise RankingEvaluationBundleError(f"{name} must be a JSON object")
    return payload


def _exact_fields(
    payload: Mapping[str, Any],
    *,
    expected: frozenset[str],
    name: str,
) -> None:
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        extra = sorted(set(payload) - expected)
        raise RankingEvaluationBundleError(
            f"{name} fields are incomplete or unexpected; missing={missing}, extra={extra}"
        )


def _object(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RankingEvaluationBundleError(f"{name} must be an object")
    return {str(key): item for key, item in value.items()}


def _digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RankingEvaluationBundleError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _positive_size(value: object, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise RankingEvaluationBundleError(f"{name} must be a positive integer")
    return value


def _timestamp(value: object, *, name: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise RankingEvaluationBundleError(f"{name} must be a non-empty timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise RankingEvaluationBundleError(f"{name} is not an ISO-8601 timestamp") from error
    if parsed.utcoffset() is None:
        raise RankingEvaluationBundleError(f"{name} must be timezone-aware")
    return parsed


def _verify_self_hash(payload: Mapping[str, Any], *, field: str, name: str) -> str:
    digest = _digest(payload.get(field), name=f"{name}.{field}")
    content = dict(payload)
    content.pop(field, None)
    if sha256_json(content) != digest:
        raise RankingEvaluationBundleError(f"{name} self-hash mismatch")
    return digest


def production_evaluation_payload(
    *,
    intent_sha256: str,
    registration_event_sha256: str,
    access_event_sha256: str,
    comparison_report_sha256: str,
    promotion_decision_sha256: str,
) -> dict[str, str]:
    """Return the immutable result identity recorded before bundle publication."""

    return {
        "schema_version": RANKING_EVALUATION_PAYLOAD_SCHEMA_VERSION,
        "intent_sha256": _digest(intent_sha256, name="intent_sha256"),
        "registration_event_sha256": _digest(
            registration_event_sha256, name="registration_event_sha256"
        ),
        "access_event_sha256": _digest(
            access_event_sha256, name="access_event_sha256"
        ),
        "comparison_report_sha256": _digest(
            comparison_report_sha256, name="comparison_report_sha256"
        ),
        "promotion_decision_sha256": _digest(
            promotion_decision_sha256, name="promotion_decision_sha256"
        ),
    }


def production_ledger_identity_payload(
    *,
    registration_event_sha256: str,
    access_event_sha256: str,
    completion_event_sha256: str,
) -> dict[str, str]:
    """Return the portable identity of the three append-only ledger events."""

    return {
        "schema_version": RANKING_EVALUATION_LEDGER_IDENTITY_SCHEMA_VERSION,
        "registration_event_sha256": _digest(
            registration_event_sha256, name="registration_event_sha256"
        ),
        "access_event_sha256": _digest(
            access_event_sha256, name="access_event_sha256"
        ),
        "completion_event_sha256": _digest(
            completion_event_sha256, name="completion_event_sha256"
        ),
    }


@dataclass(frozen=True, slots=True)
class VerifiedRankingEvaluationBundle:
    """Content and governance identities admitted for production serving."""

    manifest_path: Path
    manifest_sha256: str
    intent_sha256: str
    cohort_sha256: str
    evaluation_payload_sha256: str
    ledger_identity_sha256: str
    evaluator_source_sha256: str
    dependency_lock_sha256: str
    n_resamples: int
    policy: RankerPromotionPolicy
    comparison_report_path: Path
    promotion_decision_path: Path
    comparison_report: dict[str, Any]
    promotion_decision: RankerPromotionDecision


def _verify_intent(
    payload: Mapping[str, Any],
) -> tuple[str, str, str, str, int, int, RankerPromotionPolicy]:
    _exact_fields(
        payload,
        expected=frozenset(
            {
                "schema_version",
                "cohort_sha256",
                "cohort",
                "ranker_artifact",
                "evaluation_runtime",
                "challenger_model",
                "evaluation_parameters",
                "intent_sha256",
            }
        ),
        name="evaluation intent",
    )
    if payload.get("schema_version") != RANKING_EVALUATION_INTENT_SCHEMA_VERSION:
        raise RankingEvaluationBundleError("unsupported ranking evaluation intent schema")
    intent_sha256 = _verify_self_hash(
        payload, field="intent_sha256", name="evaluation intent"
    )
    cohort = _object(payload.get("cohort"), name="evaluation intent cohort")
    cohort_sha256 = _digest(payload.get("cohort_sha256"), name="cohort_sha256")
    if sha256_json(cohort) != cohort_sha256 or cohort.get("split_name") != "test":
        raise RankingEvaluationBundleError("evaluation intent cohort binding is invalid")
    artifact = _object(payload.get("ranker_artifact"), name="intent ranker artifact")
    _exact_fields(
        artifact,
        expected=frozenset({"model_sha256", "metadata_sha256", "manifest_sha256"}),
        name="intent ranker artifact",
    )
    for field in artifact:
        _digest(artifact[field], name=f"intent ranker artifact.{field}")
    runtime = _object(payload.get("evaluation_runtime"), name="intent evaluation runtime")
    _exact_fields(
        runtime,
        expected=frozenset(
            {
                "evaluator_module",
                "evaluator_source_sha256",
                "dependency_lock_name",
                "dependency_lock_sha256",
            }
        ),
        name="intent evaluation runtime",
    )
    if (
        runtime.get("evaluator_module") != "training.evaluate_ranking"
        or runtime.get("dependency_lock_name") != "uv.lock"
    ):
        raise RankingEvaluationBundleError(
            "evaluation intent does not bind the governed evaluator runtime"
        )
    evaluator_source_sha256 = _digest(
        runtime.get("evaluator_source_sha256"), name="evaluator_source_sha256"
    )
    dependency_lock_sha256 = _digest(
        runtime.get("dependency_lock_sha256"), name="dependency_lock_sha256"
    )
    challenger = payload.get("challenger_model")
    if not isinstance(challenger, str) or not challenger or challenger in {"bm25", "rrf_hybrid"}:
        raise RankingEvaluationBundleError("evaluation intent challenger is invalid")
    parameters = _object(
        payload.get("evaluation_parameters"), name="intent evaluation parameters"
    )
    _exact_fields(
        parameters,
        expected=frozenset({"n_resamples", "bootstrap_seed", "promotion_policy"}),
        name="intent evaluation parameters",
    )
    n_resamples = parameters.get("n_resamples")
    bootstrap_seed = parameters.get("bootstrap_seed")
    if (
        type(n_resamples) is not int
        or not MINIMUM_PRODUCTION_BOOTSTRAP_RESAMPLES
        <= n_resamples
        <= MAXIMUM_PRODUCTION_BOOTSTRAP_RESAMPLES
    ):
        raise RankingEvaluationBundleError(
            "production n_resamples must be between "
            f"{MINIMUM_PRODUCTION_BOOTSTRAP_RESAMPLES} and "
            f"{MAXIMUM_PRODUCTION_BOOTSTRAP_RESAMPLES}"
        )
    if type(bootstrap_seed) is not int:
        raise RankingEvaluationBundleError("bootstrap_seed must be an integer")
    try:
        policy = RankerPromotionPolicy.from_dict(
            _object(parameters.get("promotion_policy"), name="promotion policy")
        )
        assert_production_ranker_promotion_policy(policy)
    except (TypeError, ValueError) as error:
        raise RankingEvaluationBundleError(
            f"invalid production ranker promotion policy: {error}"
        ) from error
    return (
        intent_sha256,
        cohort_sha256,
        evaluator_source_sha256,
        dependency_lock_sha256,
        n_resamples,
        bootstrap_seed,
        policy,
    )


def _verify_event(
    payload: Mapping[str, Any],
    *,
    schema: str,
    expected_fields: frozenset[str],
    name: str,
) -> str:
    _exact_fields(payload, expected=expected_fields, name=name)
    if payload.get("schema_version") != schema:
        raise RankingEvaluationBundleError(f"{name} has an unsupported schema")
    return _verify_self_hash(payload, field="event_sha256", name=name)


def verify_ranking_evaluation_bundle(
    manifest_path: str | Path,
) -> VerifiedRankingEvaluationBundle:
    """Verify the complete sealed evaluation and append-only ledger identity."""

    resolved_manifest = Path(manifest_path).resolve()
    manifest = _load_json_object(resolved_manifest, name="ranking evaluation bundle manifest")
    _exact_fields(
        manifest,
        expected=frozenset(
            {
                "schema_version",
                "intent_sha256",
                "cohort_sha256",
                "evaluation_payload_sha256",
                "ledger_identity_sha256",
                "ledger",
                "files",
                "manifest_sha256",
            }
        ),
        name="ranking evaluation bundle manifest",
    )
    if manifest.get("schema_version") != RANKING_EVALUATION_BUNDLE_SCHEMA_VERSION:
        raise RankingEvaluationBundleError("unsupported ranking evaluation bundle schema")
    manifest_sha256 = _verify_self_hash(
        manifest, field="manifest_sha256", name="ranking evaluation bundle manifest"
    )
    root = resolved_manifest.parent
    actual_files = {path.name for path in root.iterdir() if path.is_file()}
    if actual_files != _BUNDLE_FILE_NAMES | {resolved_manifest.name}:
        raise RankingEvaluationBundleError(
            "ranking evaluation bundle contains missing or unexpected files"
        )
    files = _object(manifest.get("files"), name="ranking evaluation bundle files")
    if set(files) != _BUNDLE_FILE_NAMES:
        raise RankingEvaluationBundleError(
            "ranking evaluation bundle manifest does not bind the complete file set"
        )
    paths: dict[str, Path] = {}
    for name in sorted(_BUNDLE_FILE_NAMES):
        evidence = _object(files[name], name=f"bundle file evidence {name}")
        _exact_fields(
            evidence,
            expected=frozenset({"sha256", "size_bytes"}),
            name=f"bundle file evidence {name}",
        )
        path = root / name
        expected_size = _positive_size(evidence.get("size_bytes"), name=f"{name}.size_bytes")
        expected_sha256 = _digest(evidence.get("sha256"), name=f"{name}.sha256")
        if not path.is_file() or path.stat().st_size != expected_size:
            raise RankingEvaluationBundleError(f"sealed bundle file size mismatch: {name}")
        if sha256_file(path) != expected_sha256:
            raise RankingEvaluationBundleError(f"sealed bundle file hash mismatch: {name}")
        paths[name] = path

    intent = _load_json_object(paths["evaluation-intent.json"], name="evaluation intent")
    (
        intent_sha256,
        cohort_sha256,
        evaluator_source_sha256,
        dependency_lock_sha256,
        n_resamples,
        bootstrap_seed,
        policy,
    ) = _verify_intent(intent)
    if (
        manifest.get("intent_sha256") != intent_sha256
        or manifest.get("cohort_sha256") != cohort_sha256
    ):
        raise RankingEvaluationBundleError("bundle manifest does not match evaluation intent")

    registration = _load_json_object(
        paths["ranking-test-registration.json"], name="ranking test registration"
    )
    registration_event_sha256 = _verify_event(
        registration,
        schema=RANKING_LEDGER_REGISTRATION_SCHEMA_VERSION,
        expected_fields=frozenset(
            {
                "schema_version",
                "cohort_sha256",
                "intent_sha256",
                "ranker_model_sha256",
                "registered_at_utc",
                "event_sha256",
            }
        ),
        name="ranking test registration",
    )
    access = _load_json_object(paths["ranking-test-access.json"], name="ranking test access")
    access_event_sha256 = _verify_event(
        access,
        schema=RANKING_LEDGER_ACCESS_SCHEMA_VERSION,
        expected_fields=frozenset(
            {
                "schema_version",
                "cohort_sha256",
                "intent_sha256",
                "first_accessed_at_utc",
                "event_sha256",
            }
        ),
        name="ranking test access",
    )
    completion = _load_json_object(
        paths["ranking-test-completion.json"], name="ranking test completion"
    )
    completion_event_sha256 = _verify_event(
        completion,
        schema=RANKING_LEDGER_COMPLETION_SCHEMA_VERSION,
        expected_fields=frozenset(
            {
                "schema_version",
                "cohort_sha256",
                "intent_sha256",
                "registration_event_sha256",
                "access_event_sha256",
                "evaluation_payload_sha256",
                "comparison_report_sha256",
                "promotion_decision_sha256",
                "completed_at_utc",
                "event_sha256",
            }
        ),
        name="ranking test completion",
    )
    artifact = _object(intent.get("ranker_artifact"), name="intent ranker artifact")
    for event, name in (
        (registration, "registration"),
        (access, "access"),
        (completion, "completion"),
    ):
        if (
            event.get("cohort_sha256") != cohort_sha256
            or event.get("intent_sha256") != intent_sha256
        ):
            raise RankingEvaluationBundleError(f"{name} event belongs to a different intent")
    if registration.get("ranker_model_sha256") != artifact.get("model_sha256"):
        raise RankingEvaluationBundleError("registration event binds a different ranker model")
    if (
        completion.get("registration_event_sha256") != registration_event_sha256
        or completion.get("access_event_sha256") != access_event_sha256
    ):
        raise RankingEvaluationBundleError("completion event ledger chain is invalid")
    registered_at = _timestamp(
        registration.get("registered_at_utc"), name="registered_at_utc"
    )
    accessed_at = _timestamp(access.get("first_accessed_at_utc"), name="first_accessed_at_utc")
    completed_at = _timestamp(completion.get("completed_at_utc"), name="completed_at_utc")
    if not registered_at <= accessed_at <= completed_at:
        raise RankingEvaluationBundleError("ranking ledger event chronology is invalid")

    report = _load_json_object(
        paths["ranking-comparison.json"], name="ranking comparison report"
    )
    if report.get("schema_version") != COMPARISON_REPORT_SCHEMA_VERSION:
        raise RankingEvaluationBundleError("unsupported ranking comparison report schema")
    report_sha256 = _verify_self_hash(
        report, field="report_sha256", name="ranking comparison report"
    )
    try:
        decision = RankerPromotionDecision.from_dict(
            _load_json_object(
                paths["ranker-promotion-decision.json"],
                name="ranker promotion decision",
            )
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RankingEvaluationBundleError(
            f"ranker promotion decision is invalid: {error}"
        ) from error
    if not decision.passed or decision.failures:
        raise RankingEvaluationBundleError("sealed promotion decision did not pass")
    if (
        decision.comparison_report_sha256 != report_sha256
        or decision.challenger_model != intent.get("challenger_model")
        or decision.policy != policy
        or decision.ranker_model_sha256 != artifact.get("model_sha256")
        or decision.ranker_metadata_sha256 != artifact.get("metadata_sha256")
        or decision.ranker_manifest_sha256 != artifact.get("manifest_sha256")
    ):
        raise RankingEvaluationBundleError(
            "promotion decision does not match its preregistered intent and report"
        )
    report_parameters = _object(
        report.get("evaluation_parameters"), name="comparison evaluation parameters"
    )
    report_dataset = _object(report.get("dataset"), name="comparison dataset")
    if (
        report_parameters.get("bootstrap_resamples") != n_resamples
        or report_parameters.get("bootstrap_seed") != bootstrap_seed
        or report_dataset.get("split_name") != "test"
        or report_dataset.get("split_checksum")
        != _object(intent.get("cohort"), name="intent cohort").get("query_split_checksum")
    ):
        raise RankingEvaluationBundleError(
            "comparison report does not match preregistered evaluation parameters"
        )

    evaluation_payload = production_evaluation_payload(
        intent_sha256=intent_sha256,
        registration_event_sha256=registration_event_sha256,
        access_event_sha256=access_event_sha256,
        comparison_report_sha256=report_sha256,
        promotion_decision_sha256=decision.decision_sha256,
    )
    evaluation_payload_sha256 = sha256_json(evaluation_payload)
    if (
        completion.get("evaluation_payload_sha256") != evaluation_payload_sha256
        or completion.get("comparison_report_sha256") != report_sha256
        or completion.get("promotion_decision_sha256") != decision.decision_sha256
        or manifest.get("evaluation_payload_sha256") != evaluation_payload_sha256
    ):
        raise RankingEvaluationBundleError("completion event result identity is invalid")

    ledger_payload = production_ledger_identity_payload(
        registration_event_sha256=registration_event_sha256,
        access_event_sha256=access_event_sha256,
        completion_event_sha256=completion_event_sha256,
    )
    ledger_identity_sha256 = sha256_json(ledger_payload)
    ledger = _object(manifest.get("ledger"), name="bundle ledger identity")
    if ledger != ledger_payload or manifest.get("ledger_identity_sha256") != ledger_identity_sha256:
        raise RankingEvaluationBundleError("bundle ledger identity is invalid")

    return VerifiedRankingEvaluationBundle(
        manifest_path=resolved_manifest,
        manifest_sha256=manifest_sha256,
        intent_sha256=intent_sha256,
        cohort_sha256=cohort_sha256,
        evaluation_payload_sha256=evaluation_payload_sha256,
        ledger_identity_sha256=ledger_identity_sha256,
        evaluator_source_sha256=evaluator_source_sha256,
        dependency_lock_sha256=dependency_lock_sha256,
        n_resamples=n_resamples,
        policy=policy,
        comparison_report_path=paths["ranking-comparison.json"],
        promotion_decision_path=paths["ranker-promotion-decision.json"],
        comparison_report=report,
        promotion_decision=decision,
    )
