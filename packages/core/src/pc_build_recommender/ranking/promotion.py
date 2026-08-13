"""Fail-closed promotion gates for a learned component ranker."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pc_build_recommender.evaluation.manifest import sha256_json
from pc_build_recommender.retrieval.benchmark import RankingComparisonReport

from .models import ProductRanker

PROMOTION_DECISION_SCHEMA_VERSION = "pc-build-recommender.ranker-promotion-decision.v2"
MAX_PROMOTION_DECISION_BYTES = 1024 * 1024

_PASSING_MEASURED_FIELDS = {
    "test_query_count",
    "test_query_group_count",
    "label_source",
    "split_name",
    "challenger_recall_at_50",
    "relative_ndcg_lift_percent_over_bm25",
    "bm25_ndcg_delta_ci_lower",
    "rrf_ndcg_delta_ci_lower",
    "ranker_training_label_source",
    "ranker_version",
    "ranker_model_sha256",
    "ranker_metadata_sha256",
    "ranker_manifest_sha256",
    "artifact_binding_sha256",
    "metadata_payload_sha256",
    "feature_snapshot_sha256",
    "candidate_snapshot_sha256",
    "score_snapshot_sha256",
    "ranking_sha256",
}

_PROMOTION_DECISION_FIELDS = {
    "schema_version",
    "comparison_report_sha256",
    "challenger_model",
    "decision",
    "passed",
    "failures",
    "measured_values",
    "policy",
    "ranker_version",
    "ranker_model_sha256",
    "ranker_metadata_sha256",
    "ranker_manifest_sha256",
    "decision_sha256",
}


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _record_finite_metric(
    measured: dict[str, float | int | str | None],
    failures: list[str],
    *,
    key: str,
    value: object,
    label: str,
) -> float | None:
    numeric = _finite_number(value)
    measured[key] = numeric
    if numeric is None:
        failures.append(f"{label} is missing or non-finite")
    return numeric


@dataclass(frozen=True, slots=True)
class RankerPromotionPolicy:
    minimum_test_query_groups: int = 50
    minimum_recall_at_50: float = 0.95
    minimum_relative_ndcg_lift_percent_over_bm25: float = 15.0
    minimum_bm25_ndcg_delta_ci_lower: float = 0.0
    rrf_ndcg_noninferiority_margin: float = 0.01

    def __post_init__(self) -> None:
        if type(self.minimum_test_query_groups) is not int or self.minimum_test_query_groups < 1:
            raise ValueError("minimum_test_query_groups must be positive")
        for name, value in (
            ("minimum_recall_at_50", self.minimum_recall_at_50),
            (
                "minimum_relative_ndcg_lift_percent_over_bm25",
                self.minimum_relative_ndcg_lift_percent_over_bm25,
            ),
            ("minimum_bm25_ndcg_delta_ci_lower", self.minimum_bm25_ndcg_delta_ci_lower),
            ("rrf_ndcg_noninferiority_margin", self.rrf_ndcg_noninferiority_margin),
        ):
            if _finite_number(value) is None:
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= self.minimum_recall_at_50 <= 1.0:
            raise ValueError("minimum_recall_at_50 must be between zero and one")
        if self.minimum_relative_ndcg_lift_percent_over_bm25 < 0.0:
            raise ValueError("minimum relative NDCG lift must be non-negative")
        if self.rrf_ndcg_noninferiority_margin < 0.0:
            raise ValueError("RRF non-inferiority margin must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "minimum_test_query_groups": self.minimum_test_query_groups,
            "minimum_recall_at_50": self.minimum_recall_at_50,
            "minimum_relative_ndcg_lift_percent_over_bm25": (
                self.minimum_relative_ndcg_lift_percent_over_bm25
            ),
            "minimum_bm25_ndcg_delta_ci_lower": self.minimum_bm25_ndcg_delta_ci_lower,
            "rrf_ndcg_noninferiority_margin": self.rrf_ndcg_noninferiority_margin,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RankerPromotionPolicy:
        required = {
            "minimum_test_query_groups",
            "minimum_recall_at_50",
            "minimum_relative_ndcg_lift_percent_over_bm25",
            "minimum_bm25_ndcg_delta_ci_lower",
            "rrf_ndcg_noninferiority_margin",
        }
        if set(payload) != required:
            raise ValueError("ranker promotion policy fields are incomplete or unexpected")
        minimum_groups = payload["minimum_test_query_groups"]
        if type(minimum_groups) is not int:
            raise TypeError("minimum_test_query_groups must be an integer")
        numeric: dict[str, float] = {}
        for name in required - {"minimum_test_query_groups"}:
            value = _finite_number(payload[name])
            if value is None:
                raise ValueError(f"{name} must be finite")
            numeric[name] = value
        return cls(
            minimum_test_query_groups=minimum_groups,
            minimum_recall_at_50=numeric["minimum_recall_at_50"],
            minimum_relative_ndcg_lift_percent_over_bm25=numeric[
                "minimum_relative_ndcg_lift_percent_over_bm25"
            ],
            minimum_bm25_ndcg_delta_ci_lower=numeric["minimum_bm25_ndcg_delta_ci_lower"],
            rrf_ndcg_noninferiority_margin=numeric["rrf_ndcg_noninferiority_margin"],
        )


PRODUCTION_RANKER_PROMOTION_POLICY = RankerPromotionPolicy()


def assert_production_ranker_promotion_policy(
    policy: RankerPromotionPolicy,
) -> None:
    """Reject a production policy that weakens any reviewed release threshold."""

    floor = PRODUCTION_RANKER_PROMOTION_POLICY
    failures: list[str] = []
    if policy.minimum_test_query_groups < floor.minimum_test_query_groups:
        failures.append("minimum_test_query_groups")
    if policy.minimum_recall_at_50 < floor.minimum_recall_at_50:
        failures.append("minimum_recall_at_50")
    if (
        policy.minimum_relative_ndcg_lift_percent_over_bm25
        < floor.minimum_relative_ndcg_lift_percent_over_bm25
    ):
        failures.append("minimum_relative_ndcg_lift_percent_over_bm25")
    if (
        policy.minimum_bm25_ndcg_delta_ci_lower
        < floor.minimum_bm25_ndcg_delta_ci_lower
    ):
        failures.append("minimum_bm25_ndcg_delta_ci_lower")
    # A larger non-inferiority margin is weaker, unlike the minimum thresholds.
    if policy.rrf_ndcg_noninferiority_margin > floor.rrf_ndcg_noninferiority_margin:
        failures.append("rrf_ndcg_noninferiority_margin")
    if failures:
        raise ValueError(
            "ranker promotion policy weakens production floors: "
            + ", ".join(failures)
        )


@dataclass(frozen=True, slots=True)
class RankerPromotionDecision:
    comparison_report_sha256: str
    challenger_model: str
    passed: bool
    failures: tuple[str, ...]
    measured_values: dict[str, float | int | str | None]
    policy: RankerPromotionPolicy
    ranker_version: str | None
    ranker_model_sha256: str | None
    ranker_metadata_sha256: str | None
    ranker_manifest_sha256: str | None
    decision_sha256: str

    def content_payload(self) -> dict[str, object]:
        return {
            "schema_version": PROMOTION_DECISION_SCHEMA_VERSION,
            "comparison_report_sha256": self.comparison_report_sha256,
            "challenger_model": self.challenger_model,
            "decision": "promote" if self.passed else "do_not_promote",
            "passed": self.passed,
            "failures": list(self.failures),
            "measured_values": self.measured_values,
            "policy": self.policy.to_dict(),
            "ranker_version": self.ranker_version,
            "ranker_model_sha256": self.ranker_model_sha256,
            "ranker_metadata_sha256": self.ranker_metadata_sha256,
            "ranker_manifest_sha256": self.ranker_manifest_sha256,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.content_payload(), "decision_sha256": self.decision_sha256}

    def __post_init__(self) -> None:
        if type(self.passed) is not bool:
            raise TypeError("passed must be a boolean")
        if self.decision_sha256:
            _require_sha256(self.decision_sha256, "decision_sha256")
            if sha256_json(self.content_payload()) != self.decision_sha256:
                raise ValueError("promotion decision hash does not match its contents")
        _require_sha256(self.comparison_report_sha256, "comparison_report_sha256")
        if not self.challenger_model:
            raise ValueError("challenger_model must not be empty")
        if self.passed and self.failures:
            raise ValueError("a passing promotion decision cannot contain failures")
        if not self.passed and not self.failures:
            raise ValueError("a failed promotion decision must explain why")
        for name, value in self.measured_values.items():
            if isinstance(value, bool):
                raise TypeError(f"measured value {name!r} must not be boolean")
            if isinstance(value, int | float) and not math.isfinite(float(value)):
                raise ValueError(f"measured value {name!r} must be finite")
        if self.passed:
            if not self.ranker_version:
                raise ValueError("passing promotion decision requires a ranker version")
            _require_sha256(self.ranker_model_sha256, "ranker_model_sha256")
            _require_sha256(self.ranker_metadata_sha256, "ranker_metadata_sha256")
            _require_sha256(self.ranker_manifest_sha256, "ranker_manifest_sha256")
            missing = _PASSING_MEASURED_FIELDS - set(self.measured_values)
            if missing:
                raise ValueError(
                    f"passing promotion decision lacks measured fields: {sorted(missing)}"
                )
            if self.measured_values.get("label_source") != "human":
                raise ValueError("passing promotion decision requires human evaluation labels")
            if self.measured_values.get("split_name") != "test":
                raise ValueError("passing promotion decision requires the frozen test split")
            if self.measured_values.get("ranker_training_label_source") != "human":
                raise ValueError("passing promotion decision requires human training labels")
            if self.measured_values.get("ranker_version") != self.ranker_version:
                raise ValueError("measured ranker version does not match promotion decision")
            if self.measured_values.get("ranker_model_sha256") != self.ranker_model_sha256:
                raise ValueError("measured ranker model hash does not match promotion decision")
            if self.measured_values.get("ranker_metadata_sha256") != self.ranker_metadata_sha256:
                raise ValueError("measured ranker metadata hash does not match promotion decision")
            if self.measured_values.get("ranker_manifest_sha256") != self.ranker_manifest_sha256:
                raise ValueError("measured ranker manifest hash does not match promotion decision")
            for name in (
                "challenger_recall_at_50",
                "relative_ndcg_lift_percent_over_bm25",
                "bm25_ndcg_delta_ci_lower",
                "rrf_ndcg_delta_ci_lower",
            ):
                if _finite_number(self.measured_values.get(name)) is None:
                    raise ValueError(f"passing promotion decision requires finite {name}")
            for name in ("test_query_count", "test_query_group_count"):
                value = self.measured_values.get(name)
                if type(value) is not int or value < 1:
                    raise ValueError(f"passing promotion decision requires positive {name}")
            group_count = self.measured_values["test_query_group_count"]
            recall = _finite_number(self.measured_values["challenger_recall_at_50"])
            lift = _finite_number(self.measured_values["relative_ndcg_lift_percent_over_bm25"])
            bm25_ci_lower = _finite_number(self.measured_values["bm25_ndcg_delta_ci_lower"])
            rrf_ci_lower = _finite_number(self.measured_values["rrf_ndcg_delta_ci_lower"])
            assert isinstance(group_count, int)
            assert recall is not None
            assert lift is not None
            assert bm25_ci_lower is not None
            assert rrf_ci_lower is not None
            if group_count < self.policy.minimum_test_query_groups:
                raise ValueError("passing decision violates minimum test query-group policy")
            if recall < self.policy.minimum_recall_at_50:
                raise ValueError("passing decision violates Recall@50 policy")
            if lift < self.policy.minimum_relative_ndcg_lift_percent_over_bm25:
                raise ValueError("passing decision violates relative NDCG lift policy")
            if bm25_ci_lower <= self.policy.minimum_bm25_ndcg_delta_ci_lower:
                raise ValueError("passing decision violates BM25 confidence-interval policy")
            if rrf_ci_lower < -self.policy.rrf_ndcg_noninferiority_margin:
                raise ValueError("passing decision violates RRF non-inferiority policy")
        elif self.ranker_model_sha256 is not None:
            _require_sha256(self.ranker_model_sha256, "ranker_model_sha256")
        if not self.passed:
            if self.ranker_metadata_sha256 is not None:
                _require_sha256(self.ranker_metadata_sha256, "ranker_metadata_sha256")
            if self.ranker_manifest_sha256 is not None:
                _require_sha256(self.ranker_manifest_sha256, "ranker_manifest_sha256")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RankerPromotionDecision:
        if set(payload) != _PROMOTION_DECISION_FIELDS:
            raise ValueError("ranker promotion decision fields are incomplete or unexpected")
        if payload.get("schema_version") != PROMOTION_DECISION_SCHEMA_VERSION:
            raise ValueError("unsupported ranker promotion decision schema")
        policy_payload = payload.get("policy")
        measured_payload = payload.get("measured_values")
        failures_payload = payload.get("failures")
        if not isinstance(policy_payload, Mapping):
            raise TypeError("ranker promotion policy must be an object")
        if not isinstance(measured_payload, Mapping):
            raise TypeError("ranker promotion measured values must be an object")
        if not isinstance(failures_payload, list) or not all(
            isinstance(reason, str) for reason in failures_payload
        ):
            raise TypeError("ranker promotion failures must be a list of strings")
        passed = payload.get("passed")
        if type(passed) is not bool:
            raise TypeError("ranker promotion passed must be a boolean")
        expected_decision = "promote" if passed else "do_not_promote"
        if payload.get("decision") != expected_decision:
            raise ValueError("ranker promotion decision label conflicts with passed")
        ranker_version = payload.get("ranker_version")
        model_sha256 = payload.get("ranker_model_sha256")
        metadata_sha256 = payload.get("ranker_metadata_sha256")
        manifest_sha256 = payload.get("ranker_manifest_sha256")
        decision_sha256 = payload.get("decision_sha256")
        return cls(
            comparison_report_sha256=_require_sha256(
                payload.get("comparison_report_sha256"),
                "comparison_report_sha256",
            ),
            challenger_model=str(payload.get("challenger_model", "")),
            passed=passed,
            failures=tuple(failures_payload),
            measured_values={str(key): value for key, value in measured_payload.items()},
            policy=RankerPromotionPolicy.from_dict(policy_payload),
            ranker_version=str(ranker_version) if ranker_version is not None else None,
            ranker_model_sha256=(
                _require_sha256(model_sha256, "ranker_model_sha256")
                if model_sha256 is not None
                else None
            ),
            ranker_metadata_sha256=(
                _require_sha256(metadata_sha256, "ranker_metadata_sha256")
                if metadata_sha256 is not None
                else None
            ),
            ranker_manifest_sha256=(
                _require_sha256(manifest_sha256, "ranker_manifest_sha256")
                if manifest_sha256 is not None
                else None
            ),
            decision_sha256=_require_sha256(decision_sha256, "decision_sha256"),
        )


def evaluate_ranker_promotion(
    report: RankingComparisonReport,
    *,
    challenger_model: str,
    ranker: ProductRanker | None,
    bm25_model: str = "bm25",
    rrf_model: str = "rrf_hybrid",
    policy: RankerPromotionPolicy | None = None,
) -> RankerPromotionDecision:
    """Apply predeclared evidence and metric gates without model-specific exceptions."""

    selected_policy = policy or RankerPromotionPolicy()
    try:
        expected_report_sha256 = sha256_json(report.content_payload())
    except ValueError as error:
        raise ValueError("ranking comparison report contains non-finite evidence") from error
    if report.report_sha256 != expected_report_sha256:
        raise ValueError("ranking comparison report hash does not match its contents")
    failures: list[str] = []
    measured: dict[str, float | int | str | None] = {
        "test_query_count": (
            report.query_count
            if type(report.query_count) is int and report.query_count > 0
            else None
        ),
        "test_query_group_count": (
            report.query_group_count
            if type(report.query_group_count) is int and report.query_group_count > 0
            else None
        ),
        "label_source": report.label_source,
        "split_name": report.split_name,
    }
    if measured["test_query_count"] is None:
        failures.append("test query count must be a positive integer")
    if measured["test_query_group_count"] is None:
        failures.append("test query-group count must be a positive integer")
    if not report.eligible_for_promotion:
        failures.append("ranking comparison report is not promotion-eligible")
        failures.extend(report.promotion_block_reasons)
    if (
        type(report.query_group_count) is int
        and report.query_group_count < selected_policy.minimum_test_query_groups
    ):
        failures.append(
            "frozen test query-group count is below the promotion minimum: "
            f"{report.query_group_count} < {selected_policy.minimum_test_query_groups}"
        )
    challenger = report.model_evaluations.get(challenger_model)
    if challenger is None:
        failures.append(f"challenger model {challenger_model!r} is missing from the report")
    else:
        if not challenger.eligible_for_promotion:
            failures.append("challenger evaluation is not promotion-eligible")
        recall_at_50 = _record_finite_metric(
            measured,
            failures,
            key="challenger_recall_at_50",
            value=challenger.recall_at.get(50),
            label="challenger Recall@50",
        )
        if recall_at_50 is not None and recall_at_50 < selected_policy.minimum_recall_at_50:
            failures.append(
                f"challenger Recall@50 {recall_at_50:.6f} is below "
                f"{selected_policy.minimum_recall_at_50:.6f}"
            )

    bm25_pair = report.paired_comparisons.get(f"{challenger_model}_minus_{bm25_model}")
    bm25_ndcg = bm25_pair.get("ndcg_at_10") if bm25_pair is not None else None
    if bm25_ndcg is None:
        failures.append("paired challenger-minus-BM25 NDCG@10 evidence is missing")
    else:
        relative_delta = _record_finite_metric(
            measured,
            failures,
            key="relative_ndcg_lift_percent_over_bm25",
            value=bm25_ndcg.relative_delta_percent,
            label="relative NDCG@10 lift over BM25",
        )
        bm25_ci_lower = _record_finite_metric(
            measured,
            failures,
            key="bm25_ndcg_delta_ci_lower",
            value=bm25_ndcg.ci_lower,
            label="paired NDCG@10 confidence interval lower bound over BM25",
        )
        if relative_delta is not None and (
            relative_delta < selected_policy.minimum_relative_ndcg_lift_percent_over_bm25
        ):
            failures.append("relative NDCG@10 lift over BM25 is below the promotion target")
        if bm25_ci_lower is not None and (
            bm25_ci_lower <= selected_policy.minimum_bm25_ndcg_delta_ci_lower
        ):
            failures.append("paired NDCG@10 confidence interval over BM25 does not exclude zero")

    rrf_pair = report.paired_comparisons.get(f"{challenger_model}_minus_{rrf_model}")
    rrf_ndcg = rrf_pair.get("ndcg_at_10") if rrf_pair is not None else None
    if rrf_ndcg is None:
        failures.append("paired challenger-minus-RRF NDCG@10 evidence is missing")
    else:
        rrf_ci_lower = _record_finite_metric(
            measured,
            failures,
            key="rrf_ndcg_delta_ci_lower",
            value=rrf_ndcg.ci_lower,
            label="paired NDCG@10 confidence interval lower bound over RRF",
        )
        if rrf_ci_lower is not None and (
            rrf_ci_lower < -selected_policy.rrf_ndcg_noninferiority_margin
        ):
            failures.append("challenger is inferior to RRF beyond the allowed NDCG margin")

    binding = report.artifact_bound_rankings.get(challenger_model)
    if binding is None:
        failures.append("challenger ranking is not bound to a verified model artifact")
    else:
        measured.update(
            {
                "artifact_binding_sha256": binding.evidence_sha256,
                "metadata_payload_sha256": binding.metadata_payload_sha256,
                "feature_snapshot_sha256": binding.feature_snapshot_sha256,
                "candidate_snapshot_sha256": binding.candidate_snapshot_sha256,
                "score_snapshot_sha256": binding.score_snapshot_sha256,
                "ranking_sha256": binding.ranking_sha256,
            }
        )
        if binding.ranking_sha256 != report.ranking_sha256.get(challenger_model):
            failures.append("artifact binding and report ranking hashes do not match")
        if binding.candidate_snapshot_sha256 != report.candidate_checksum:
            failures.append("artifact binding and report candidate snapshots do not match")

    ranker_version: str | None = None
    ranker_model_sha256: str | None = None
    ranker_metadata_sha256: str | None = None
    ranker_manifest_sha256: str | None = None
    if ranker is None:
        failures.append("manifest-verified challenger ranker is missing")
    else:
        ranker_metadata = ranker.metadata
        ranker_version = ranker_metadata.ranker_version
        ranker_model_sha256 = ranker_metadata.model_sha256
        measured["ranker_training_label_source"] = ranker_metadata.training_label_source
        measured["ranker_version"] = ranker_version
        measured["ranker_model_sha256"] = ranker_model_sha256
        if not ranker_metadata.promotion_eligible:
            failures.append("ranker metadata is not promotion-eligible")
            failures.extend(ranker_metadata.promotion_block_reasons)
        if ranker_metadata.query_group_split_checksum != report.split_checksum:
            failures.append("ranker and evaluation query-group split hashes do not match")
        if ranker_metadata.training_judgment_manifest_sha256 != report.judgment_manifest_sha256:
            failures.append("ranker and evaluation judgment manifest hashes do not match")
        if ranker_model_sha256 is None:
            failures.append("ranker model artifact hash is missing")
        if not ranker.verified_artifact_loaded:
            failures.append("challenger ranker was not loaded from a verified artifact")
        try:
            ranker_artifact_identity = ranker.artifact_identity
        except RuntimeError:
            failures.append("verified ranker artifact identity is missing")
        else:
            ranker_metadata_sha256 = ranker_artifact_identity.metadata_sha256
            ranker_manifest_sha256 = ranker_artifact_identity.manifest_sha256
            measured["ranker_metadata_sha256"] = ranker_metadata_sha256
            measured["ranker_manifest_sha256"] = ranker_manifest_sha256
            if ranker_artifact_identity.model_sha256 != ranker_model_sha256:
                failures.append("ranker metadata and artifact model hashes do not match")
            if binding is not None:
                if (
                    binding.model_sha256 != ranker_artifact_identity.model_sha256
                    or binding.metadata_sha256 != ranker_artifact_identity.metadata_sha256
                    or binding.manifest_sha256 != ranker_artifact_identity.manifest_sha256
                ):
                    failures.append("evaluated ranking is bound to different artifact bytes")
                if dict(binding.ranker_metadata_payload) != ranker_metadata.to_dict():
                    failures.append("evaluated ranking metadata does not match loaded artifact")
                if binding.ranker_version != ranker_metadata.ranker_version:
                    failures.append("evaluated ranking version does not match loaded artifact")
                if binding.feature_version != ranker_metadata.feature_version or (
                    binding.feature_names != ranker_metadata.feature_names
                ):
                    failures.append("evaluated feature contract does not match loaded artifact")

    unique_failures = tuple(dict.fromkeys(failures))
    provisional = RankerPromotionDecision(
        comparison_report_sha256=report.report_sha256,
        challenger_model=challenger_model,
        passed=not unique_failures,
        failures=unique_failures,
        measured_values=measured,
        policy=selected_policy,
        ranker_version=ranker_version,
        ranker_model_sha256=ranker_model_sha256,
        ranker_metadata_sha256=ranker_metadata_sha256,
        ranker_manifest_sha256=ranker_manifest_sha256,
        decision_sha256="",
    )
    return RankerPromotionDecision(
        comparison_report_sha256=report.report_sha256,
        challenger_model=challenger_model,
        passed=not unique_failures,
        failures=unique_failures,
        measured_values=measured,
        policy=selected_policy,
        ranker_version=ranker_version,
        ranker_model_sha256=ranker_model_sha256,
        ranker_metadata_sha256=ranker_metadata_sha256,
        ranker_manifest_sha256=ranker_manifest_sha256,
        decision_sha256=sha256_json(provisional.content_payload()),
    )


def load_ranker_promotion_decision(path: str | Path) -> RankerPromotionDecision:
    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"non-finite JSON value is not permitted: {value}")

    with Path(path).open("rb") as handle:
        raw = handle.read(MAX_PROMOTION_DECISION_BYTES + 1)
    if len(raw) > MAX_PROMOTION_DECISION_BYTES:
        raise ValueError("ranker promotion decision exceeds the safety limit")
    payload = json.loads(
        raw.decode("utf-8"),
        parse_constant=reject_nonfinite,
    )
    if not isinstance(payload, Mapping):
        raise TypeError("ranker promotion decision must be a JSON object")
    return RankerPromotionDecision.from_dict(payload)


def write_ranker_promotion_decision(
    decision: RankerPromotionDecision,
    path: str | Path,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    serialised = json.dumps(
        decision.to_dict(),
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(serialised)
            handle.write("\n")
            temporary_name = handle.name
        os.replace(temporary_name, target)
    finally:
        if temporary_name is not None:
            temporary = Path(temporary_name)
            if temporary.exists():
                temporary.unlink()
    return target
