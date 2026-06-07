from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from pc_build_recommender.domain import ComponentKind
from pc_build_recommender.optimizer import BuildOptimizer
from pc_build_recommender.optimizer.evaluation import (
    CLAIM_VALID_BUILD_TARGET,
    MAX_SCENARIO_COUNT,
    GeneratedOptimizerEvaluation,
    OptimizerEvaluationConfig,
    OptimizerEvaluationError,
    assess_10k_valid_build_claim,
    generate_optimizer_problem,
    independently_validate_solution,
    load_optimizer_evaluation,
    main,
    run_optimizer_evaluation,
    write_optimizer_evaluation,
)


def _artifact_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_bounded_evaluation_invokes_optimizer_and_retains_checked_outputs() -> None:
    report = run_optimizer_evaluation(
        OptimizerEvaluationConfig(scenario_count=6, solver_time_limit_seconds=0.25)
    )

    report.verify()
    assert report.payload["evaluation_passed"] is True
    assert report.payload["optimizer_invocation_count"] == 6
    assert report.payload["optimizer_output_count"] == 6
    assert report.payload["independently_checked_output_count"] == 6
    assert report.payload["independently_valid_output_count"] == 6
    assert report.payload["invalid_output_count"] == 0
    assert report.payload["unique_output_count"] == 6
    assert report.payload["result_status_counts"] == {
        "FEASIBLE": 0,
        "INFEASIBLE": 0,
        "MODEL_INVALID": 0,
        "OPTIMAL": 6,
        "UNKNOWN": 0,
    }
    assert report.payload["retained_scenario_record_count"] == 6
    assert report.payload["retained_output_record_count"] == 6
    records = report.payload["scenario_records"]
    assert isinstance(records, list)
    first_output = records[0]["outputs"][0]
    compatibility = first_output["versioned_compatibility"]
    assert compatibility["rule_version"] == "compat_v2"
    assert compatibility["rule_result_count"] > 8
    assert compatibility["status_counts"]["FAIL"] == 0
    assert compatibility["status_counts"]["UNKNOWN"] == 0
    claim = report.payload["claim_assessment"]
    assert isinstance(claim, dict)
    assert claim["eligible"] is False
    assert str(CLAIM_VALID_BUILD_TARGET) in str(claim["blockers"])


def test_evaluation_distinguishes_confirmed_infeasible_requests() -> None:
    report = run_optimizer_evaluation(
        OptimizerEvaluationConfig(
            scenario_count=6,
            infeasible_every=2,
            solver_time_limit_seconds=0.25,
        )
    )

    assert report.payload["evaluation_passed"] is True
    assert report.payload["result_status_counts"] == {
        "FEASIBLE": 0,
        "INFEASIBLE": 3,
        "MODEL_INVALID": 0,
        "OPTIMAL": 3,
        "UNKNOWN": 0,
    }
    assert report.payload["optimizer_output_count"] == 3
    assert report.payload["infeasibility_oracle_confirmed_count"] == 3


def test_multiple_outputs_are_independently_checked_and_diverse() -> None:
    report = run_optimizer_evaluation(
        OptimizerEvaluationConfig(
            scenario_count=2,
            candidates_per_category=2,
            solutions_per_scenario=3,
            solver_time_limit_seconds=0.25,
        )
    )

    report.verify()
    assert report.payload["optimizer_output_count"] == 6
    assert report.payload["independently_valid_output_count"] == 6
    assert report.payload["unique_output_count"] == 6


def test_independent_oracle_catches_tampered_optimizer_output() -> None:
    config = OptimizerEvaluationConfig(scenario_count=1, solver_time_limit_seconds=0.25)
    _, problem = generate_optimizer_problem(config, scenario_index=0)
    solution = BuildOptimizer().optimize(problem, max_solutions=1).solutions[0]

    wrong_total = replace(solution, total_price_cents=solution.total_price_cents + 1)
    assert "reported acquisition total" in " ".join(
        independently_validate_solution(problem, wrong_total)
    )

    wrong_objective = replace(solution, objective_value=solution.objective_value + 1)
    assert "reported objective value" in " ".join(
        independently_validate_solution(problem, wrong_objective)
    )

    selected = dict(solution.selected)
    selected[ComponentKind.GPU] = selected[ComponentKind.CPU]
    wrong_category = replace(solution, selected=selected)
    category_errors = " ".join(independently_validate_solution(problem, wrong_category))
    assert "gpu key contains a cpu candidate" in category_errors


def test_claim_requires_all_10k_outputs_to_be_checked_valid_and_retained() -> None:
    eligible = assess_10k_valid_build_claim(
        optimizer_output_count=CLAIM_VALID_BUILD_TARGET,
        independently_checked_output_count=CLAIM_VALID_BUILD_TARGET,
        independently_valid_output_count=CLAIM_VALID_BUILD_TARGET,
        unique_output_count=CLAIM_VALID_BUILD_TARGET,
        retained_output_record_count=CLAIM_VALID_BUILD_TARGET,
        invalid_output_count=0,
        evaluation_passed=True,
        records_verified=True,
    )
    assert eligible.eligible
    assert not eligible.blockers

    missing_record = assess_10k_valid_build_claim(
        optimizer_output_count=CLAIM_VALID_BUILD_TARGET,
        independently_checked_output_count=CLAIM_VALID_BUILD_TARGET,
        independently_valid_output_count=CLAIM_VALID_BUILD_TARGET,
        unique_output_count=CLAIM_VALID_BUILD_TARGET,
        retained_output_record_count=CLAIM_VALID_BUILD_TARGET - 1,
        invalid_output_count=0,
        evaluation_passed=True,
        records_verified=True,
    )
    assert not missing_record.eligible
    assert any("retained" in blocker for blocker in missing_record.blockers)


def test_content_addressed_optimizer_report_round_trip(tmp_path: Path) -> None:
    report = run_optimizer_evaluation(
        OptimizerEvaluationConfig(scenario_count=3, solver_time_limit_seconds=0.25)
    )

    path = write_optimizer_evaluation(report, tmp_path)
    loaded = load_optimizer_evaluation(path)

    assert loaded == report
    assert report.artifact_sha256[:16] in path.name


def test_config_and_cli_reject_runs_above_hard_bound(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="scenario_count"):
        OptimizerEvaluationConfig(scenario_count=MAX_SCENARIO_COUNT + 1)

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--count",
                str(MAX_SCENARIO_COUNT + 1),
                "--output-dir",
                str(tmp_path),
            ]
        )
    assert exc_info.value.code == 2


def test_report_verification_rejects_claim_decoupled_from_records() -> None:
    report = run_optimizer_evaluation(
        OptimizerEvaluationConfig(scenario_count=1, solver_time_limit_seconds=0.25)
    )
    payload = dict(report.payload)
    raw_claim = payload["claim_assessment"]
    assert isinstance(raw_claim, dict)
    claim = dict(raw_claim)
    claim["eligible"] = True
    payload["claim_assessment"] = claim

    forged = GeneratedOptimizerEvaluation(payload=payload, artifact_sha256=report.artifact_sha256)
    with pytest.raises(Exception, match="hash mismatch"):
        forged.verify()


def test_report_verification_rejects_rehashed_source_digest_tampering() -> None:
    report = run_optimizer_evaluation(
        OptimizerEvaluationConfig(scenario_count=1, solver_time_limit_seconds=0.25)
    )
    payload = dict(report.payload)
    raw_source_hashes = payload["source_sha256"]
    assert isinstance(raw_source_hashes, dict)
    source_hashes = dict(raw_source_hashes)
    source_hashes["optimizer_engine"] = "0" * 64
    payload["source_sha256"] = source_hashes
    forged = GeneratedOptimizerEvaluation(
        payload=payload,
        artifact_sha256=_artifact_sha256(payload),
    )

    with pytest.raises(
        OptimizerEvaluationError,
        match="does not match current implementation sources: optimizer_engine",
    ):
        forged.verify()


def test_report_verification_rejects_untracked_source_digest() -> None:
    report = run_optimizer_evaluation(
        OptimizerEvaluationConfig(scenario_count=1, solver_time_limit_seconds=0.25)
    )
    payload = dict(report.payload)
    raw_source_hashes = payload["source_sha256"]
    assert isinstance(raw_source_hashes, dict)
    source_hashes = dict(raw_source_hashes)
    source_hashes["untracked_source"] = "0" * 64
    payload["source_sha256"] = source_hashes
    forged = GeneratedOptimizerEvaluation(
        payload=payload,
        artifact_sha256=_artifact_sha256(payload),
    )

    with pytest.raises(
        OptimizerEvaluationError,
        match="exactly the required implementation sources",
    ):
        forged.verify()
