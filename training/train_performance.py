"""Train one category/workload performance model with leakage-safe splits."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from shutil import copyfile
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from pc_build_recommender.performance_models import (
    DATASET_MANIFEST_FILENAME,
    SYNTHETIC_GPU_FEATURE_COLUMNS,
    TRAINING_EVIDENCE_FILENAME,
    TRAINING_REPORT_FILENAME,
    DatasetEvidence,
    PerformanceModelConfig,
    estimate_peak_training_memory_mb,
    save_performance_artifact,
    seal_performance_artifact,
    train_performance_model,
)
from training._common import (
    HostMemoryPreflight,
    comma_separated,
    portable_path_reference,
    print_json,
    require_host_memory_headroom,
    sha256_file,
    sha256_text,
    utc_now_iso,
    write_json,
)
from training.mlflow_tracking import (
    OptionalMLflowRun,
    add_mlflow_arguments,
    promotion_blocker_tags,
    tracking_config_from_args,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _json_object(value: str) -> dict[str, Any]:
    decoded = json.loads(value)
    if not isinstance(decoded, dict):
        raise argparse.ArgumentTypeError("expected a JSON object")
    return decoded


def _explicit_boolean_series(series: pd.Series, *, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype(bool)
    mapping = {"true": True, "false": False, "1": True, "0": False}
    normalized = series.astype(str).str.strip().str.casefold()
    unknown = sorted(set(normalized).difference(mapping))
    if unknown:
        raise ValueError(f"{name} must contain explicit booleans; found {unknown[:5]}")
    return normalized.map(mapping).astype(bool)


def _report_payload(
    result: Any,
    *,
    input_path: Path,
    artifact_path: Path,
    host_memory_preflight: HostMemoryPreflight,
) -> dict[str, Any]:
    artifact = result.artifact
    return {
        "created_at": utc_now_iso(),
        "task": "workload_performance_regression",
        "category": artifact.config.category,
        "workload": artifact.config.workload,
        "input": {
            "path": portable_path_reference(input_path, workspace_root=REPOSITORY_ROOT),
            "sha256": sha256_file(input_path),
        },
        "artifact_path": portable_path_reference(artifact_path, workspace_root=REPOSITORY_ROOT),
        "model_version": artifact.model_version,
        "training_data_sha256": artifact.training_data_sha256,
        "feature_columns": list(artifact.config.feature_columns),
        "split_group_counts": artifact.split_group_counts,
        "split_row_counts": artifact.split_row_counts,
        "models": {
            name: evaluation.to_dict() for name, evaluation in sorted(artifact.evaluations.items())
        },
        "synthetic_data": artifact.data_use.to_dict(),
        "dataset_evidence": artifact.dataset_evidence.to_dict(),
        "calibration": artifact.calibration.to_dict(),
        "grouped_test": artifact.grouped_test.to_dict(),
        "test_uncertainty": artifact.test_uncertainty.to_dict(),
        "resources": {
            "estimated_peak_training_memory_mb": artifact.estimated_peak_training_memory_mb,
            "configured_max_training_memory_mb": artifact.config.max_training_memory_mb,
            "max_cpu_threads": artifact.config.max_cpu_threads,
            "gpu_max_bin": artifact.config.gpu_max_bin,
            "host_memory_preflight": host_memory_preflight.to_dict(),
        },
        "promotion": {
            "eligible": artifact.promotable,
            "precise_predictions_enabled": artifact.precise_predictions_enabled,
            "confidence": artifact.confidence_level,
            "block_reasons": list(artifact.promotion_block_reasons),
        },
        "device": {
            "requested": artifact.requested_device,
            "actual": artifact.actual_device,
            "fallback_reason": artifact.device_fallback_reason,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="CSV with explicit provenance")
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        help="verified preparation manifest; omitting it permanently blocks promotion",
    )
    parser.add_argument("--category", default="gpu")
    parser.add_argument("--workload", default="gaming_1440p")
    parser.add_argument(
        "--features",
        default=",".join(SYNTHETIC_GPU_FEATURE_COLUMNS),
        help="comma-separated numeric feature columns",
    )
    parser.add_argument("--target-column", default="target_score")
    parser.add_argument(
        "--target-transform",
        choices=("identity", "log1p"),
        default="identity",
        help=(
            "learner target scale; metrics, intervals, and served predictions are always "
            "converted back to the native benchmark unit"
        ),
    )
    parser.add_argument("--product-id-column", default="product_id")
    parser.add_argument("--family-column", default="product_family")
    parser.add_argument("--generation-column", default="hardware_generation")
    parser.add_argument("--synthetic-column", default="is_synthetic")
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--device", choices=("auto", "cpu", "gpu", "cuda"), default="auto")
    parser.add_argument(
        "--strict-device",
        action="store_true",
        help="fail instead of falling back when the requested learner is unavailable",
    )
    parser.add_argument("--lightgbm-params", type=_json_object, default={})
    parser.add_argument("--minimum-test-rows", type=int, default=20)
    parser.add_argument("--minimum-test-groups", type=int, default=10)
    parser.add_argument("--minimum-r2", type=float, default=0.85)
    parser.add_argument("--maximum-mape-percent", type=float, default=12.0)
    parser.add_argument("--prediction-interval-alpha", type=float, default=0.10)
    parser.add_argument("--minimum-calibration-rows", type=int, default=20)
    parser.add_argument("--minimum-calibration-groups", type=int, default=10)
    parser.add_argument("--maximum-coverage-shortfall", type=float, default=0.10)
    parser.add_argument("--maximum-test-ood-fraction", type=float, default=0.20)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-confidence-level", type=float, default=0.95)
    parser.add_argument("--minimum-baseline-mape-improvement-percent", type=float, default=0.0)
    parser.add_argument("--max-training-memory-mb", type=int, default=2048)
    parser.add_argument(
        "--max-host-used-gb",
        type=float,
        default=55.0,
        help=(
            "refuse training when current host RAM plus the conservative model allocation "
            "reaches this cap"
        ),
    )
    parser.add_argument(
        "--minimum-free-memory-mb",
        type=float,
        default=1024.0,
        help="minimum host RAM that must remain after the conservative model allocation",
    )
    parser.add_argument("--max-cpu-threads", type=int, default=4)
    parser.add_argument("--gpu-max-bin", type=int, default=63)
    parser.add_argument(
        "--allow-synthetic-diagnostics",
        action="store_true",
        help="train a permanently non-promotable smoke-test artifact on synthetic rows",
    )
    add_mlflow_arguments(parser, default_experiment="pcbr-performance-models")
    return parser


def _dataset_evidence_from_manifest(
    manifest_path: Path | None,
    *,
    input_path: Path,
    frame: pd.DataFrame,
    config: PerformanceModelConfig,
) -> DatasetEvidence:
    if manifest_path is None:
        return DatasetEvidence.unverified()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not str(payload.get("schema_version", "")):
        raise ValueError("dataset manifest must be a versioned JSON object")
    output = payload.get("output")
    cohort = payload.get("selected_cohort")
    target = payload.get("target")
    promotion = payload.get("promotion")
    if not all(isinstance(value, dict) for value in (output, cohort, target, promotion)):
        raise ValueError("dataset manifest is missing output/cohort/target/promotion evidence")
    assert isinstance(output, dict)
    assert isinstance(cohort, dict)
    assert isinstance(target, dict)
    assert isinstance(promotion, dict)
    actual_sha256 = sha256_file(input_path)
    if output.get("sha256") != actual_sha256:
        raise ValueError("dataset manifest CSV SHA-256 does not match the training input")
    if int(output.get("rows", -1)) != len(frame) or int(payload.get("row_count", -1)) != len(frame):
        raise ValueError("dataset manifest row count does not match the training input")
    if tuple(payload.get("features", ())) != config.feature_columns:
        raise ValueError("dataset manifest feature order does not match the model config")
    if cohort.get("category") != config.category or cohort.get("workload") != config.workload:
        raise ValueError("dataset manifest category/workload does not match the model config")
    if target.get("column") != config.target_column or target.get("higher_is_better") is not True:
        raise ValueError("dataset manifest target contract is incompatible with the regressor")

    blockers = [str(reason) for reason in promotion.get("block_reasons", ())]
    eligible = promotion.get("eligible")
    if not isinstance(eligible, bool):
        raise ValueError("dataset manifest promotion eligibility must be an explicit boolean")
    eligibility_column = "eligible_for_external_claims"
    if eligibility_column not in frame:
        blockers.append("row-level external-claim eligibility was not declared")
    else:
        row_eligibility = _explicit_boolean_series(
            frame[eligibility_column], name=eligibility_column
        )
        if not bool(row_eligibility.all()):
            blockers.append("one or more training rows are ineligible for external claims")
    if not eligible and not blockers:
        blockers.append("dataset manifest marks this dataset non-promotable")
    promotion_eligible = eligible and not blockers
    return DatasetEvidence(
        verified=True,
        eligible_for_promotion=promotion_eligible,
        manifest_sha256=sha256_file(manifest_path),
        block_reasons=tuple(dict.fromkeys(blockers)),
    )


def _track_training(
    args: argparse.Namespace,
    report: dict[str, Any],
    *,
    artifact_path: Path,
    report_path: Path,
    config: PerformanceModelConfig,
    finalize_artifact: Callable[[], None],
) -> None:
    blockers = list(report["promotion"]["block_reasons"])
    feature_contract = sha256_text("\n".join(config.feature_columns))
    with OptionalMLflowRun(tracking_config_from_args(args)) as tracking:
        tracking.log_params(
            {
                "task": report["task"],
                "category": report["category"],
                "workload": report["workload"],
                "dataset": report["input"],
                "training_data_sha256": report["training_data_sha256"],
                "model_version": report["model_version"],
                "feature_contract_sha256": feature_contract,
                "feature_columns": report["feature_columns"],
                "split_seed": config.split_seed,
                "split_weights": config.split_weights,
                "split_group_counts": report["split_group_counts"],
                "split_row_counts": report["split_row_counts"],
                "confidence_thresholds": {
                    "minimum_r2": config.min_confident_r2,
                    "maximum_mape_percent": config.max_confident_mape_percent,
                    "minimum_test_rows": config.min_confident_test_rows,
                },
                "lightgbm": config.lightgbm_params,
                "device": report["device"],
            }
        )
        metrics: dict[str, Any] = {}
        for model_name, evaluation in report["models"].items():
            for split_name in ("validation", "test"):
                metrics.update(
                    {
                        f"{model_name}.{split_name}.{name}": value
                        for name, value in evaluation[split_name].items()
                    }
                )
        tracking.log_metrics(metrics)
        tracking.log_tags(
            {
                "task": report["task"],
                "model.version": report["model_version"],
                "feature.version": f"sha256:{feature_contract}",
                "device.requested": report["device"]["requested"],
                "device.actual": report["device"]["actual"],
                "device.fallback": report["device"]["fallback_reason"],
                **promotion_blocker_tags(blockers),
            }
        )
        tracking.log_dict(
            {
                "eligible": report["promotion"]["eligible"],
                "precise_predictions_enabled": report["promotion"]["precise_predictions_enabled"],
                "confidence": report["promotion"]["confidence"],
                "block_reasons": blockers,
            },
            "evidence/promotion-gate.json",
        )
        report["mlflow_tracking"] = tracking.describe()
        write_json(report_path, report)
        finalize_artifact()
        tracking.log_native_artifacts(artifact_path, artifact_path="performance-model")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    frame = pd.read_csv(args.input)
    if args.synthetic_column not in frame:
        raise ValueError(
            f"{args.synthetic_column!r} is required; model data needs row-level provenance"
        )
    frame[args.synthetic_column] = _explicit_boolean_series(
        frame[args.synthetic_column], name=args.synthetic_column
    )
    if "eligible_for_external_claims" in frame:
        frame["eligible_for_external_claims"] = _explicit_boolean_series(
            frame["eligible_for_external_claims"], name="eligible_for_external_claims"
        )
    synthetic_count = int(frame[args.synthetic_column].sum())
    if synthetic_count and not args.allow_synthetic_diagnostics:
        raise ValueError(
            f"input contains {synthetic_count} synthetic rows; pass "
            "--allow-synthetic-diagnostics only for a non-promotable smoke test"
        )

    lightgbm_params = dict(args.lightgbm_params)
    lightgbm_params.pop("device_type", None)
    config = PerformanceModelConfig(
        category=args.category,
        workload=args.workload,
        feature_columns=comma_separated(args.features),
        target_column=args.target_column,
        target_transform=args.target_transform,
        product_id_column=args.product_id_column,
        family_column=args.family_column,
        generation_column=args.generation_column,
        synthetic_column=args.synthetic_column,
        split_seed=args.seed,
        min_confident_r2=args.minimum_r2,
        max_confident_mape_percent=args.maximum_mape_percent,
        min_confident_test_rows=args.minimum_test_rows,
        min_confident_test_groups=args.minimum_test_groups,
        prediction_interval_alpha=args.prediction_interval_alpha,
        min_calibration_rows=args.minimum_calibration_rows,
        min_calibration_groups=args.minimum_calibration_groups,
        max_interval_coverage_shortfall=args.maximum_coverage_shortfall,
        max_test_ood_fraction=args.maximum_test_ood_fraction,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_confidence_level=args.bootstrap_confidence_level,
        minimum_baseline_mape_improvement_percent=(args.minimum_baseline_mape_improvement_percent),
        requested_device=args.device,
        allow_device_fallback=not args.strict_device,
        max_training_memory_mb=args.max_training_memory_mb,
        max_cpu_threads=args.max_cpu_threads,
        gpu_max_bin=args.gpu_max_bin,
        lightgbm_params=lightgbm_params,
    )
    dataset_evidence = _dataset_evidence_from_manifest(
        args.dataset_manifest,
        input_path=args.input,
        frame=frame,
        config=config,
    )
    host_memory_preflight = require_host_memory_headroom(
        max_used_gib=args.max_host_used_gb,
        estimated_additional_mib=estimate_peak_training_memory_mb(frame, config),
        minimum_free_mib=args.minimum_free_memory_mb,
    )
    result = train_performance_model(frame, config, dataset_evidence=dataset_evidence)
    artifact_path = save_performance_artifact(result.artifact, args.artifact_dir)
    training_families = {
        str(family)
        for product_id, family in zip(
            frame[config.product_id_column], frame[config.family_column], strict=True
        )
        if result.split_assignments[str(product_id)] == "train"
    }
    validation_families = {
        str(family)
        for product_id, family in zip(
            frame[config.product_id_column], frame[config.family_column], strict=True
        )
        if result.split_assignments[str(product_id)] == "validation"
    }
    calibration_families = {
        str(family)
        for product_id, family in zip(
            frame[config.product_id_column], frame[config.family_column], strict=True
        )
        if result.split_assignments[str(product_id)] == "calibration"
    }
    test_families = {
        str(family)
        for product_id, family in zip(
            frame[config.product_id_column], frame[config.family_column], strict=True
        )
        if result.split_assignments[str(product_id)] == "test"
    }
    all_development_families = (
        training_families | validation_families | calibration_families | test_families
    )
    write_json(
        artifact_path / TRAINING_EVIDENCE_FILENAME,
        {
            "source_sha256": sha256_file(args.input),
            "prepared_frame_sha256": result.artifact.training_data_sha256,
            "leakage_unit": config.family_column,
            "training_group_hashes": sorted(map(sha256_text, training_families)),
            "validation_group_hashes": sorted(map(sha256_text, validation_families)),
            "calibration_group_hashes": sorted(map(sha256_text, calibration_families)),
            "internal_test_group_hashes": sorted(map(sha256_text, test_families)),
            "development_group_hashes": sorted(map(sha256_text, all_development_families)),
            "dataset_manifest_sha256": result.artifact.dataset_evidence.manifest_sha256,
        },
    )
    sealed_evidence_filenames = [TRAINING_EVIDENCE_FILENAME, TRAINING_REPORT_FILENAME]
    if args.dataset_manifest is not None:
        sealed_manifest_path = artifact_path / DATASET_MANIFEST_FILENAME
        copyfile(args.dataset_manifest, sealed_manifest_path)
        if sha256_file(sealed_manifest_path) != result.artifact.dataset_evidence.manifest_sha256:
            raise RuntimeError(
                "sealed dataset manifest digest does not match verified dataset evidence"
            )
        sealed_evidence_filenames.append(DATASET_MANIFEST_FILENAME)
    report = _report_payload(
        result,
        input_path=args.input,
        artifact_path=artifact_path,
        host_memory_preflight=host_memory_preflight,
    )
    artifact_report_path = artifact_path / TRAINING_REPORT_FILENAME

    def finalize_artifact() -> None:
        seal_performance_artifact(
            artifact_path,
            evidence_filenames=sealed_evidence_filenames,
        )

    _track_training(
        args,
        report,
        artifact_path=artifact_path,
        report_path=artifact_report_path,
        config=config,
        finalize_artifact=finalize_artifact,
    )
    if args.report is not None and args.report.resolve() != artifact_report_path.resolve():
        write_json(args.report, report)
    print_json(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
