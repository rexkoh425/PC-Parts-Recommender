"""Evaluate a persisted performance model on an external frozen CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from pc_build_recommender.evaluation.contracts import DataUseDeclaration
from pc_build_recommender.performance_models import (
    calculate_regression_metrics,
    grouped_bootstrap_uncertainty,
    grouped_test_diagnostics,
    load_performance_artifact,
    performance_frame_sha256,
    split_performance_frame,
    validate_performance_frame,
)
from training._common import print_json, sha256_file, sha256_text, utc_now_iso, write_json
from training.train_performance import (
    _dataset_evidence_from_manifest,
    _explicit_boolean_series,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        help="verified external-set manifest; omitting it blocks promotion evidence",
    )
    parser.add_argument(
        "--include-synthetic-diagnostics",
        action="store_true",
        help="include synthetic rows but permanently mark the resulting metrics non-promotable",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    artifact = load_performance_artifact(args.artifact_dir)
    frame = pd.read_csv(args.input)
    synthetic_column = artifact.config.synthetic_column
    if synthetic_column not in frame:
        raise ValueError(f"{synthetic_column!r} is required; evaluation needs row-level provenance")
    frame[synthetic_column] = _explicit_boolean_series(
        frame[synthetic_column], name=synthetic_column
    )
    prepared = validate_performance_frame(frame, artifact.config)
    expected_unit = artifact.config.family_column
    development_hashes = set(artifact.development_group_hashes)
    evaluation_hashes = {
        sha256_text(str(value)) for value in prepared[expected_unit].tolist()
    }
    training_overlap_count = len(development_hashes & evaluation_hashes)
    training_overlap_fraction = training_overlap_count / max(1, len(evaluation_hashes))
    evaluation_data_sha256: str | None = None
    evaluation_reuses_training_rows = False
    try:
        split_for_hash = split_performance_frame(prepared, artifact.config)
        evaluation_data_sha256 = performance_frame_sha256(split_for_hash, artifact.config)
        evaluation_reuses_training_rows = (
            evaluation_data_sha256 == artifact.training_data_sha256
        )
    except ValueError:
        # A small external set can be valid for metrics even when it is too small to
        # reproduce the three-way training split used by the semantic artifact hash.
        pass
    flags = prepared[synthetic_column].astype(bool).tolist()
    data_use = DataUseDeclaration.from_flags(
        flags,
        include_synthetic=args.include_synthetic_diagnostics,
    )
    evaluated = (
        prepared
        if args.include_synthetic_diagnostics
        else prepared.loc[~prepared[synthetic_column]].copy()
    )
    if len(evaluated) < 2:
        raise ValueError(
            "fewer than two reportable rows remain; use --include-synthetic-diagnostics "
            "only for smoke-test metrics"
        )
    predictions = np.asarray(
        artifact.booster.predict(
            evaluated.loc[:, artifact.config.feature_columns],
            num_iteration=artifact.best_iteration,
        ),
        dtype=float,
    )
    metrics = calculate_regression_metrics(
        evaluated[artifact.config.target_column].to_numpy(dtype=float), predictions
    )
    external_evidence = _dataset_evidence_from_manifest(
        args.dataset_manifest,
        input_path=args.input,
        frame=frame,
        config=artifact.config,
    )
    external_groups = evaluated[artifact.config.family_column].astype(str).tolist()
    external_group_count = len(set(external_groups))
    uncertainty = (
        grouped_bootstrap_uncertainty(
            evaluated[artifact.config.target_column].to_numpy(dtype=float),
            predictions,
            external_groups,
            confidence_level=artifact.config.bootstrap_confidence_level,
            n_resamples=artifact.config.bootstrap_resamples,
            seed=artifact.config.split_seed,
        )
        if external_group_count >= 2
        else None
    )
    diagnostics = grouped_test_diagnostics(
        test_frame=evaluated,
        observed=evaluated[artifact.config.target_column].to_numpy(dtype=float),
        predicted=predictions,
        group_column=artifact.config.family_column,
        development_groups=set(),
        feature_columns=artifact.config.feature_columns,
        feature_profiles=artifact.feature_profiles,
    )
    interval_radius = artifact.calibration.absolute_error_quantile
    actual = evaluated[artifact.config.target_column].to_numpy(dtype=float)
    lower = np.maximum(0.0, predictions - interval_radius)
    upper = np.maximum(0.0, predictions + interval_radius)
    external_coverage = float(np.mean((actual >= lower) & (actual <= upper)))
    minimum_coverage = (
        artifact.calibration.nominal_coverage
        - artifact.config.max_interval_coverage_shortfall
    )
    promotion_eligible = (
        artifact.promotable
        and data_use.eligible_for_reported_metrics
        and external_evidence.eligible_for_promotion
        and not evaluation_reuses_training_rows
        and training_overlap_count == 0
        and metrics.sample_count >= artifact.config.min_confident_test_rows
        and external_group_count >= artifact.config.min_confident_test_groups
        and metrics.r2 >= artifact.config.min_confident_r2
        and metrics.mape_percent <= artifact.config.max_confident_mape_percent
        and uncertainty is not None
        and uncertainty.r2_lower >= artifact.config.min_confident_r2
        and uncertainty.mape_percent_upper <= artifact.config.max_confident_mape_percent
        and external_coverage >= minimum_coverage
        and diagnostics.outside_training_envelope_fraction
        <= artifact.config.max_test_ood_fraction
    )
    blockers: list[str] = []
    if not artifact.promotable:
        blockers.extend(artifact.promotion_block_reasons)
    if reason := data_use.reporting_block_reason:
        blockers.append(reason)
    blockers.extend(external_evidence.block_reasons)
    if evaluation_reuses_training_rows:
        blockers.append("evaluation rows are identical to the artifact training dataset")
    if training_overlap_count:
        blockers.append(
            f"evaluation overlaps {training_overlap_count} artifact development families"
        )
    if metrics.sample_count < artifact.config.min_confident_test_rows:
        blockers.append("external evaluation has too few rows")
    if external_group_count < artifact.config.min_confident_test_groups:
        blockers.append("external evaluation has too few distinct leakage groups")
    if metrics.r2 < artifact.config.min_confident_r2:
        blockers.append("external R2 is below the promotion threshold")
    if metrics.mape_percent > artifact.config.max_confident_mape_percent:
        blockers.append("external MAPE exceeds the promotion threshold")
    if uncertainty is not None:
        if uncertainty.r2_lower < artifact.config.min_confident_r2:
            blockers.append("external grouped-bootstrap R2 lower bound is below threshold")
        if uncertainty.mape_percent_upper > artifact.config.max_confident_mape_percent:
            blockers.append("external grouped-bootstrap MAPE upper bound exceeds threshold")
    if external_coverage < minimum_coverage:
        blockers.append("external prediction-interval coverage is below threshold")
    if (
        diagnostics.outside_training_envelope_fraction
        > artifact.config.max_test_ood_fraction
    ):
        blockers.append("external feature-envelope OOD fraction exceeds threshold")
    report = {
        "created_at": utc_now_iso(),
        "task": "workload_performance_external_evaluation",
        "model_version": artifact.model_version,
        "category": artifact.config.category,
        "workload": artifact.config.workload,
        "input": {
            "path": str(args.input.resolve()),
            "sha256": sha256_file(args.input),
            "semantic_sha256": evaluation_data_sha256,
            "reuses_training_rows": evaluation_reuses_training_rows,
            "training_group_overlap_count": training_overlap_count,
            "training_group_overlap_fraction": training_overlap_fraction,
        },
        "metrics": metrics.to_dict(),
        "grouped_uncertainty": uncertainty.to_dict() if uncertainty is not None else None,
        "grouped_test": diagnostics.to_dict(),
        "prediction_interval": {
            "nominal_coverage": artifact.calibration.nominal_coverage,
            "empirical_coverage": external_coverage,
            "minimum_required_coverage": minimum_coverage,
            "absolute_error_quantile": interval_radius,
        },
        "dataset_evidence": external_evidence.to_dict(),
        "synthetic_data": data_use.to_dict(),
        "promotion": {
            "eligible": promotion_eligible,
            "block_reasons": sorted(set(blockers)),
        },
    }
    report_path = args.report or args.artifact_dir / "external_evaluation.json"
    write_json(report_path, report)
    print_json(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
    print("DEBUG", locals())  # noqa
