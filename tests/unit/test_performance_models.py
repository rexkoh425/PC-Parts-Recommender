from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from training._common import sha256_file

from pc_build_recommender.evaluation.contracts import DataUseDeclaration
from pc_build_recommender.performance_models import (
    SYNTHETIC_GPU_FEATURE_COLUMNS,
    DatasetEvidence,
    ObservedPerformanceObservation,
    PerformanceModelConfig,
    PerformanceModelRegistry,
    WorkloadModelSpec,
    calibrate_prediction_intervals,
    estimate_peak_training_memory_mb,
    estimate_performance,
    load_performance_artifact,
    make_synthetic_performance_dataset,
    performance_frame_sha256,
    save_performance_artifact,
    seal_performance_artifact,
    split_performance_frame,
    train_performance_model,
    validate_performance_frame,
)
from pc_build_recommender.performance_models.training import (
    _base_lightgbm_parameters,
    _promotion_decision,
)


def _config(**overrides: object) -> PerformanceModelConfig:
    values: dict[str, object] = {
        "category": "gpu",
        "workload": "gaming_1440p",
        "feature_columns": SYNTHETIC_GPU_FEATURE_COLUMNS,
        "min_confident_test_rows": 10,
        "bootstrap_resamples": 100,
        "max_cpu_threads": 1,
        "lightgbm_params": {
            "n_estimators": 100,
            "learning_rate": 0.06,
            "num_leaves": 16,
        },
    }
    values.update(overrides)
    return PerformanceModelConfig(**values)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def synthetic_frame() -> pd.DataFrame:
    return make_synthetic_performance_dataset(n_families=36, variants_per_family=3)


@pytest.fixture(scope="module")
def trained_result(synthetic_frame: pd.DataFrame):  # type: ignore[no-untyped-def]
    return train_performance_model(synthetic_frame, _config())


def test_synthetic_starter_data_is_deterministic_and_non_promotable() -> None:
    first = make_synthetic_performance_dataset(n_families=18, variants_per_family=2)
    second = make_synthetic_performance_dataset(n_families=18, variants_per_family=2)

    pd.testing.assert_frame_equal(first, second)
    assert first["is_synthetic"].eq(True).all()  # noqa: E712
    assert first["eligible_for_external_claims"].eq(False).all()  # noqa: E712
    assert first["dataset_role"].eq("development_only_non_promotable").all()
    assert first.attrs["eligible_for_external_claims"] is False


def test_split_is_order_independent_and_keeps_families_together(
    synthetic_frame: pd.DataFrame,
) -> None:
    config = _config()
    first = split_performance_frame(synthetic_frame, config)
    shuffled = synthetic_frame.sample(frac=1.0, random_state=44).reset_index(drop=True)
    second = split_performance_frame(shuffled, config)

    first_assignments = first.set_index("product_id")["split"].to_dict()
    second_assignments = second.set_index("product_id")["split"].to_dict()
    assert first_assignments == second_assignments
    assert first.groupby("product_family")["split"].nunique().eq(1).all()
    assert set(first["split"]) == {"train", "validation", "calibration", "test"}


def test_training_is_reproducible_after_input_reordering(
    synthetic_frame: pd.DataFrame,
    trained_result,  # type: ignore[no-untyped-def]
) -> None:
    reordered = synthetic_frame.sample(frac=1.0, random_state=99).reset_index(drop=True)

    repeated = train_performance_model(reordered, _config())

    assert repeated.artifact.model_version == trained_result.artifact.model_version
    assert repeated.artifact.evaluations == trained_result.artifact.evaluations
    assert repeated.split_assignments == trained_result.split_assignments


def test_validation_rejects_mixed_scope_and_family_generation_leakage(
    synthetic_frame: pd.DataFrame,
) -> None:
    mixed_workload = synthetic_frame.copy()
    mixed_workload.loc[0, "workload"] = "local_ai"
    with pytest.raises(ValueError, match="configured value"):
        validate_performance_frame(mixed_workload, _config())

    spanning_family = synthetic_frame.copy()
    family = spanning_family.loc[0, "product_family"]
    family_rows = spanning_family.index[spanning_family["product_family"] == family]
    spanning_family.loc[family_rows[0], "hardware_generation"] = "another-generation"
    with pytest.raises(ValueError, match="exactly one hardware generation"):
        validate_performance_frame(spanning_family, _config())


def test_training_reports_baselines_metrics_and_provenance(trained_result) -> None:  # type: ignore[no-untyped-def]
    artifact = trained_result.artifact

    assert set(artifact.evaluations) == {"train_median", "ridge", "lightgbm"}
    assert artifact.evaluations["lightgbm"].test.sample_count == artifact.split_row_counts["test"]
    assert artifact.data_use.synthetic_rows == artifact.data_use.total_rows
    assert artifact.promotable is False
    assert artifact.precise_predictions_enabled is False
    assert any("synthetic" in reason for reason in artifact.promotion_block_reasons)
    assert artifact.requested_device == "cpu"
    assert artifact.actual_device == "cpu"
    assert artifact.device_fallback_reason is None
    assert len(artifact.model_version) == 64
    assert all(count > 0 for count in artifact.split_group_counts.values())
    assert set(artifact.feature_profiles) == set(SYNTHETIC_GPU_FEATURE_COLUMNS)
    assert artifact.allowed_missing_fraction == 0.0
    assert artifact.dataset_evidence.verified is False
    assert artifact.calibration.calibration_sample_count == artifact.split_row_counts["calibration"]
    assert artifact.grouped_test.development_group_overlap_count == 0
    assert artifact.test_uncertainty.group_count == artifact.split_group_counts["test"]
    assert artifact.estimated_peak_training_memory_mb <= artifact.config.max_training_memory_mb


def test_auto_device_request_records_acceleration_or_fallback() -> None:
    frame = make_synthetic_performance_dataset(n_families=24, variants_per_family=2)
    config = _config(lightgbm_params={"device_type": "auto", "n_estimators": 20})

    artifact = train_performance_model(frame, config).artifact

    assert artifact.requested_device == "auto"
    assert artifact.actual_device in {"cuda", "gpu", "cpu"}
    if artifact.actual_device != "cuda":
        assert artifact.device_fallback_reason


def test_observed_score_always_precedes_a_non_promotable_model(trained_result) -> None:  # type: ignore[no-untyped-def]
    estimate = estimate_performance(
        trained_result.artifact,
        observed_score=123.4,
        observed_source="https://benchmarks.example/result/1",
    )

    assert estimate.basis == "observed"
    assert estimate.decision == "observed_benchmark"
    assert estimate.score == pytest.approx(123.4)
    assert estimate.model_version is None
    assert estimate.supporting_sources == ("https://benchmarks.example/result/1",)


def test_synthetic_artifact_returns_relative_only_prediction(
    trained_result,
    synthetic_frame: pd.DataFrame,  # type: ignore[no-untyped-def]
) -> None:
    features = synthetic_frame.iloc[0].loc[list(SYNTHETIC_GPU_FEATURE_COLUMNS)]
    estimate = estimate_performance(trained_result.artifact, features)

    assert estimate.basis == "relative_only"
    assert estimate.decision == "model_not_promotion_eligible"
    assert estimate.score is None
    assert estimate.relative_score > 0
    assert estimate.reason is not None and "synthetic" in estimate.reason


def test_log_target_transform_evaluates_and_serves_native_benchmark_units(
    synthetic_frame: pd.DataFrame,
) -> None:
    artifact = train_performance_model(
        synthetic_frame,
        _config(target_transform="log1p"),
    ).artifact
    features = {
        name: float(synthetic_frame.iloc[0][name]) for name in SYNTHETIC_GPU_FEATURE_COLUMNS
    }
    raw_prediction = float(
        artifact.booster.predict(
            pd.DataFrame([features], columns=artifact.config.feature_columns),
            num_iteration=artifact.best_iteration,
        )[0]
    )

    estimate = estimate_performance(artifact, features)

    assert artifact.config.target_transform == "log1p"
    assert estimate.relative_score == pytest.approx(max(0.0, float(np.expm1(raw_prediction))))
    assert artifact.evaluations["lightgbm"].validation.mape_percent >= 0.0
    assert artifact.evaluations["lightgbm"].test.mape_percent >= 0.0


def test_config_rejects_an_unknown_target_transform() -> None:
    with pytest.raises(ValueError, match="target_transform"):
        _config(target_transform="square")


def test_legacy_artifact_config_defaults_to_identity_target_transform() -> None:
    payload = _config().to_dict()
    payload.pop("target_transform")

    restored = PerformanceModelConfig.from_dict(payload)

    assert restored.target_transform == "identity"


def test_confident_artifact_can_expose_a_predicted_score(
    trained_result,  # type: ignore[no-untyped-def]
) -> None:
    artifact = _precise_test_artifact(trained_result.artifact)
    features = {
        name: (profile.minimum + profile.maximum) / 2.0
        for name, profile in artifact.feature_profiles.items()
    }

    estimate = estimate_performance(artifact, features)

    assert estimate.basis == "predicted"
    assert estimate.decision == "precise_model_prediction"
    assert estimate.score == pytest.approx(estimate.relative_score)
    assert estimate.model_version == artifact.model_version
    assert estimate.lower_score is not None
    assert estimate.upper_score is not None
    assert estimate.lower_score <= estimate.score <= estimate.upper_score


def test_out_of_distribution_request_is_downgraded_even_for_confident_model(
    trained_result,  # type: ignore[no-untyped-def]
) -> None:
    artifact = _precise_test_artifact(trained_result.artifact)
    features = {
        name: (profile.minimum + profile.maximum) / 2.0
        for name, profile in artifact.feature_profiles.items()
    }
    first_feature = artifact.config.feature_columns[0]
    features[first_feature] = artifact.feature_profiles[first_feature].maximum + 1.0

    estimate = estimate_performance(artifact, features)

    assert estimate.basis == "relative_only"
    assert estimate.decision == "input_outside_training_contract"
    assert estimate.score is None
    assert estimate.confidence == "low"
    assert estimate.reason is not None and "outside training range" in estimate.reason


def test_excessive_missing_features_are_relative_only(
    trained_result,  # type: ignore[no-untyped-def]
) -> None:
    artifact = _precise_test_artifact(trained_result.artifact)
    features = {
        name: (profile.minimum + profile.maximum) / 2.0
        for name, profile in artifact.feature_profiles.items()
    }
    features[artifact.config.feature_columns[0]] = None

    estimate = estimate_performance(artifact, features)

    assert estimate.basis == "relative_only"
    assert estimate.decision == "input_outside_training_contract"
    assert estimate.reason is not None and "missing fraction" in estimate.reason


def test_artifact_round_trip_preserves_predictions(
    tmp_path,
    trained_result,
    synthetic_frame: pd.DataFrame,  # type: ignore[no-untyped-def]
) -> None:
    artifact_dir = save_performance_artifact(trained_result.artifact, tmp_path / "model")
    restored = load_performance_artifact(artifact_dir)
    manifest = json.loads((artifact_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    features = synthetic_frame.iloc[2].loc[list(SYNTHETIC_GPU_FEATURE_COLUMNS)]

    before = estimate_performance(trained_result.artifact, features)
    after = estimate_performance(restored, features)

    assert restored.model_version == trained_result.artifact.model_version
    assert tuple(restored.config.split_weights) == (
        "train",
        "validation",
        "calibration",
        "test",
    )
    restored_hash_frame = split_performance_frame(synthetic_frame, restored.config)
    assert (
        performance_frame_sha256(restored_hash_frame, restored.config)
        == restored.training_data_sha256
    )
    assert restored.evaluations == trained_result.artifact.evaluations
    assert restored.actual_device == trained_result.artifact.actual_device
    assert restored.dataset_evidence == trained_result.artifact.dataset_evidence
    assert restored.calibration == trained_result.artifact.calibration
    assert restored.grouped_test == trained_result.artifact.grouped_test
    assert restored.test_uncertainty == trained_result.artifact.test_uncertainty
    assert manifest["development_only"] is not (
        restored.promotable and restored.precise_predictions_enabled
    )
    assert manifest["promotion_blockers"] == list(restored.promotion_block_reasons)
    assert after.basis == before.basis
    assert after.decision == before.decision
    assert after.relative_score == pytest.approx(before.relative_score, rel=1e-12)


def _write_sealed_evidence(
    artifact_dir, artifact, *, prepared_frame_sha256: str | None = None
) -> None:  # type: ignore[no-untyped-def]
    split_hashes = artifact.split_group_hashes
    source_sha256 = "a" * 64
    (artifact_dir / "training_evidence.json").write_text(
        json.dumps(
            {
                "source_sha256": source_sha256,
                "prepared_frame_sha256": prepared_frame_sha256 or artifact.training_data_sha256,
                "dataset_manifest_sha256": artifact.dataset_evidence.manifest_sha256,
                "training_group_hashes": list(split_hashes["train"]),
                "validation_group_hashes": list(split_hashes["validation"]),
                "calibration_group_hashes": list(split_hashes["calibration"]),
                "internal_test_group_hashes": list(split_hashes["test"]),
                "development_group_hashes": list(artifact.development_group_hashes),
            }
        ),
        encoding="utf-8",
    )
    (artifact_dir / "training_report.json").write_text(
        json.dumps(
            {
                "input": {"sha256": source_sha256},
                "training_data_sha256": artifact.training_data_sha256,
                "model_version": artifact.model_version,
                "dataset_evidence": artifact.dataset_evidence.to_dict(),
            }
        ),
        encoding="utf-8",
    )


def test_sealed_artifact_binds_training_evidence_before_loading(
    tmp_path,
    trained_result,  # type: ignore[no-untyped-def]
) -> None:
    artifact_dir = save_performance_artifact(trained_result.artifact, tmp_path / "model")
    _write_sealed_evidence(artifact_dir, trained_result.artifact)

    seal_performance_artifact(
        artifact_dir,
        evidence_filenames=("training_evidence.json", "training_report.json"),
    )
    restored = load_performance_artifact(artifact_dir)
    manifest = json.loads((artifact_dir / "artifact_manifest.json").read_text(encoding="utf-8"))

    assert restored.model_version == trained_result.artifact.model_version
    assert manifest["schema_version"].endswith(".v2")
    assert manifest["evidence_files"] == ["training_evidence.json", "training_report.json"]


def test_sealed_artifact_rejects_evidence_for_another_prepared_frame(
    tmp_path,
    trained_result,  # type: ignore[no-untyped-def]
) -> None:
    artifact_dir = save_performance_artifact(trained_result.artifact, tmp_path / "model")
    _write_sealed_evidence(
        artifact_dir,
        trained_result.artifact,
        prepared_frame_sha256="0" * 64,
    )
    seal_performance_artifact(
        artifact_dir,
        evidence_filenames=("training_evidence.json", "training_report.json"),
    )

    with pytest.raises(ValueError, match="does not match the prepared training frame"):
        load_performance_artifact(artifact_dir)


def test_promotable_artifact_requires_dataset_and_training_evidence(
    tmp_path,
    trained_result,  # type: ignore[no-untyped-def]
) -> None:
    promotable = _precise_test_artifact(trained_result.artifact)
    artifact_dir = save_performance_artifact(promotable, tmp_path / "model")
    _write_sealed_evidence(artifact_dir, promotable)

    with pytest.raises(ValueError, match="require sealed dataset and training evidence"):
        seal_performance_artifact(
            artifact_dir,
            evidence_filenames=("training_evidence.json", "training_report.json"),
        )


def test_legacy_promotable_artifact_is_rejected_before_serving(
    tmp_path,
    trained_result,  # type: ignore[no-untyped-def]
) -> None:
    artifact_dir = save_performance_artifact(trained_result.artifact, tmp_path / "model")
    metadata_path = artifact_dir / "metadata.json"
    manifest_path = artifact_dir / "artifact_manifest.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["promotable"] = True
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["metadata.json"]["sha256"] = sha256_file(metadata_path)
    manifest["files"]["metadata.json"]["size_bytes"] = metadata_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="require an evidence-sealed v2 manifest"):
        load_performance_artifact(artifact_dir)


def test_artifact_loader_rejects_model_tampering(tmp_path, trained_result) -> None:  # type: ignore[no-untyped-def]
    artifact_dir = save_performance_artifact(trained_result.artifact, tmp_path / "model")
    model_path = artifact_dir / "model.txt"
    model_path.write_text(
        model_path.read_text(encoding="utf-8") + "\n# tampered\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match manifest"):
        load_performance_artifact(artifact_dir)


def test_artifact_loader_rejects_revoked_artifact(tmp_path, trained_result) -> None:  # type: ignore[no-untyped-def]
    artifact_dir = save_performance_artifact(trained_result.artifact, tmp_path / "model")
    manifest_path = artifact_dir / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["revoked"] = True
    manifest["revocation_reasons"] = ["invalid source target semantics"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="revoked: invalid source target semantics"):
        load_performance_artifact(artifact_dir)


def _precise_test_artifact(artifact):  # type: ignore[no-untyped-def]
    permissive_config = replace(
        artifact.config,
        min_confident_r2=-100.0,
        max_confident_mape_percent=10000.0,
        min_confident_test_rows=2,
        min_confident_test_groups=2,
        min_calibration_rows=2,
        min_calibration_groups=2,
        max_interval_coverage_shortfall=0.99,
        max_test_ood_fraction=1.0,
    )
    return replace(
        artifact,
        config=permissive_config,
        data_use=DataUseDeclaration.from_flags([False] * 30),
        dataset_evidence=DatasetEvidence(
            verified=True,
            eligible_for_promotion=True,
            manifest_sha256="0" * 64,
        ),
        confidence_level="high",
        precise_predictions_enabled=True,
        promotable=True,
        promotion_block_reasons=(),
    )


def test_calibration_is_independent_deterministic_and_reports_coverage() -> None:
    first = calibrate_prediction_intervals(
        [10.0, 20.0, 30.0, 40.0],
        [9.0, 18.0, 33.0, 36.0],
        ["a", "a", "b", "b"],
        [12.0, 22.0, 35.0],
        [11.0, 20.0, 32.0],
        alpha=0.10,
    )
    second = calibrate_prediction_intervals(
        [10.0, 20.0, 30.0, 40.0],
        [9.0, 18.0, 33.0, 36.0],
        ["a", "a", "b", "b"],
        [12.0, 22.0, 35.0],
        [11.0, 20.0, 32.0],
        alpha=0.10,
    )

    assert first == second
    assert first.calibration_group_count == 2
    assert first.absolute_error_quantile == pytest.approx(4.0)
    assert first.test_coverage == pytest.approx(1.0)
    assert 0.0 < first.test_coverage_lower_95 < first.test_coverage


def test_feature_contract_rejects_extra_inference_fields(trained_result) -> None:  # type: ignore[no-untyped-def]
    features = {
        name: (profile.minimum + profile.maximum) / 2.0
        for name, profile in trained_result.artifact.feature_profiles.items()
    }
    features["unregistered_feature"] = 1.0

    with pytest.raises(ValueError, match="unexpected columns"):
        estimate_performance(trained_result.artifact, features)


def test_registry_requires_exact_observed_cohort_before_skipping_model_lookup() -> None:
    spec = WorkloadModelSpec(
        category="cpu",
        workload="blender_4_0_0_junkshop_cpu_windows",
        feature_columns=("core_count", "thread_count"),
        metric="normalized_render_throughput",
        unit="score",
        higher_is_better=True,
        cohort=(("backend", "CPU"), ("scene", "junkshop")),
    )
    registry = PerformanceModelRegistry()
    registry.register_spec(spec)
    observed = ObservedPerformanceObservation(
        product_id="cpu-1",
        category=spec.category,
        workload=spec.workload,
        score=82.5,
        metric=spec.metric,
        unit=spec.unit,
        higher_is_better=True,
        source="https://opendata.blender.org/benchmarks/example",
        cohort=spec.cohort,
    )

    estimate = registry.estimate(
        category=spec.category,
        workload=spec.workload,
        product_id="cpu-1",
        observed=observed,
    )

    assert estimate.basis == "observed"
    assert estimate.score == pytest.approx(82.5)
    mismatched = replace(observed, cohort=(("backend", "CPU"), ("scene", "classroom")))
    with pytest.raises(ValueError, match="not comparable"):
        registry.estimate(
            category=spec.category,
            workload=spec.workload,
            product_id="cpu-1",
            observed=mismatched,
        )


def test_memory_budget_estimate_is_deterministic_and_fail_fast(
    synthetic_frame: pd.DataFrame,
) -> None:
    config = _config(max_training_memory_mb=64)
    first = estimate_peak_training_memory_mb(synthetic_frame, config)
    second = estimate_peak_training_memory_mb(synthetic_frame.copy(), config)

    assert first == pytest.approx(second)
    assert 0.0 < first < 64.0
    with pytest.raises(ValueError, match="at least 64"):
        replace(config, max_training_memory_mb=63)


def test_artifact_loader_rejects_promotion_metadata_tampering(
    tmp_path,
    trained_result,  # type: ignore[no-untyped-def]
) -> None:
    artifact_dir = save_performance_artifact(trained_result.artifact, tmp_path / "model")
    metadata_path = artifact_dir / "metadata.json"
    metadata_path.write_text(
        metadata_path.read_text(encoding="utf-8").replace(
            '"promotable": false', '"promotable": true'
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="metadata.json .*does not match manifest"):
        load_performance_artifact(artifact_dir)


def test_artifact_bytes_are_deterministic(tmp_path, trained_result) -> None:  # type: ignore[no-untyped-def]
    first = save_performance_artifact(trained_result.artifact, tmp_path / "first")
    second = save_performance_artifact(trained_result.artifact, tmp_path / "second")

    for filename in ("model.txt", "metadata.json", "artifact_manifest.json"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()


def test_gpu_parameters_are_capped_and_reproducibility_owned_by_trainer() -> None:
    config = _config(
        requested_device="cuda",
        max_cpu_threads=2,
        gpu_max_bin=31,
        lightgbm_params={"n_estimators": 20, "max_bin": 255},
    )

    parameters = _base_lightgbm_parameters(config, device="cuda")

    assert parameters["max_bin"] == 31
    assert parameters["n_jobs"] == 2
    assert parameters["deterministic"] is True
    assert parameters["random_state"] == config.split_seed
    assert parameters["data_random_seed"] == config.split_seed
    with pytest.raises(ValueError, match="managed by the trainer"):
        _base_lightgbm_parameters(
            _config(lightgbm_params={"n_jobs": 99}),
            device="cpu",
        )


def test_undercovered_intervals_block_promotion(trained_result) -> None:  # type: ignore[no-untyped-def]
    artifact = trained_result.artifact
    undercovered = replace(artifact.calibration, test_covered_count=0, test_coverage_lower_95=0.0)

    promotable, blockers, _ = _promotion_decision(
        config=artifact.config,
        data_use=DataUseDeclaration.from_flags([False] * artifact.data_use.total_rows),
        dataset_evidence=DatasetEvidence(
            verified=True,
            eligible_for_promotion=True,
            manifest_sha256="1" * 64,
        ),
        evaluations=artifact.evaluations,
        calibration=undercovered,
        grouped_test=artifact.grouped_test,
        uncertainty=artifact.test_uncertainty,
    )

    assert promotable is False
    assert any("prediction-interval coverage" in blocker for blocker in blockers)


def test_split_group_hashes_are_disjoint_and_integrity_bound(trained_result) -> None:  # type: ignore[no-untyped-def]
    artifact = trained_result.artifact
    split_sets = [set(values) for values in artifact.split_group_hashes.values()]

    for index, left in enumerate(split_sets):
        assert all(not left.intersection(right) for right in split_sets[index + 1 :])
    assert set(artifact.development_group_hashes) == set().union(*split_sets)
