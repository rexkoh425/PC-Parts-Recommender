"""Safe, non-pickle persistence for LightGBM performance artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import lightgbm as lgb

from pc_build_recommender.evaluation.contracts import DataUseDeclaration
from pc_build_recommender.evaluation.manifest import sha256_file

from .contracts import (
    ARTIFACT_SCHEMA_VERSION,
    DatasetEvidence,
    FeatureProfile,
    GroupedTestDiagnostics,
    ModelEvaluation,
    PerformanceModelArtifact,
    PerformanceModelConfig,
    PredictionIntervalCalibration,
    RegressionUncertainty,
)

MODEL_FILENAME = "model.txt"
METADATA_FILENAME = "metadata.json"
ARTIFACT_MANIFEST_FILENAME = "artifact_manifest.json"
LEGACY_ARTIFACT_MANIFEST_SCHEMA_VERSION = "pc-build-recommender.performance-artifact-manifest.v1"
ARTIFACT_MANIFEST_SCHEMA_VERSION = "pc-build-recommender.performance-artifact-manifest.v2"
TRAINING_EVIDENCE_FILENAME = "training_evidence.json"
TRAINING_REPORT_FILENAME = "training_report.json"
DATASET_MANIFEST_FILENAME = "dataset_manifest.json"
REQUIRED_SEALED_EVIDENCE_FILENAMES = (
    TRAINING_EVIDENCE_FILENAME,
    TRAINING_REPORT_FILENAME,
)
REQUIRED_PROMOTION_EVIDENCE_FILENAMES = (
    *REQUIRED_SEALED_EVIDENCE_FILENAMES,
    DATASET_MANIFEST_FILENAME,
)
_CORE_ARTIFACT_FILENAMES = (MODEL_FILENAME, METADATA_FILENAME)


def _model_version(
    *,
    booster: lgb.Booster,
    config: PerformanceModelConfig,
    training_data_sha256: str,
    best_iteration: int,
) -> str:
    payload = json.dumps(
        {"config": config.to_dict(), "training_data_sha256": training_data_sha256},
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(payload)
    digest.update(booster.model_to_string(num_iteration=best_iteration).encode("utf-8"))
    return digest.hexdigest()


def _data_use_from_dict(payload: dict[str, Any]) -> DataUseDeclaration:
    return DataUseDeclaration(
        total_rows=int(payload["total_rows"]),
        evaluated_rows=int(payload["evaluated_rows"]),
        synthetic_rows=int(payload["synthetic_rows"]),
        synthetic_rows_excluded=bool(payload["synthetic_rows_excluded"]),
        synthetic_flags_declared=bool(payload["synthetic_flags_declared"]),
    )


def _metadata_payload(artifact: PerformanceModelArtifact, *, model_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": artifact.schema_version,
        "config": artifact.config.to_dict(),
        "evaluations": {
            name: evaluation.to_dict() for name, evaluation in artifact.evaluations.items()
        },
        "synthetic_data": artifact.data_use.to_dict(),
        "training_data_sha256": artifact.training_data_sha256,
        "model_version": artifact.model_version,
        "split_group_counts": artifact.split_group_counts,
        "split_row_counts": artifact.split_row_counts,
        "split_group_hashes": {
            name: list(hashes) for name, hashes in artifact.split_group_hashes.items()
        },
        "development_group_hashes": list(artifact.development_group_hashes),
        "feature_profiles": {
            name: profile.to_dict() for name, profile in artifact.feature_profiles.items()
        },
        "dataset_evidence": artifact.dataset_evidence.to_dict(),
        "calibration": artifact.calibration.to_dict(),
        "grouped_test": artifact.grouped_test.to_dict(),
        "test_uncertainty": artifact.test_uncertainty.to_dict(),
        "estimated_peak_training_memory_mb": artifact.estimated_peak_training_memory_mb,
        "allowed_missing_fraction": artifact.allowed_missing_fraction,
        "best_iteration": artifact.best_iteration,
        "confidence_level": artifact.confidence_level,
        "precise_predictions_enabled": artifact.precise_predictions_enabled,
        "promotable": artifact.promotable,
        "promotion_block_reasons": list(artifact.promotion_block_reasons),
        "requested_device": artifact.requested_device,
        "actual_device": artifact.actual_device,
        "device_fallback_reason": artifact.device_fallback_reason,
        "model_file": MODEL_FILENAME,
        "model_sha256": model_sha256,
    }


def _artifact_manifest_payload(
    artifact: PerformanceModelArtifact,
    *,
    model_path: Path,
    metadata_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": LEGACY_ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "artifact_schema_version": artifact.schema_version,
        "model_version": artifact.model_version,
        "development_only": not (artifact.promotable and artifact.precise_predictions_enabled),
        "serving_mode": (
            "precise"
            if artifact.promotable and artifact.precise_predictions_enabled
            else "development_relative_only"
        ),
        "promotion_blockers": list(artifact.promotion_block_reasons),
        "files": {
            MODEL_FILENAME: {
                "sha256": sha256_file(model_path),
                "size_bytes": model_path.stat().st_size,
            },
            METADATA_FILENAME: {
                "sha256": sha256_file(metadata_path),
                "size_bytes": metadata_path.stat().st_size,
            },
        },
    }


def _file_entry(path: Path) -> dict[str, int | str]:
    return {
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _evidence_filenames(filenames: Sequence[str]) -> tuple[str, ...]:
    if any(not isinstance(value, str) for value in filenames):
        raise TypeError("performance-artifact evidence filenames must be strings")
    normalized = tuple(sorted({str(value) for value in filenames}))
    if not normalized:
        raise ValueError("evidence sealing requires at least one evidence file")
    for filename in normalized:
        candidate = Path(filename)
        if (
            not filename
            or candidate.name != filename
            or filename in _CORE_ARTIFACT_FILENAMES
            or filename in {".", ".."}
        ):
            raise ValueError(f"invalid performance-artifact evidence filename: {filename!r}")
    if len(normalized) != len(filenames):
        raise ValueError("performance-artifact evidence filenames must be unique")
    return normalized


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must be a JSON object")
    return payload


def _validate_file_entries(
    source: Path,
    entries: object,
    *,
    expected_filenames: set[str],
) -> None:
    if not isinstance(entries, dict) or set(entries) != expected_filenames:
        raise ValueError("performance artifact manifest file set is invalid")
    for filename in sorted(expected_filenames):
        file_path = source / filename
        entry = entries[filename]
        if not isinstance(entry, dict):
            raise TypeError(f"artifact manifest entry for {filename!r} must be an object")
        if not file_path.is_file():
            raise FileNotFoundError(file_path)
        if file_path.stat().st_size != int(entry["size_bytes"]):
            raise ValueError(f"performance artifact {filename} size does not match manifest")
        if sha256_file(file_path) != str(entry["sha256"]):
            raise ValueError(f"performance artifact {filename} digest does not match manifest")


def seal_performance_artifact(
    path: str | Path,
    *,
    evidence_filenames: Sequence[str],
) -> Path:
    """Bind training evidence to an existing model/metadata pair.

    ``save_performance_artifact`` deliberately writes only the core model pair
    so callers can assemble run-specific evidence first. This final step is
    deterministic and fail-closed: production-eligible metadata cannot be
    loaded unless the required evidence files are present and hash-bound.
    """

    source = Path(path)
    manifest_path = source / ARTIFACT_MANIFEST_FILENAME
    manifest = _read_json_object(manifest_path, label="performance artifact manifest")
    if manifest.get("revoked") is True:
        raise ValueError("cannot seal a revoked performance artifact")
    if manifest.get("schema_version") != LEGACY_ARTIFACT_MANIFEST_SCHEMA_VERSION:
        raise ValueError("only an unsealed v1 performance artifact can be sealed")
    _validate_file_entries(
        source,
        manifest.get("files"),
        expected_filenames=set(_CORE_ARTIFACT_FILENAMES),
    )
    metadata = _read_json_object(source / METADATA_FILENAME, label="performance artifact metadata")
    if metadata.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("performance artifact metadata has an unsupported schema")
    evidence_names = _evidence_filenames(evidence_filenames)
    evidence_paths = {filename: source / filename for filename in evidence_names}
    for filename, evidence_path in evidence_paths.items():
        if not evidence_path.is_file():
            raise FileNotFoundError(f"performance artifact evidence is missing: {filename}")
    if bool(metadata.get("promotable")) and not set(REQUIRED_PROMOTION_EVIDENCE_FILENAMES).issubset(
        evidence_names
    ):
        raise ValueError(
            "promotable performance artifacts require sealed dataset and training evidence"
        )

    core_entries = {
        filename: _file_entry(source / filename) for filename in _CORE_ARTIFACT_FILENAMES
    }
    evidence_entries = {
        filename: _file_entry(evidence_path) for filename, evidence_path in evidence_paths.items()
    }
    _write_json_atomic(
        manifest_path,
        {
            "schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
            "artifact_schema_version": metadata["schema_version"],
            "model_version": metadata["model_version"],
            "development_only": not (
                bool(metadata["promotable"]) and bool(metadata["precise_predictions_enabled"])
            ),
            "serving_mode": (
                "precise"
                if bool(metadata["promotable"]) and bool(metadata["precise_predictions_enabled"])
                else "development_relative_only"
            ),
            "promotion_blockers": list(metadata["promotion_block_reasons"]),
            "evidence_files": list(evidence_names),
            "files": {**core_entries, **evidence_entries},
        },
    )
    return source


def _validate_sealed_evidence(
    source: Path,
    *,
    evidence_filenames: set[str],
    metadata: dict[str, Any],
) -> None:
    missing = set(REQUIRED_SEALED_EVIDENCE_FILENAMES).difference(evidence_filenames)
    if missing:
        raise ValueError(
            "sealed performance artifact is missing required evidence files: "
            + ", ".join(sorted(missing))
        )
    evidence = _read_json_object(
        source / TRAINING_EVIDENCE_FILENAME,
        label="performance training evidence",
    )
    report = _read_json_object(
        source / TRAINING_REPORT_FILENAME,
        label="performance training report",
    )
    if evidence.get("prepared_frame_sha256") != metadata.get("training_data_sha256"):
        raise ValueError("performance training evidence does not match the prepared training frame")
    if report.get("training_data_sha256") != metadata.get("training_data_sha256"):
        raise ValueError("performance training report does not match the prepared training frame")
    if report.get("model_version") != metadata.get("model_version"):
        raise ValueError("performance training report does not match the model version")
    report_input = report.get("input")
    if not isinstance(report_input, dict):
        raise ValueError("performance training report has an invalid input record")
    if evidence.get("source_sha256") != report_input.get("sha256"):
        raise ValueError("performance training evidence does not match the report input")
    if evidence.get("dataset_manifest_sha256") != metadata.get("dataset_evidence", {}).get(
        "manifest_sha256"
    ):
        raise ValueError("performance training evidence does not match the dataset manifest")
    if report.get("dataset_evidence") != metadata.get("dataset_evidence"):
        raise ValueError("performance training report does not match dataset evidence")

    split_map = {
        "training_group_hashes": "train",
        "validation_group_hashes": "validation",
        "calibration_group_hashes": "calibration",
        "internal_test_group_hashes": "test",
    }
    metadata_splits = metadata.get("split_group_hashes")
    if not isinstance(metadata_splits, dict):
        raise ValueError("performance artifact metadata has invalid split evidence")
    for evidence_name, split_name in split_map.items():
        if sorted(map(str, evidence.get(evidence_name, ()))) != sorted(
            map(str, metadata_splits.get(split_name, ()))
        ):
            raise ValueError(
                f"performance training evidence does not match the {split_name} split evidence"
            )
    if sorted(map(str, evidence.get("development_group_hashes", ()))) != sorted(
        map(str, metadata.get("development_group_hashes", ()))
    ):
        raise ValueError("performance training evidence does not match development split evidence")

    expected_manifest_sha256 = metadata.get("dataset_evidence", {}).get("manifest_sha256")
    if expected_manifest_sha256 is not None:
        if DATASET_MANIFEST_FILENAME not in evidence_filenames:
            raise ValueError(
                "performance artifact with verified data requires a sealed dataset manifest"
            )
        if sha256_file(source / DATASET_MANIFEST_FILENAME) != expected_manifest_sha256:
            raise ValueError("sealed dataset manifest digest does not match metadata")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
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
            json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
            temporary_name = handle.name
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and Path(temporary_name).exists():
            Path(temporary_name).unlink()


def save_performance_artifact(
    artifact: PerformanceModelArtifact,
    path: str | Path,
) -> Path:
    """Persist a model as LightGBM text plus verified JSON metadata."""

    destination = Path(path)
    if destination.exists() and not destination.is_dir():
        raise NotADirectoryError(destination)
    destination.mkdir(parents=True, exist_ok=True)
    model_path = destination / MODEL_FILENAME
    temporary_model: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination,
            prefix=f".{MODEL_FILENAME}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_model = handle.name
        artifact.booster.save_model(temporary_model, num_iteration=artifact.best_iteration)
        os.replace(temporary_model, model_path)
    finally:
        if temporary_model is not None and Path(temporary_model).exists():
            Path(temporary_model).unlink()

    payload = _metadata_payload(artifact, model_sha256=sha256_file(model_path))
    metadata_path = destination / METADATA_FILENAME
    _write_json_atomic(metadata_path, payload)
    _write_json_atomic(
        destination / ARTIFACT_MANIFEST_FILENAME,
        _artifact_manifest_payload(
            artifact,
            model_path=model_path,
            metadata_path=metadata_path,
        ),
    )
    return destination


def load_performance_artifact(path: str | Path) -> PerformanceModelArtifact:
    """Load an artifact only after validating its schema and model digest."""

    source = Path(path)
    manifest_path = source / ARTIFACT_MANIFEST_FILENAME
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise TypeError("performance artifact manifest must be a JSON object")
    if manifest.get("revoked") is True:
        reasons = manifest.get("revocation_reasons")
        reason_text = (
            "; ".join(str(reason) for reason in reasons)
            if isinstance(reasons, list) and reasons
            else "no reason supplied"
        )
        raise ValueError(f"performance artifact is revoked: {reason_text}")
    manifest_schema = manifest.get("schema_version")
    if manifest_schema not in {
        LEGACY_ARTIFACT_MANIFEST_SCHEMA_VERSION,
        ARTIFACT_MANIFEST_SCHEMA_VERSION,
    }:
        raise ValueError("unsupported performance artifact manifest schema")
    if manifest.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("artifact manifest references an unsupported artifact schema")
    evidence_filenames: set[str] = set()
    if manifest_schema == ARTIFACT_MANIFEST_SCHEMA_VERSION:
        raw_evidence_filenames = manifest.get("evidence_files")
        if not isinstance(raw_evidence_filenames, list):
            raise TypeError("sealed performance artifact evidence_files must be a list")
        evidence_filenames = set(_evidence_filenames(raw_evidence_filenames))
        if len(evidence_filenames) != len(raw_evidence_filenames):
            raise ValueError("sealed performance artifact evidence files must be unique")
    expected_files = set(_CORE_ARTIFACT_FILENAMES) | evidence_filenames
    _validate_file_entries(source, manifest.get("files"), expected_filenames=expected_files)
    metadata_path = source / METADATA_FILENAME
    with metadata_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError("performance artifact metadata must be a JSON object")
    if payload.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(f"unsupported artifact schema: {payload.get('schema_version')!r}")
    if payload.get("model_file") != MODEL_FILENAME:
        raise ValueError("artifact model file name is not supported")
    model_path = source / MODEL_FILENAME
    expected_digest = str(payload["model_sha256"])
    if sha256_file(model_path) != expected_digest:
        raise ValueError("performance model file digest does not match metadata")

    config_payload = payload["config"]
    evaluation_payload = payload["evaluations"]
    data_use_payload = payload["synthetic_data"]
    if not isinstance(config_payload, dict):
        raise TypeError("artifact config must be an object")
    if not isinstance(evaluation_payload, dict):
        raise TypeError("artifact evaluations must be an object")
    if not isinstance(data_use_payload, dict):
        raise TypeError("artifact synthetic_data must be an object")
    evaluations = {
        str(name): ModelEvaluation.from_dict(dict(value))
        for name, value in evaluation_payload.items()
        if isinstance(value, dict)
    }
    booster = lgb.Booster(model_file=str(model_path))
    config = PerformanceModelConfig.from_dict(config_payload)
    if booster.feature_name() != list(config.feature_columns):
        raise ValueError("LightGBM feature names/order do not match the artifact config")
    training_data_sha256 = str(payload["training_data_sha256"])
    best_iteration = int(payload["best_iteration"])
    model_version = str(payload["model_version"])
    if manifest.get("model_version") != model_version:
        raise ValueError("artifact manifest model version does not match metadata")
    if (
        _model_version(
            booster=booster,
            config=config,
            training_data_sha256=training_data_sha256,
            best_iteration=best_iteration,
        )
        != model_version
    ):
        raise ValueError("performance model version does not match its model and config")
    if manifest_schema == ARTIFACT_MANIFEST_SCHEMA_VERSION:
        _validate_sealed_evidence(
            source,
            evidence_filenames=evidence_filenames,
            metadata=payload,
        )
    elif bool(payload.get("promotable")):
        raise ValueError("promotable performance artifacts require an evidence-sealed v2 manifest")
    return PerformanceModelArtifact(
        config=config,
        booster=booster,
        evaluations=evaluations,
        data_use=_data_use_from_dict(data_use_payload),
        training_data_sha256=training_data_sha256,
        model_version=model_version,
        split_group_counts={
            str(name): int(count) for name, count in dict(payload["split_group_counts"]).items()
        },
        split_row_counts={
            str(name): int(count) for name, count in dict(payload["split_row_counts"]).items()
        },
        split_group_hashes={
            str(name): tuple(str(value) for value in hashes)
            for name, hashes in dict(payload["split_group_hashes"]).items()
        },
        development_group_hashes=tuple(str(value) for value in payload["development_group_hashes"]),
        feature_profiles={
            str(name): FeatureProfile.from_dict(dict(profile))
            for name, profile in dict(payload["feature_profiles"]).items()
        },
        dataset_evidence=DatasetEvidence.from_dict(dict(payload["dataset_evidence"])),
        calibration=PredictionIntervalCalibration.from_dict(dict(payload["calibration"])),
        grouped_test=GroupedTestDiagnostics.from_dict(dict(payload["grouped_test"])),
        test_uncertainty=RegressionUncertainty.from_dict(dict(payload["test_uncertainty"])),
        estimated_peak_training_memory_mb=float(payload["estimated_peak_training_memory_mb"]),
        allowed_missing_fraction=float(payload["allowed_missing_fraction"]),
        best_iteration=best_iteration,
        confidence_level=str(payload["confidence_level"]),  # type: ignore[arg-type]
        precise_predictions_enabled=bool(payload["precise_predictions_enabled"]),
        promotable=bool(payload["promotable"]),
        promotion_block_reasons=tuple(str(reason) for reason in payload["promotion_block_reasons"]),
        requested_device=str(payload["requested_device"]),
        actual_device=str(payload["actual_device"]),
        device_fallback_reason=(
            str(payload["device_fallback_reason"])
            if payload.get("device_fallback_reason") is not None
            else None
        ),
        schema_version=str(payload["schema_version"]),
    )
