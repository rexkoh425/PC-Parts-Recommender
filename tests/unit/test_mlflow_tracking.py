from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
from training.mlflow_tracking import (
    MLflowTrackingConfig,
    OptionalMLflowRun,
    build_artifact_manifest,
    finite_metrics,
    flatten_parameters,
)


def test_tracking_is_disabled_by_default_and_does_not_create_a_store(tmp_path: Path) -> None:
    store = tmp_path / "mlruns"
    config = MLflowTrackingConfig(
        enabled=False,
        experiment_name="disabled-test",
        tracking_uri=store.resolve().as_uri(),
    )

    with OptionalMLflowRun(config) as tracking:
        tracking.log_params({"dataset": {"version": "v1"}})
        assert tracking.describe()["status"] == "disabled"

    assert not store.exists()


def test_enabled_tracking_is_non_fatal_when_optional_dependency_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_import = importlib.import_module

    def missing_mlflow(name: str) -> Any:
        if name == "mlflow":
            raise ModuleNotFoundError("No module named 'mlflow'")
        return real_import(name)

    monkeypatch.setattr("training.mlflow_tracking.importlib.import_module", missing_mlflow)
    config = MLflowTrackingConfig(
        enabled=True,
        experiment_name="missing-dependency-test",
        tracking_uri=(tmp_path / "mlruns").resolve().as_uri(),
    )

    with OptionalMLflowRun(config) as tracking:
        tracking.log_metrics({"heldout.r2": 0.9})
        report = tracking.describe()

    assert report["status"] == "dependency_missing"
    assert report["run_id"] is None
    assert not (tmp_path / "mlruns").exists()


def test_artifact_manifest_is_stable_and_rejects_pickle(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "metadata.json").write_text('{"version": 1}\n', encoding="utf-8")
    (model_dir / "model.txt").write_text("tree\n", encoding="utf-8")

    first = build_artifact_manifest(model_dir)
    second = build_artifact_manifest(model_dir)

    assert first == second
    assert first["file_count"] == 2
    assert len(first["content_sha256"]) == 64
    assert [entry["path"] for entry in first["files"]] == ["metadata.json", "model.txt"]

    (model_dir / "unsafe.pkl").write_bytes(b"not-even-a-real-pickle")
    with pytest.raises(ValueError, match="pickle-style artifacts"):
        build_artifact_manifest(model_dir)


def test_tracking_uri_credentials_are_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://user:secret@example.test/mlflow")
    config = MLflowTrackingConfig(enabled=True, experiment_name="redaction-test")

    assert config.resolved_tracking_uri.endswith("example.test/mlflow")
    assert config.safe_tracking_uri == "https://example.test/mlflow"
    assert "secret" not in config.safe_tracking_uri


def test_parameter_and_metric_normalisation_is_bounded() -> None:
    params = flatten_parameters(
        {"model": {"seed": 42, "features": ["x"] * 1000}, "none": None}
    )

    assert params["model.seed"] == 42
    assert str(params["model.features"]).startswith("sha256:")
    assert params["none"] == "null"
    assert finite_metrics({"r2": 0.9, "nan": float("nan"), "flag": True}) == {"r2": 0.9}


@pytest.mark.skipif(importlib.util.find_spec("mlflow") is None, reason="mlops extra absent")
def test_real_mlflow_file_backend_logs_native_artifacts(tmp_path: Path) -> None:
    mlflow: Any = importlib.import_module("mlflow")

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.txt").write_text("native booster\n", encoding="utf-8")
    (model_dir / "metadata.json").write_text(
        json.dumps({"format": "native"}) + "\n", encoding="utf-8"
    )
    config = MLflowTrackingConfig(
        enabled=True,
        experiment_name="pcbr-mlflow-test",
        run_name="native-artifact-test",
        tracking_uri=(tmp_path / "mlruns").resolve().as_uri(),
    )

    with OptionalMLflowRun(config) as tracking:
        tracking.log_params({"dataset": {"sha256": "abc", "version": "v1"}})
        tracking.log_metrics({"heldout.r2": 0.91})
        manifest = tracking.log_native_artifacts(model_dir)
        run_id = tracking.run_id
        assert tracking.active

    assert run_id is not None
    assert manifest["file_count"] == 2
    run = mlflow.tracking.MlflowClient(tracking_uri=config.resolved_tracking_uri).get_run(run_id)
    assert run.info.status == "FINISHED"
    assert run.data.params["dataset.sha256"] == "abc"
    assert run.data.metrics["heldout.r2"] == pytest.approx(0.91)
    artifacts = mlflow.tracking.MlflowClient(
        tracking_uri=config.resolved_tracking_uri
    ).list_artifacts(run_id, "model")
    assert {Path(item.path).name for item in artifacts} == {"metadata.json", "model.txt"}
