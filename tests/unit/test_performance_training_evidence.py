from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from training._common import sha256_file
from training.train_performance import (
    _dataset_evidence_from_manifest,
)
from training.train_performance import (
    main as train_performance_main,
)

from pc_build_recommender.performance_models import (
    SYNTHETIC_GPU_FEATURE_COLUMNS,
    PerformanceModelConfig,
    load_performance_artifact,
    make_synthetic_performance_dataset,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "eligible_for_external_claims": [True, True],
            "feature_a": [1.0, 2.0],
            "feature_b": [3.0, 4.0],
        }
    )


def _config() -> PerformanceModelConfig:
    return PerformanceModelConfig(
        category="cpu",
        workload="measured_cpu_workload",
        feature_columns=("feature_a", "feature_b"),
        bootstrap_resamples=100,
    )


def _write_manifest(
    path: Path,
    *,
    csv_path: Path,
    frame: pd.DataFrame,
    eligible: bool = True,
) -> None:
    payload = {
        "schema_version": "test.performance-dataset.v1",
        "row_count": len(frame),
        "features": ["feature_a", "feature_b"],
        "output": {
            "rows": len(frame),
            "sha256": sha256_file(csv_path),
        },
        "selected_cohort": {
            "category": "cpu",
            "workload": "measured_cpu_workload",
        },
        "target": {
            "column": "target_score",
            "higher_is_better": True,
        },
        "promotion": {
            "eligible": eligible,
            "block_reasons": [] if eligible else ["source sample is not representative"],
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_verified_manifest_can_enable_dataset_evidence(tmp_path: Path) -> None:
    frame = _frame()
    csv_path = tmp_path / "data.csv"
    manifest_path = tmp_path / "manifest.json"
    frame.to_csv(csv_path, index=False)
    _write_manifest(manifest_path, csv_path=csv_path, frame=frame)

    evidence = _dataset_evidence_from_manifest(
        manifest_path,
        input_path=csv_path,
        frame=frame,
        config=_config(),
    )

    assert evidence.verified is True
    assert evidence.eligible_for_promotion is True
    assert evidence.block_reasons == ()


def test_upstream_non_promotable_policy_is_preserved(tmp_path: Path) -> None:
    frame = _frame()
    csv_path = tmp_path / "data.csv"
    manifest_path = tmp_path / "manifest.json"
    frame.to_csv(csv_path, index=False)
    _write_manifest(manifest_path, csv_path=csv_path, frame=frame, eligible=False)

    evidence = _dataset_evidence_from_manifest(
        manifest_path,
        input_path=csv_path,
        frame=frame,
        config=_config(),
    )

    assert evidence.eligible_for_promotion is False
    assert "source sample is not representative" in evidence.block_reasons


def test_manifest_digest_mismatch_fails_before_training(tmp_path: Path) -> None:
    frame = _frame()
    csv_path = tmp_path / "data.csv"
    manifest_path = tmp_path / "manifest.json"
    frame.to_csv(csv_path, index=False)
    _write_manifest(manifest_path, csv_path=csv_path, frame=frame)
    csv_path.write_text(csv_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="CSV SHA-256"):
        _dataset_evidence_from_manifest(
            manifest_path,
            input_path=csv_path,
            frame=frame,
            config=_config(),
        )


def test_training_cli_publishes_a_sealed_non_promotable_artifact(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    csv_path = tmp_path / "synthetic.csv"
    artifact_dir = tmp_path / "artifact"
    make_synthetic_performance_dataset(n_families=30, variants_per_family=2).to_csv(
        csv_path,
        index=False,
    )

    assert (
        train_performance_main(
            [
                "--input",
                str(csv_path),
                "--artifact-dir",
                str(artifact_dir),
                "--features",
                ",".join(SYNTHETIC_GPU_FEATURE_COLUMNS),
                "--target-transform",
                "log1p",
                "--allow-synthetic-diagnostics",
                "--device",
                "cpu",
                "--max-cpu-threads",
                "1",
                "--bootstrap-resamples",
                "100",
                "--lightgbm-params",
                '{"n_estimators":20}',
            ]
        )
        == 0
    )
    capsys.readouterr()
    manifest = json.loads((artifact_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    report = json.loads((artifact_dir / "training_report.json").read_text(encoding="utf-8"))
    restored = load_performance_artifact(artifact_dir)

    assert manifest["schema_version"].endswith(".v2")
    assert manifest["evidence_files"] == ["training_evidence.json", "training_report.json"]
    assert report["input"]["path"] == "<external>/synthetic.csv"
    assert report["artifact_path"] == "<external>/artifact"
    assert str(tmp_path) not in json.dumps(report)
    assert restored.promotable is False
    assert restored.config.target_transform == "log1p"


def test_training_cli_refuses_a_host_memory_cap_before_model_training(tmp_path: Path) -> None:
    csv_path = tmp_path / "synthetic.csv"
    artifact_dir = tmp_path / "artifact"
    make_synthetic_performance_dataset(n_families=30, variants_per_family=2).to_csv(
        csv_path,
        index=False,
    )

    with pytest.raises(MemoryError, match="host memory preflight refused training"):
        train_performance_main(
            [
                "--input",
                str(csv_path),
                "--artifact-dir",
                str(artifact_dir),
                "--features",
                ",".join(SYNTHETIC_GPU_FEATURE_COLUMNS),
                "--allow-synthetic-diagnostics",
                "--max-host-used-gb",
                "0.01",
            ]
        )

    assert not artifact_dir.exists()


def test_training_cli_seals_a_verified_dataset_manifest_when_supplied(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    csv_path = tmp_path / "synthetic.csv"
    source_manifest_path = tmp_path / "dataset-manifest.json"
    artifact_dir = tmp_path / "artifact"
    frame = make_synthetic_performance_dataset(n_families=30, variants_per_family=2)
    frame.to_csv(csv_path, index=False)
    source_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "test.performance-dataset.v1",
                "row_count": len(frame),
                "features": list(SYNTHETIC_GPU_FEATURE_COLUMNS),
                "output": {"rows": len(frame), "sha256": sha256_file(csv_path)},
                "selected_cohort": {"category": "gpu", "workload": "gaming_1440p"},
                "target": {"column": "target_score", "higher_is_better": True},
                "promotion": {
                    "eligible": False,
                    "block_reasons": ["synthetic fixture is not promotion eligible"],
                },
            }
        ),
        encoding="utf-8",
    )

    assert (
        train_performance_main(
            [
                "--input",
                str(csv_path),
                "--artifact-dir",
                str(artifact_dir),
                "--dataset-manifest",
                str(source_manifest_path),
                "--features",
                ",".join(SYNTHETIC_GPU_FEATURE_COLUMNS),
                "--allow-synthetic-diagnostics",
                "--device",
                "cpu",
                "--max-cpu-threads",
                "1",
                "--bootstrap-resamples",
                "100",
                "--lightgbm-params",
                '{"n_estimators":20}',
            ]
        )
        == 0
    )
    capsys.readouterr()
    manifest = json.loads((artifact_dir / "artifact_manifest.json").read_text(encoding="utf-8"))

    assert manifest["evidence_files"] == [
        "dataset_manifest.json",
        "training_evidence.json",
        "training_report.json",
    ]
    assert (
        artifact_dir / "dataset_manifest.json"
    ).read_bytes() == source_manifest_path.read_bytes()
    assert load_performance_artifact(artifact_dir).dataset_evidence.verified is True
