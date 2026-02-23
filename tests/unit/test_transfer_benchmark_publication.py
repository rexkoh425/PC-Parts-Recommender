"""Failure-atomic behavior for the external ER transfer benchmark writer."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from training.benchmark_entity_resolution_transfer import _evaluate_and_save


class _Evaluation:
    def to_dict(self) -> dict[str, float]:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "average_precision": 1.0}


class _FakeResolver:
    def __init__(self, name: str, events: list[str], *, fail_on_predict: bool = False) -> None:
        self.name = name
        self.events = events
        self.fail_on_predict = fail_on_predict

    def predict_proba(self, _rows: object) -> np.ndarray[Any, Any]:
        self.events.append(f"{self.name}:predict")
        if self.fail_on_predict:
            raise RuntimeError(f"{self.name} failed during evaluation")
        return np.asarray([0.1, 0.9], dtype=np.float64)

    def evaluate(self, _rows: object, *, classification_threshold: float) -> _Evaluation:
        assert 0.0 <= classification_threshold <= 1.0
        self.events.append(f"{self.name}:evaluate")
        return _Evaluation()

    def save_artifact(self, path: Path) -> Path:
        self.events.append(f"{self.name}:save")
        path.mkdir(parents=True)
        return path


def _splits() -> dict[str, tuple[Any, ...]]:
    rows = (SimpleNamespace(label=0), SimpleNamespace(label=1))
    return {"validation": rows, "test": rows}


def test_transfer_benchmark_does_not_persist_partial_baselines_when_later_evaluation_fails(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    artifact_dir = tmp_path / "artifacts"
    dataset_manifest = tmp_path / "dataset_manifest.json"
    dataset_manifest.write_text("{}\n", encoding="utf-8")
    models: dict[str, Any] = {
        "exact": _FakeResolver("exact", events),
        "lightgbm": _FakeResolver("lightgbm", events, fail_on_predict=True),
    }

    with pytest.raises(RuntimeError, match="lightgbm failed during evaluation"):
        _evaluate_and_save(
            models,
            _splits(),
            artifact_dir=artifact_dir,
            dataset_manifest=dataset_manifest,
        )

    assert events == ["exact:predict", "exact:evaluate", "exact:evaluate", "lightgbm:predict"]
    assert not artifact_dir.exists()


def test_transfer_benchmark_persists_only_after_every_model_evaluates(tmp_path: Path) -> None:
    events: list[str] = []
    artifact_dir = tmp_path / "artifacts"
    dataset_manifest = tmp_path / "dataset_manifest.json"
    dataset_manifest.write_text("{}\n", encoding="utf-8")
    models: dict[str, Any] = {
        "exact": _FakeResolver("exact", events),
        "lightgbm": _FakeResolver("lightgbm", events),
    }

    reports = _evaluate_and_save(
        models,
        _splits(),
        artifact_dir=artifact_dir,
        dataset_manifest=dataset_manifest,
    )

    assert events == [
        "exact:predict",
        "exact:evaluate",
        "exact:evaluate",
        "lightgbm:predict",
        "lightgbm:evaluate",
        "lightgbm:evaluate",
        "exact:save",
        "lightgbm:save",
    ]
    assert set(reports) == {"exact", "lightgbm"}
    assert (artifact_dir / "exact" / "transfer_benchmark_evidence.json").is_file()
    assert (artifact_dir / "lightgbm" / "transfer_benchmark_evidence.json").is_file()
    evidence = json.loads(
        (artifact_dir / "exact" / "transfer_benchmark_evidence.json").read_text(encoding="utf-8")
    )
    assert evidence["dataset_manifest"] == "<external>/dataset_manifest.json"
    assert reports["exact"]["artifact_path"] == "<external>/exact"
    assert str(tmp_path) not in json.dumps({"evidence": evidence, "reports": reports})
