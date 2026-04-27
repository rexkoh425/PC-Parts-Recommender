from __future__ import annotations

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_api_image_installs_semantic_serving_runtime_in_offline_mode() -> None:
    dockerfile = (REPOSITORY_ROOT / "infra" / "api.Dockerfile").read_text(encoding="utf-8")

    assert dockerfile.count("uv sync --locked --no-dev") == 2
    assert dockerfile.count("--extra serving") == 2
    assert "HF_HUB_OFFLINE=1" in dockerfile
    assert "TRANSFORMERS_OFFLINE=1" in dockerfile
    assert "HF_HUB_DISABLE_TELEMETRY=1" in dockerfile
    assert "TOKENIZERS_PARALLELISM=false" in dockerfile
    assert "huggingface-cli download" not in dockerfile
    assert "snapshot_download" not in dockerfile


def test_serving_extra_declares_sentence_transformers() -> None:
    project = (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'serving = [\n  "sentence-transformers>=3.4,<6",\n]' in project
