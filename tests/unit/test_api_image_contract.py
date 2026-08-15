from __future__ import annotations

import tomllib
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


def test_api_image_contains_the_module_entrypoint_used_by_catalog_release() -> None:
    dockerfile = (REPOSITORY_ROOT / "infra" / "api.Dockerfile").read_text(encoding="utf-8")

    assert "scripts/__init__.py" in dockerfile
    assert "scripts/import_catalog_release.py" in dockerfile
    assert "scripts/import_processed_catalog.py" in dockerfile
    assert "COPY --chown=pcbr:pcbr pipelines ./pipelines" in dockerfile
    assert (
        'python -c "import yaml; from pipelines.source_release import '
        'verify_awin_production_batch_release"'
    ) in dockerfile


def test_serving_extra_declares_direct_runtime_dependencies() -> None:
    with (REPOSITORY_ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)

    serving = project["project"]["optional-dependencies"]["serving"]
    assert "sentence-transformers>=3.4,<6" in serving
    assert "pyyaml>=6,<7" in serving
