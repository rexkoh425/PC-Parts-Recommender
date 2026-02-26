from __future__ import annotations

import sys
from pathlib import Path

import pytest
from scripts.import_processed_catalog import main


def _production_arguments(tmp_path: Path) -> list[str]:
    return [
        "import_processed_catalog.py",
        "--buildcores",
        str(tmp_path / "catalog.jsonl"),
        "--offers",
        str(tmp_path / "offers.jsonl"),
        "--reviewed-mappings",
        str(tmp_path / "reviewed.json"),
        "--review-evidence",
        str(tmp_path / "review-evidence.jsonl"),
        "--serving-manifest",
        str(tmp_path / "serving-manifest.json"),
        "--serving-manifest-sha256",
        "a" * 64,
        "--require-production-ready",
    ]


@pytest.mark.parametrize(
    "caller_authority",
    [
        ["--entity-resolution-model", "caller-model"],
        ["--entity-resolution-evaluation", "caller-evaluation.json"],
        ["--allow-unpromoted-entity-resolution-shadow"],
    ],
)
def test_production_import_rejects_caller_supplied_er_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caller_authority: list[str],
) -> None:
    monkeypatch.setattr(sys, "argv", _production_arguments(tmp_path) + caller_authority)

    with pytest.raises(
        ValueError,
        match="authority must come only from the serving manifest",
    ):
        main()


def test_production_import_requires_operator_pinned_serving_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = _production_arguments(tmp_path)
    manifest_index = arguments.index("--serving-manifest")
    del arguments[manifest_index : manifest_index + 2]
    digest_index = arguments.index("--serving-manifest-sha256")
    del arguments[digest_index : digest_index + 2]
    monkeypatch.setattr(sys, "argv", arguments)

    with pytest.raises(ValueError, match="requires a pinned serving manifest"):
        main()
