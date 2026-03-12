from __future__ import annotations

from pathlib import Path

import pytest

from pc_build_recommender.evaluation import (
    DataUseDeclaration,
    build_dataset_manifest,
    load_dataset_manifest,
    verify_dataset_manifest,
    write_dataset_manifest,
)


def _manifest(root: Path):  # type: ignore[no-untyped-def]
    return build_dataset_manifest(
        dataset_name="entity-resolution-pilot",
        dataset_version="v1",
        root=root,
        files=["labels.jsonl", "splits.json"],
        row_count=2,
        group_count=2,
        data_use=DataUseDeclaration(
            total_rows=2,
            evaluated_rows=2,
            synthetic_rows=0,
            synthetic_rows_excluded=True,
        ),
        metadata={"label_schema": "pair-v1"},
    )


def test_manifest_hash_is_order_independent_and_detects_file_changes(tmp_path: Path) -> None:
    (tmp_path / "labels.jsonl").write_text('{"label": 1}\n', encoding="utf-8")
    (tmp_path / "splits.json").write_text('{"family-a": "test"}\n', encoding="utf-8")

    first = _manifest(tmp_path)
    reversed_files = build_dataset_manifest(
        dataset_name="entity-resolution-pilot",
        dataset_version="v1",
        root=tmp_path,
        files=["splits.json", "labels.jsonl"],
        row_count=2,
        group_count=2,
        data_use=first.data_use,
        metadata={"label_schema": "pair-v1"},
    )

    assert first.content_sha256 == reversed_files.content_sha256
    assert verify_dataset_manifest(first, root=tmp_path)

    output = write_dataset_manifest(first, tmp_path / "manifest.json")
    loaded = load_dataset_manifest(output)
    assert loaded == first

    (tmp_path / "labels.jsonl").write_text('{"label": 0}\n', encoding="utf-8")
    assert not verify_dataset_manifest(first, root=tmp_path)


def test_manifest_rejects_files_outside_dataset_root(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="outside dataset root"):
        build_dataset_manifest(
            dataset_name="bad",
            dataset_version="v1",
            root=dataset_root,
            files=[outside],
            row_count=0,
            group_count=0,
            data_use=DataUseDeclaration(
                total_rows=0,
                evaluated_rows=0,
                synthetic_rows=0,
                synthetic_rows_excluded=True,
            ),
        )
