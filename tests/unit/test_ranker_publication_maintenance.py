from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import scripts.maintain_ranker_publication_stages as maintenance_cli

import pc_build_recommender.ranking.publication as publication_module
from pc_build_recommender.ranking import (
    RankerPublicationMaintenanceError,
    maintain_ranker_publication_stages,
)

NOW = datetime(2026, 7, 23, 12, 0, tzinfo=UTC)
BUNDLE_NAME = "ranker-artifact"


def _stage(parent: Path, token: str, *, age: timedelta, filename: str = "partial.tmp") -> Path:
    path = parent / f".{BUNDLE_NAME}.publish-{token}"
    path.mkdir()
    (path / filename).write_text("partial", encoding="utf-8")
    timestamp = (NOW - age).timestamp()
    os.utime(path, (timestamp, timestamp))
    return path


def _maintain(
    parent: Path,
    *,
    dry_run: bool,
    minimum_age: timedelta = timedelta(hours=1),
):  # type: ignore[no-untyped-def]
    return maintain_ranker_publication_stages(
        parent,
        bundle_name=BUNDLE_NAME,
        minimum_age=minimum_age,
        dry_run=dry_run,
        now=NOW,
        maximum_entries=100,
    )


def test_dry_run_reports_old_stage_without_mutating_it(tmp_path: Path) -> None:
    stage = _stage(tmp_path, "abcdefgh", age=timedelta(days=2))

    report = _maintain(tmp_path, dry_run=True)

    assert stage.is_dir()
    assert report.dry_run is True
    assert report.would_remove_count == 1
    assert report.items[0].status == "would_remove"


def test_apply_removes_only_old_exact_stage_and_never_final_bundle(tmp_path: Path) -> None:
    final_bundle = tmp_path / BUNDLE_NAME
    final_bundle.mkdir()
    sentinel = final_bundle / "keep.txt"
    sentinel.write_text("committed", encoding="utf-8")
    old_stage = _stage(tmp_path, "oldstage1", age=timedelta(days=2))
    similar_but_invalid = tmp_path / f".{BUNDLE_NAME}.publish-short"
    similar_but_invalid.mkdir()

    report = _maintain(tmp_path, dry_run=False)

    assert report.removed_count == 1
    assert not old_stage.exists()
    assert sentinel.read_text(encoding="utf-8") == "committed"
    assert similar_but_invalid.is_dir()


def test_new_and_actively_locked_stages_are_preserved(tmp_path: Path) -> None:
    new_stage = _stage(tmp_path, "newstage1", age=timedelta(minutes=5))
    active_stage = _stage(tmp_path, "activestg", age=timedelta(days=2))
    activity_lock = publication_module.acquire_ranker_stage_activity_lock(active_stage)
    old_timestamp = (NOW - timedelta(days=2)).timestamp()
    os.utime(active_stage, (old_timestamp, old_timestamp))
    try:
        report = _maintain(tmp_path, dry_run=False)
    finally:
        activity_lock.release(remove=True)

    statuses = {item.stage_name: item.status for item in report.items}
    assert statuses[new_stage.name] == "preserved_new"
    assert statuses[active_stage.name] == "preserved_active"
    assert new_stage.is_dir()
    assert active_stage.is_dir()

    os.utime(active_stage, (old_timestamp, old_timestamp))
    second = _maintain(tmp_path, dry_run=False)
    assert second.removed_count == 1
    assert not active_stage.exists()
    assert new_stage.is_dir()


def test_nested_or_linklike_stage_content_blocks_deletion(tmp_path: Path) -> None:
    nested = _stage(tmp_path, "tampered1", age=timedelta(days=2))
    (nested / "unexpected-directory").mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("must survive", encoding="utf-8")
    linked = _stage(tmp_path, "symlink01", age=timedelta(days=2))
    link = linked / "outside-link"
    try:
        link.symlink_to(outside)
    except OSError:
        linked.rmdir() if not any(linked.iterdir()) else None
    report = _maintain(tmp_path, dry_run=False)

    statuses = {item.stage_name: item.status for item in report.items}
    assert statuses[nested.name] == "blocked"
    assert nested.is_dir()
    if link.is_symlink():
        assert statuses[linked.name] == "blocked"
        assert linked.is_dir()
    assert outside.read_text(encoding="utf-8") == "must survive"


def test_junction_classification_and_race_restat_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    junction_like = _stage(tmp_path, "junction1", age=timedelta(days=2))
    racing = _stage(tmp_path, "racing001", age=timedelta(days=2))
    original_is_linklike = publication_module._is_linklike
    original_snapshot = publication_module._snapshot_flat_stage
    snapshot_calls = 0

    def classify_junction(path: Path) -> bool:
        return path == junction_like or original_is_linklike(path)

    def mutate_after_first_snapshot(path: Path):  # type: ignore[no-untyped-def]
        nonlocal snapshot_calls
        snapshot = original_snapshot(path)
        if path == racing:
            snapshot_calls += 1
            if snapshot_calls == 1:
                (racing / "late-file").write_text("race", encoding="utf-8")
        return snapshot

    monkeypatch.setattr(publication_module, "_is_linklike", classify_junction)
    monkeypatch.setattr(publication_module, "_snapshot_flat_stage", mutate_after_first_snapshot)

    report = _maintain(tmp_path, dry_run=False)

    statuses = {item.stage_name: item.status for item in report.items}
    assert statuses[junction_like.name] == "blocked"
    assert statuses[racing.name] == "blocked"
    assert junction_like.is_dir()
    assert racing.is_dir()


def test_scope_validation_refuses_relative_root_home_and_unbounded_scan(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute path"):
        _maintain(Path("relative"), dry_run=True)
    with pytest.raises(ValueError, match="roots or broad"):
        _maintain(Path(tmp_path.anchor), dry_run=True)
    with pytest.raises(ValueError, match="home directory|roots or broad"):
        _maintain(Path.home().resolve(), dry_run=True)

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    try:
        linked_parent.symlink_to(real_parent, target_is_directory=True)
    except OSError:
        pass
    else:
        with pytest.raises(RankerPublicationMaintenanceError, match="symlink or junction"):
            _maintain(linked_parent, dry_run=True)

    (tmp_path / "one").touch()
    (tmp_path / "two").touch()
    with pytest.raises(RankerPublicationMaintenanceError, match="bounded entry limit"):
        maintain_ranker_publication_stages(
            tmp_path,
            bundle_name=BUNDLE_NAME,
            minimum_age=timedelta(hours=1),
            now=NOW,
            maximum_entries=1,
        )


def test_cli_defaults_to_dry_run_and_requires_apply_for_removal(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    stage = _stage(tmp_path, "clistage1", age=timedelta(days=2))
    base_args = [
        "--parent",
        str(tmp_path),
        "--bundle-name",
        BUNDLE_NAME,
        "--minimum-age-hours",
        "1",
        "--now",
        NOW.isoformat(),
    ]

    assert maintenance_cli.main(base_args) == 0
    dry_payload = json.loads(capsys.readouterr().out)
    assert dry_payload["dry_run"] is True
    assert dry_payload["would_remove_count"] == 1
    assert stage.is_dir()

    assert maintenance_cli.main([*base_args, "--apply"]) == 0
    applied_payload = json.loads(capsys.readouterr().out)
    assert applied_payload["dry_run"] is False
    assert applied_payload["removed_count"] == 1
    assert not stage.exists()
