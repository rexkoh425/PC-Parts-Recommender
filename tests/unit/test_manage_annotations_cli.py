from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
from scripts import manage_annotations as cli

from pc_build_recommender.annotation import (
    AnnotationProjectProgress,
    AnnotationProjectStatus,
    AnnotationService,
    AnnotationTaskType,
    ClaimedAnnotationTask,
    VerifiedOIDCIdentity,
)


def _identity_payload(*, subject: str = "reviewer-1") -> dict[str, object]:
    return {
        "schema_version": cli.IDENTITY_SCHEMA_VERSION,
        "verification_status": "verified",
        "verification_method": "oidc-middleware-signature-and-claims-check",
        "verified_at": "2026-07-23T01:00:00+00:00",
        "issuer": "https://identity.example.test",
        "subject": subject,
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _args(*values: str) -> Any:
    return cli._parser().parse_args(list(values))


def test_actor_identity_is_fail_closed_without_a_trusted_pathway() -> None:
    args = _args("open-project", "--project-id", "project-1")

    with pytest.raises(cli.AnnotationCLIInputError, match="raw identity claims are rejected"):
        cli._load_actor_identity(args, {})


def test_actor_identity_file_requires_upstream_verification_metadata(tmp_path: Path) -> None:
    identity_path = tmp_path / "identity.json"
    _write_json(identity_path, _identity_payload())
    args = _args(
        "--verified-identity-file",
        str(identity_path),
        "open-project",
        "--project-id",
        "project-1",
    )

    identity = cli._load_actor_identity(args, {})

    assert identity == VerifiedOIDCIdentity(
        issuer="https://identity.example.test",
        subject="reviewer-1",
    )

    unsafe = _identity_payload()
    unsafe["id_token"] = "must-not-be-persisted"
    _write_json(identity_path, unsafe)
    with pytest.raises(cli.AnnotationCLIInputError, match="must not persist bearer token"):
        cli._load_actor_identity(args, {})


def test_trusted_environment_identity_needs_both_opt_in_and_assertion() -> None:
    environment = {
        cli.IDENTITY_ASSERTED_ENV: "true",
        cli.IDENTITY_ISSUER_ENV: "https://issuer.example.test",
        cli.IDENTITY_SUBJECT_ENV: "subject-7",
    }
    disabled = _args("open-project", "--project-id", "project-1")
    enabled = _args(
        "--allow-trusted-env-identity",
        "open-project",
        "--project-id",
        "project-1",
    )
    required_file = _args(
        "--allow-trusted-env-identity",
        "--require-verified-identity-file",
        "open-project",
        "--project-id",
        "project-1",
    )

    with pytest.raises(cli.AnnotationCLIInputError):
        cli._load_actor_identity(disabled, environment)
    assert cli._load_actor_identity(enabled, environment) == VerifiedOIDCIdentity(
        issuer="https://issuer.example.test",
        subject="subject-7",
    )
    with pytest.raises(cli.AnnotationCLIInputError, match="identity file is required"):
        cli._load_actor_identity(required_file, environment)


def test_bootstrap_admin_rejects_environment_identity_before_database_access(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def unexpected_service(_: str | None) -> Any:
        raise AssertionError("database must not be accessed")

    monkeypatch.setattr(cli, "_create_service", unexpected_service)
    environment = {
        cli.IDENTITY_ASSERTED_ENV: "true",
        cli.IDENTITY_ISSUER_ENV: "https://issuer.example.test",
        cli.IDENTITY_SUBJECT_ENV: "bootstrap-subject",
    }

    status = cli.main(
        [
            "--allow-trusted-env-identity",
            "bootstrap-admin",
            "--display-name",
            "First administrator",
        ],
        environ=environment,
    )

    assert status == 2
    error = json.loads(capsys.readouterr().err)
    assert error["status"] == "error"
    assert "requires an upstream-verified identity file" in error["message"]


class _ImportService:
    def __init__(self) -> None:
        self.groups: list[dict[str, Any]] = []
        self.items: list[dict[str, Any]] = []

    def import_batch(
        self,
        actor: VerifiedOIDCIdentity,
        project_id: str,
        *,
        groups: Iterator[dict[str, Any]],
    ) -> tuple[int, int]:
        for group in groups:
            self.groups.append({"actor": actor, "project_id": project_id, **group})
            self.items.extend(group["items"])
        return len(self.groups), len(self.items)


def test_jsonl_import_is_single_pass_atomic_service_input_and_bounded(tmp_path: Path) -> None:
    batch_path = tmp_path / "groups.jsonl"
    records = [
        {
            "schema_version": cli.GROUP_SCHEMA_VERSION,
            "group_key": f"query-{index}",
            "leakage_group_id": f"family-{index}",
            "category": "gpu",
            "split_name": "train" if index == 1 else "validation",
            "context_payload": {
                "query_text": "GPU for local AI",
                "structured_constraints": {"minimum_gpu_vram_gb": 16},
            },
            "items": [
                {
                    "target_id": f"gpu-{index}",
                    "evidence_payload": {"canonical_name": f"GPU {index}"},
                    "priority": index,
                }
            ],
        }
        for index in (1, 2)
    ]
    batch_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    args = _args(
        "import-batch",
        "--project-id",
        "project-1",
        "--input",
        str(batch_path),
        "--max-line-bytes",
        "4096",
    )
    actor = VerifiedOIDCIdentity("https://issuer.example.test", "admin")
    fake = _ImportService()

    result = cli._run_command(args, actor, cast(AnnotationService, fake))

    assert result["validated_groups"] == 2
    assert result["imported_groups"] == 2
    assert result["imported_items"] == 2
    assert [group["group_key"] for group in fake.groups] == ["query-1", "query-2"]
    assert [item["target_id"] for item in fake.items] == ["gpu-1", "gpu-2"]

    with pytest.raises(cli.AnnotationCLIInputError, match="exceeds 32 encoded bytes"):
        tuple(cli._iter_jsonl_group_records(batch_path, max_line_bytes=32))


class _DecisionService:
    def __init__(self, task: ClaimedAnnotationTask) -> None:
        self.task = task
        self.submission: dict[str, Any] | None = None

    def claim_review(
        self,
        actor: VerifiedOIDCIdentity,
        project_id: str,
        *,
        lease_seconds: int,
    ) -> ClaimedAnnotationTask:
        assert actor.subject == "reviewer"
        assert project_id == "project-1"
        assert lease_seconds == 600
        return self.task

    def submit_judgment(
        self,
        actor: VerifiedOIDCIdentity,
        assignment_id: str,
        **values: Any,
    ) -> str:
        self.submission = {"actor": actor, "assignment_id": assignment_id, **values}
        return "judgment-1"


def test_claim_file_carries_lease_secret_into_judgment_submission(tmp_path: Path) -> None:
    actor = VerifiedOIDCIdentity("https://issuer.example.test", "reviewer")
    task = ClaimedAnnotationTask(
        assignment_id="assignment-1",
        lease_token="one-time-secret",
        project_id="project-1",
        item_id="item-1",
        task_type=AnnotationTaskType.RELEVANCE,
        group_key="query-1",
        target_id="gpu-1",
        category="gpu",
        context_payload={"query_text": "local AI"},
        evidence_payload={"canonical_name": "GPU 1"},
        context_sha256="a" * 64,
        evidence_sha256="b" * 64,
        lease_expires_at=datetime(2026, 7, 23, 2, tzinfo=UTC),
    )
    fake = _DecisionService(task)
    claim_args = _args(
        "claim-review",
        "--project-id",
        "project-1",
        "--lease-seconds",
        "600",
    )

    claim_payload = cli._run_command(claim_args, actor, cast(AnnotationService, fake))
    claim_path = tmp_path / "claim.json"
    _write_json(claim_path, claim_payload)
    submit_args = _args(
        "submit-judgment",
        "--claim-file",
        str(claim_path),
        "--idempotency-key",
        "retry-key-1",
        "--label",
        "0",
        "--rationale",
        "Fails the minimum VRAM requirement.",
        "--hard-failure-code",
        "minimum_gpu_vram",
    )

    result = cli._run_command(submit_args, actor, cast(AnnotationService, fake))

    assert result["judgment_id"] == "judgment-1"
    assert fake.submission == {
        "actor": actor,
        "assignment_id": "assignment-1",
        "lease_token": "one-time-secret",
        "idempotency_key": "retry-key-1",
        "evidence_sha256": "b" * 64,
        "label": 0,
        "rationale": "Fails the minimum VRAM requirement.",
        "hard_failure_codes": ["minimum_gpu_vram"],
    }


class _ProgressService:
    def project_progress(
        self,
        actor: VerifiedOIDCIdentity,
        project_id: str,
    ) -> AnnotationProjectProgress:
        assert actor.subject == "admin"
        assert project_id == "project-1"
        return AnnotationProjectProgress(
            project_id=project_id,
            project_status=AnnotationProjectStatus.OPEN,
            task_type=AnnotationTaskType.RELEVANCE,
            observed_at=datetime(2026, 7, 23, 2, tzinfo=UTC),
            group_count=4,
            item_count=12,
            item_state_counts={
                "pending": 2,
                "in_review": 3,
                "needs_adjudication": 1,
                "resolved": 6,
            },
            judgment_coverage={
                "zero_judgments": 2,
                "one_judgment": 3,
                "two_or_more_judgments": 7,
            },
            review_assignment_counts={
                "active_leased": 3,
                "elapsed_leased": 1,
                "submitted": 14,
                "expired_record": 2,
            },
            adjudication_assignment_counts={
                "active_leased": 1,
                "elapsed_leased": 0,
                "submitted": 2,
                "expired_record": 0,
            },
            adjudication_required_count=1,
            adjudication_completed_count=2,
            synthetic_group_count=0,
            synthetic_item_count=0,
            preflight_blockers=("6 item(s) are not resolved",),
            coarse_freeze_preflight_passes=False,
            release_record_present=False,
        )


def test_project_status_command_emits_only_aggregate_progress() -> None:
    args = _args("project-status", "--project-id", "project-1")
    actor = VerifiedOIDCIdentity("https://issuer.example.test", "admin")

    result = cli._run_command(args, actor, cast(AnnotationService, _ProgressService()))

    assert result == {
        "schema_version": cli.RESULT_SCHEMA_VERSION,
        "status": "ok",
        "command": "project-status",
        "project_id": "project-1",
        "project_status": "open",
        "task_type": "relevance",
        "observed_at": "2026-07-23T02:00:00+00:00",
        "groups": 4,
        "items": {
            "total": 12,
            "states": {
                "pending": 2,
                "in_review": 3,
                "needs_adjudication": 1,
                "resolved": 6,
            },
            "judgment_coverage": {
                "zero_judgments": 2,
                "one_judgment": 3,
                "two_or_more_judgments": 7,
            },
            "synthetic": 0,
        },
        "assignments": {
            "review": {
                "active_leased": 3,
                "elapsed_leased": 1,
                "submitted": 14,
                "expired_record": 2,
            },
            "adjudication": {
                "active_leased": 1,
                "elapsed_leased": 0,
                "submitted": 2,
                "expired_record": 0,
            },
        },
        "adjudication": {"required": 1, "completed": 2},
        "synthetic_groups": 0,
        "freeze_preflight": {
            "coarse_gates_pass": False,
            "blockers": ["6 item(s) are not resolved"],
            "strict_freeze_required": True,
            "release_record_present": False,
        },
    }


def test_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "identity.json"
    path.write_text(
        '{"schema_version":"one","schema_version":"two"}',
        encoding="utf-8",
    )

    with pytest.raises(cli.AnnotationCLIInputError, match="duplicate JSON key"):
        cli._read_json(path, max_bytes=1024)
