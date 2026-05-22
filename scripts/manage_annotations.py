"""Operate the production human-annotation workflow from a trusted shell boundary.

This CLI does not verify OIDC tokens.  Actor claims must come from a JSON file
written by an upstream OIDC verifier, or from explicitly trusted process
environment variables.  Raw issuer/subject command-line arguments are
intentionally unsupported.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from sqlalchemy import Engine
from sqlalchemy.exc import SQLAlchemyError

from pc_build_recommender.annotation import (
    AnnotationAuthorizationError,
    AnnotationConflictError,
    AnnotationFreezeBlockedError,
    AnnotationRole,
    AnnotationService,
    ClaimedAnnotationTask,
    VerifiedOIDCIdentity,
)
from pc_build_recommender.catalog.database import create_db_engine, create_session_factory

IDENTITY_SCHEMA_VERSION = "pc-build-recommender.verified-oidc-identity.v1"
PROJECT_SCHEMA_VERSION = "pc-build-recommender.annotation-project-import.v1"
GROUP_SCHEMA_VERSION = "pc-build-recommender.annotation-group-import.v1"
IMPORT_SCHEMA_VERSION = "pc-build-recommender.annotation-batch-import.v1"
CLAIM_SCHEMA_VERSION = "pc-build-recommender.annotation-claim.v1"
RESULT_SCHEMA_VERSION = "pc-build-recommender.annotation-cli-result.v1"

IDENTITY_FILE_ENV = "ANNOTATION_VERIFIED_IDENTITY_FILE"
IDENTITY_ASSERTED_ENV = "ANNOTATION_TRUSTED_IDENTITY_VERIFIED"
IDENTITY_ISSUER_ENV = "ANNOTATION_TRUSTED_OIDC_ISSUER"
IDENTITY_SUBJECT_ENV = "ANNOTATION_TRUSTED_OIDC_SUBJECT"

_SMALL_JSON_LIMIT = 1024 * 1024
_CLAIM_JSON_LIMIT = 16 * 1024 * 1024
_IMPORT_JSON_LIMIT = 64 * 1024 * 1024
_JSONL_LINE_LIMIT = 8 * 1024 * 1024


class AnnotationCLIInputError(ValueError):
    """Raised when an operational input contract is invalid."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Trust boundary:\n"
            "  The CLI never validates a JWT. --verified-identity-file must name an\n"
            "  identity artifact produced by trusted upstream OIDC middleware. The\n"
            "  environment pathway is disabled unless --allow-trusted-env-identity\n"
            "  is explicitly supplied. Use --require-verified-identity-file in\n"
            "  production wrappers to reject the environment pathway entirely."
        ),
    )
    parser.add_argument(
        "--database-url",
        help="SQLAlchemy URL; defaults to DATABASE_URL, then the repository local fallback.",
    )
    parser.add_argument(
        "--verified-identity-file",
        type=Path,
        help=f"Upstream-verified actor identity JSON; defaults to {IDENTITY_FILE_ENV}.",
    )
    parser.add_argument(
        "--allow-trusted-env-identity",
        action="store_true",
        help=(
            "Explicitly trust identity claims injected into the ANNOTATION_TRUSTED_* "
            "environment variables by the launching process."
        ),
    )
    parser.add_argument(
        "--require-verified-identity-file",
        action="store_true",
        help="Fail closed unless the actor identity comes from an upstream identity file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Atomically write the JSON result here instead of standard output.",
    )

    commands = parser.add_subparsers(dest="command", required=True)

    bootstrap = commands.add_parser(
        "bootstrap-admin",
        help=(
            "One-time bootstrap of the first administrator; requires the upstream-verified "
            "identity file pathway and an empty reviewer table."
        ),
    )
    bootstrap.add_argument("--display-name", required=True)

    provision = commands.add_parser(
        "provision-reviewer",
        help="Provision a reviewer from a second upstream-verified identity file.",
    )
    provision.add_argument("--reviewer-identity-file", type=Path, required=True)
    provision.add_argument("--display-name", required=True)
    provision.add_argument(
        "--role",
        action="append",
        choices=[role.value for role in AnnotationRole],
        required=True,
        dest="roles",
    )

    create_project = commands.add_parser(
        "create-project",
        help="Create a draft annotation project from a versioned JSON specification.",
    )
    create_project.add_argument("--spec", type=Path, required=True)

    import_batch = commands.add_parser(
        "import-batch",
        help="Import blinded groups and items from versioned JSON or streaming JSONL.",
    )
    import_batch.add_argument("--project-id", required=True)
    import_batch.add_argument("--input", type=Path, required=True)
    import_batch.add_argument(
        "--max-json-bytes",
        type=int,
        default=_IMPORT_JSON_LIMIT,
        help="Maximum size of a non-streaming JSON batch.",
    )
    import_batch.add_argument(
        "--max-line-bytes",
        type=int,
        default=_JSONL_LINE_LIMIT,
        help="Maximum encoded size of one JSONL group record.",
    )

    open_project = commands.add_parser("open-project", help="Open a populated draft project.")
    open_project.add_argument("--project-id", required=True)

    project_status = commands.add_parser(
        "project-status",
        help=(
            "Show an admin-only, aggregate-only collection and freeze-preflight report; "
            "it never emits evidence, labels, reviewer identities, or lease secrets."
        ),
    )
    project_status.add_argument("--project-id", required=True)

    for name, help_text in (
        ("claim-review", "Lease one blinded first-pass review task."),
        ("claim-adjudication", "Lease one independent adjudication task."),
    ):
        claim = commands.add_parser(name, help=help_text)
        claim.add_argument("--project-id", required=True)
        claim.add_argument("--lease-seconds", type=int, default=900)

    judgment = commands.add_parser(
        "submit-judgment",
        help="Submit an immutable reviewer decision using a claim JSON file.",
    )
    _add_submission_arguments(judgment, adjudication=False)

    adjudication = commands.add_parser(
        "submit-adjudication",
        help="Submit an immutable independent adjudication using a claim JSON file.",
    )
    _add_submission_arguments(adjudication, adjudication=True)

    freeze = commands.add_parser(
        "freeze-project",
        help="Run strict gates and create a deterministic content-addressed export.",
    )
    freeze.add_argument("--project-id", required=True)
    freeze.add_argument("--output-root", type=Path, required=True)
    return parser


def _add_submission_arguments(parser: argparse.ArgumentParser, *, adjudication: bool) -> None:
    parser.add_argument(
        "--claim-file",
        type=Path,
        required=True,
        help="Claim JSON emitted by claim-review or claim-adjudication; contains the lease secret.",
    )
    parser.add_argument(
        "--idempotency-key",
        required=True,
        help="Stable opaque key reused only when retrying the exact same decision.",
    )
    parser.add_argument("--label", required=True, dest="final_label" if adjudication else "label")
    parser.add_argument("--rationale", required=True)
    parser.add_argument(
        "--hard-failure-code",
        action="append",
        default=[],
        dest="final_hard_failure_codes" if adjudication else "hard_failure_codes",
        help="Repeat for each structured relevance hard-failure code.",
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AnnotationCLIInputError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _decode_json(raw: bytes, *, source: str) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AnnotationCLIInputError(f"{source} must be UTF-8 JSON") from exc
    try:
        return json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise AnnotationCLIInputError(
            f"invalid JSON in {source} at line {exc.lineno}, column {exc.colno}"
        ) from exc


def _read_json(path: Path, *, max_bytes: int) -> Any:
    if max_bytes <= 0:
        raise AnnotationCLIInputError("JSON byte limit must be positive")
    resolved = path.resolve(strict=True)
    size = resolved.stat().st_size
    if size > max_bytes:
        raise AnnotationCLIInputError(
            f"{resolved} is {size} bytes; configured maximum is {max_bytes} bytes"
        )
    return _decode_json(resolved.read_bytes(), source=str(resolved))


def _object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AnnotationCLIInputError(f"{name} must be a JSON object")
    return dict(value)


def _nonempty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AnnotationCLIInputError(f"{name} must be a non-empty string")
    return value.strip()


def _aware_timestamp(value: Any, *, name: str) -> datetime:
    text = _nonempty_string(value, name=name)
    try:
        timestamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AnnotationCLIInputError(f"{name} must be an ISO-8601 timestamp") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise AnnotationCLIInputError(f"{name} must include a timezone offset")
    return timestamp


def _load_verified_identity_file(path: Path, *, purpose: str) -> VerifiedOIDCIdentity:
    payload = _object(_read_json(path, max_bytes=_SMALL_JSON_LIMIT), name=f"{purpose} identity")
    if payload.get("schema_version") != IDENTITY_SCHEMA_VERSION:
        raise AnnotationCLIInputError(
            f"{purpose} identity schema_version must be {IDENTITY_SCHEMA_VERSION!r}"
        )
    if payload.get("verification_status") != "verified":
        raise AnnotationCLIInputError(
            f"{purpose} identity must assert verification_status='verified'"
        )
    _aware_timestamp(payload.get("verified_at"), name=f"{purpose} identity verified_at")
    _nonempty_string(
        payload.get("verification_method"),
        name=f"{purpose} identity verification_method",
    )
    forbidden = {"access_token", "id_token", "jwt", "refresh_token", "token"}
    present = sorted(forbidden.intersection(payload))
    if present:
        raise AnnotationCLIInputError(
            f"{purpose} identity must not persist bearer token fields: {', '.join(present)}"
        )
    return VerifiedOIDCIdentity(
        issuer=_nonempty_string(payload.get("issuer"), name=f"{purpose} identity issuer"),
        subject=_nonempty_string(payload.get("subject"), name=f"{purpose} identity subject"),
    )


def _configured_identity_file(
    args: argparse.Namespace,
    environ: Mapping[str, str],
) -> Path | None:
    explicit = args.verified_identity_file
    configured = environ.get(IDENTITY_FILE_ENV, "").strip()
    if explicit is not None:
        return Path(explicit)
    return Path(configured) if configured else None


def _load_actor_identity(
    args: argparse.Namespace,
    environ: Mapping[str, str],
) -> VerifiedOIDCIdentity:
    identity_file = _configured_identity_file(args, environ)
    if identity_file is not None:
        return _load_verified_identity_file(identity_file, purpose="actor")
    if args.require_verified_identity_file:
        raise AnnotationCLIInputError(
            "a verified identity file is required; environment identity fallback is disabled"
        )
    if not args.allow_trusted_env_identity:
        raise AnnotationCLIInputError(
            "no upstream-verified identity file was supplied; raw identity claims are rejected"
        )
    if environ.get(IDENTITY_ASSERTED_ENV, "").strip().casefold() != "true":
        raise AnnotationCLIInputError(
            f"trusted environment identity requires {IDENTITY_ASSERTED_ENV}=true"
        )
    return VerifiedOIDCIdentity(
        issuer=_nonempty_string(
            environ.get(IDENTITY_ISSUER_ENV),
            name=IDENTITY_ISSUER_ENV,
        ),
        subject=_nonempty_string(
            environ.get(IDENTITY_SUBJECT_ENV),
            name=IDENTITY_SUBJECT_ENV,
        ),
    )


def _load_project_spec(path: Path) -> dict[str, Any]:
    payload = _object(_read_json(path, max_bytes=_SMALL_JSON_LIMIT), name="project spec")
    if payload.get("schema_version") != PROJECT_SCHEMA_VERSION:
        raise AnnotationCLIInputError(
            f"project spec schema_version must be {PROJECT_SCHEMA_VERSION!r}"
        )
    required_strings = (
        "task_type",
        "dataset_name",
        "dataset_version",
        "rubric_version",
        "data_version",
    )
    result: dict[str, Any] = {
        name: _nonempty_string(payload.get(name), name=name) for name in required_strings
    }
    result["source_policy"] = _object(payload.get("source_policy"), name="source_policy")
    return result


def _normalise_group_record(value: Any, *, location: str) -> dict[str, Any]:
    payload = _object(value, name=f"group record at {location}")
    if payload.get("schema_version") != GROUP_SCHEMA_VERSION:
        raise AnnotationCLIInputError(
            f"group record at {location} schema_version must be {GROUP_SCHEMA_VERSION!r}"
        )
    result: dict[str, Any] = {
        name: _nonempty_string(payload.get(name), name=f"{location}.{name}")
        for name in ("group_key", "leakage_group_id", "category", "split_name")
    }
    result["context_payload"] = _object(
        payload.get("context_payload"), name=f"{location}.context_payload"
    )
    synthetic = payload.get("is_synthetic", False)
    if not isinstance(synthetic, bool):
        raise AnnotationCLIInputError(f"{location}.is_synthetic must be a boolean")
    result["is_synthetic"] = synthetic
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise AnnotationCLIInputError(f"{location}.items must be a non-empty array")
    normalised_items: list[dict[str, Any]] = []
    for index, raw_item in enumerate(items):
        item_location = f"{location}.items[{index}]"
        item = _object(raw_item, name=item_location)
        priority = item.get("priority", 0)
        if not isinstance(priority, int) or isinstance(priority, bool):
            raise AnnotationCLIInputError(f"{item_location}.priority must be an integer")
        item_synthetic = item.get("is_synthetic", False)
        if not isinstance(item_synthetic, bool):
            raise AnnotationCLIInputError(f"{item_location}.is_synthetic must be a boolean")
        normalised_items.append(
            {
                "target_id": _nonempty_string(
                    item.get("target_id"), name=f"{item_location}.target_id"
                ),
                "evidence_payload": _object(
                    item.get("evidence_payload"),
                    name=f"{item_location}.evidence_payload",
                ),
                "priority": priority,
                "is_synthetic": item_synthetic,
            }
        )
    result["items"] = normalised_items
    return result


def _iter_jsonl_group_records(path: Path, *, max_line_bytes: int) -> Iterator[dict[str, Any]]:
    if max_line_bytes <= 0:
        raise AnnotationCLIInputError("JSONL line byte limit must be positive")
    resolved = path.resolve(strict=True)
    with resolved.open("rb") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if len(raw_line) > max_line_bytes:
                raise AnnotationCLIInputError(
                    f"{resolved}:{line_number} exceeds {max_line_bytes} encoded bytes"
                )
            if not raw_line.strip():
                continue
            value = _decode_json(raw_line, source=f"{resolved}:{line_number}")
            yield _normalise_group_record(value, location=f"line {line_number}")


def _iter_group_records(
    path: Path,
    *,
    max_json_bytes: int,
    max_line_bytes: int,
) -> Iterator[dict[str, Any]]:
    if path.suffix.casefold() == ".jsonl":
        yield from _iter_jsonl_group_records(path, max_line_bytes=max_line_bytes)
        return
    payload = _object(_read_json(path, max_bytes=max_json_bytes), name="annotation batch")
    if payload.get("schema_version") != IMPORT_SCHEMA_VERSION:
        raise AnnotationCLIInputError(
            f"annotation batch schema_version must be {IMPORT_SCHEMA_VERSION!r}"
        )
    groups = payload.get("groups")
    if not isinstance(groups, list) or not groups:
        raise AnnotationCLIInputError("annotation batch groups must be a non-empty array")
    for index, group in enumerate(groups):
        yield _normalise_group_record(group, location=f"groups[{index}]")


def _load_claim(path: Path) -> dict[str, str]:
    payload = _object(_read_json(path, max_bytes=_CLAIM_JSON_LIMIT), name="claim file")
    if payload.get("schema_version") != CLAIM_SCHEMA_VERSION:
        raise AnnotationCLIInputError(f"claim file schema_version must be {CLAIM_SCHEMA_VERSION!r}")
    if payload.get("claimed") is not True:
        raise AnnotationCLIInputError("claim file does not contain a leased task")
    return {
        name: _nonempty_string(payload.get(name), name=f"claim file {name}")
        for name in ("assignment_id", "lease_token", "evidence_sha256")
    }


def _decision_label(value: str) -> str | int:
    stripped = value.strip()
    return int(stripped) if stripped in {"0", "1", "2", "3", "4"} else stripped


def _claim_payload(command: str, task: ClaimedAnnotationTask | None) -> dict[str, Any]:
    if task is None:
        return {
            "schema_version": CLAIM_SCHEMA_VERSION,
            "status": "ok",
            "command": command,
            "claimed": False,
        }
    return {
        "schema_version": CLAIM_SCHEMA_VERSION,
        "status": "ok",
        "command": command,
        "claimed": True,
        "assignment_id": task.assignment_id,
        "lease_token": task.lease_token,
        "project_id": task.project_id,
        "item_id": task.item_id,
        "task_type": task.task_type.value,
        "group_key": task.group_key,
        "target_id": task.target_id,
        "category": task.category,
        "context_payload": dict(task.context_payload),
        "evidence_payload": dict(task.evidence_payload),
        "context_sha256": task.context_sha256,
        "evidence_sha256": task.evidence_sha256,
        "lease_expires_at": task.lease_expires_at.isoformat(),
    }


def _create_service(database_url: str | None) -> tuple[AnnotationService, Engine]:
    engine = create_db_engine(database_url)
    return AnnotationService(create_session_factory(engine)), engine


def _run_command(
    args: argparse.Namespace,
    actor: VerifiedOIDCIdentity,
    service: AnnotationService,
) -> dict[str, Any]:
    if args.command == "bootstrap-admin":
        reviewer_id = service.bootstrap_administrator(
            actor,
            display_name=args.display_name,
        )
        return _result(args.command, reviewer_id=reviewer_id)

    if args.command == "provision-reviewer":
        identity = _load_verified_identity_file(
            args.reviewer_identity_file,
            purpose="reviewer",
        )
        reviewer_id = service.provision_reviewer(
            actor,
            identity=identity,
            display_name=args.display_name,
            roles=args.roles,
        )
        return _result(args.command, reviewer_id=reviewer_id)

    if args.command == "create-project":
        spec = _load_project_spec(args.spec)
        project_id = service.create_project(actor, **spec)
        return _result(args.command, project_id=project_id)

    if args.command == "import-batch":
        return _import_batch(args, actor, service)

    if args.command == "open-project":
        service.open_project(actor, args.project_id)
        return _result(args.command, project_id=args.project_id)

    if args.command == "project-status":
        progress = service.project_progress(actor, args.project_id)
        return _result(args.command, **progress.to_dict())

    if args.command == "claim-review":
        task = service.claim_review(
            actor,
            args.project_id,
            lease_seconds=args.lease_seconds,
        )
        return _claim_payload(args.command, task)

    if args.command == "claim-adjudication":
        task = service.claim_adjudication(
            actor,
            args.project_id,
            lease_seconds=args.lease_seconds,
        )
        return _claim_payload(args.command, task)

    if args.command == "submit-judgment":
        claim = _load_claim(args.claim_file)
        judgment_id = service.submit_judgment(
            actor,
            claim["assignment_id"],
            lease_token=claim["lease_token"],
            idempotency_key=args.idempotency_key,
            evidence_sha256=claim["evidence_sha256"],
            label=_decision_label(args.label),
            rationale=args.rationale,
            hard_failure_codes=args.hard_failure_codes,
        )
        return _result(args.command, judgment_id=judgment_id)

    if args.command == "submit-adjudication":
        claim = _load_claim(args.claim_file)
        adjudication_id = service.submit_adjudication(
            actor,
            claim["assignment_id"],
            lease_token=claim["lease_token"],
            idempotency_key=args.idempotency_key,
            evidence_sha256=claim["evidence_sha256"],
            final_label=_decision_label(args.final_label),
            rationale=args.rationale,
            final_hard_failure_codes=args.final_hard_failure_codes,
        )
        return _result(args.command, adjudication_id=adjudication_id)

    if args.command == "freeze-project":
        release = service.freeze_project(
            actor,
            args.project_id,
            output_root=args.output_root,
        )
        return _result(
            args.command,
            project_id=release.project_id,
            release_sha256=release.release_sha256,
            manifest_sha256=release.manifest_sha256,
            artifact_directory=str(release.artifact_directory),
            files=dict(release.files),
        )
    raise AssertionError(f"unsupported command: {args.command}")


def _import_batch(
    args: argparse.Namespace,
    actor: VerifiedOIDCIdentity,
    service: AnnotationService,
) -> dict[str, Any]:
    records = _iter_group_records(
        args.input,
        max_json_bytes=args.max_json_bytes,
        max_line_bytes=args.max_line_bytes,
    )
    imported_groups, imported_items = service.import_batch(
        actor,
        args.project_id,
        groups=records,
    )
    return _result(
        args.command,
        project_id=args.project_id,
        validated_groups=imported_groups,
        imported_groups=imported_groups,
        imported_items=imported_items,
    )


def _result(command: str, **values: Any) -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "ok",
        "command": command,
        **values,
    }


def _emit(payload: Mapping[str, Any], output: Path | None) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if output is None:
        sys.stdout.buffer.write(encoded)
        return
    destination = output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            "wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(encoded)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        with suppress(OSError):
            temporary_path.chmod(0o600)
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _emit_error(exc: Exception) -> None:
    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "error",
        "error_type": type(exc).__name__,
        "message": str(exc),
    }
    sys.stderr.write(json.dumps(payload, sort_keys=True) + "\n")


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> int:
    args = _parser().parse_args(argv)
    environment = os.environ if environ is None else environ
    engine: Engine | None = None
    try:
        identity_file = _configured_identity_file(args, environment)
        if args.command == "bootstrap-admin" and identity_file is None:
            raise AnnotationCLIInputError(
                "bootstrap-admin requires an upstream-verified identity file; "
                "trusted environment claims are not accepted"
            )
        actor = _load_actor_identity(args, environment)
        if (
            args.output is not None
            and identity_file is not None
            and args.output.resolve() == identity_file.resolve()
        ):
            raise AnnotationCLIInputError("output path must not overwrite the actor identity file")
        service, engine = _create_service(args.database_url)
        payload = _run_command(args, actor, service)
        _emit(payload, args.output)
        return 0
    except (
        AnnotationAuthorizationError,
        AnnotationCLIInputError,
        AnnotationConflictError,
        AnnotationFreezeBlockedError,
        FileNotFoundError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        _emit_error(exc)
        return 2
    except SQLAlchemyError as exc:
        _emit_error(exc)
        return 1
    finally:
        if engine is not None:
            engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
