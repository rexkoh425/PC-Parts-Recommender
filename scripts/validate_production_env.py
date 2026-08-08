"""Fail-fast validation for the single-host production deployment contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
from pathlib import Path
from typing import TypedDict
from urllib.parse import urlparse

from pc_build_recommender.evaluation.manifest import sha256_file
from pc_build_recommender.retrieval import (
    EncoderBundleValidationError,
    validate_encoder_bundle,
)

REQUIRED_VALUES = {
    "COMPOSE_PROJECT_NAME",
    "PCBR_DEPLOYMENT_ID",
    "PCBR_PUBLIC_WEB_URL",
    "PCBR_PUBLIC_API_URL",
    "PCBR_BIND_ADDRESS",
    "PCBR_POSTGRES_ADMIN_USER",
    "PCBR_DATABASE_NAME",
    "PCBR_MIGRATOR_USER",
    "PCBR_APP_USER",
    "PCBR_DAGSTER_DATABASE",
    "PCBR_DAGSTER_USER",
    "PCBR_MLFLOW_DATABASE",
    "PCBR_MLFLOW_USER",
    "PCBR_MONITOR_USER",
    "PCBR_API_ENVIRONMENT",
    "PCBR_API_SERVICE_MODE",
    "PCBR_API_DOCS_ENABLED",
    "PCBR_API_CORS_ORIGINS",
    "PCBR_API_DATA_VERSION",
    "PCBR_API_RANKING_MODEL_VERSION",
    "PCBR_API_COMPATIBILITY_RULE_VERSION",
    "PCBR_API_SOLVER_VERSION",
    "PCBR_API_SERVING_MANIFEST_PATH",
    "PCBR_API_SERVING_MANIFEST_SHA256",
    "PCBR_API_SEMANTIC_ENCODER_BUNDLE_SHA256",
    "PCBR_API_MAX_REQUEST_BODY_BYTES",
    "PCBR_API_BUILD_GENERATION_MAX_CONCURRENCY",
    "PCBR_API_BUILD_GENERATION_MAX_QUEUE_SIZE",
    "PCBR_API_BUILD_GENERATION_QUEUE_TIMEOUT_SECONDS",
}
IMAGE_VALUES = {
    "PCBR_API_IMAGE",
    "PCBR_WEB_IMAGE",
    "PCBR_DAGSTER_IMAGE",
    "PCBR_MLFLOW_IMAGE",
    "PCBR_POSTGRES_IMAGE",
    "PCBR_POSTGRES_EXPORTER_IMAGE",
    "PCBR_PROMETHEUS_IMAGE",
    "PCBR_BLACKBOX_EXPORTER_IMAGE",
    "PCBR_ALPINE_IMAGE",
}
SECRET_PATHS = {
    "PCBR_POSTGRES_ADMIN_PASSWORD_FILE",
    "PCBR_MIGRATOR_PASSWORD_FILE",
    "PCBR_APP_PASSWORD_FILE",
    "PCBR_DAGSTER_PASSWORD_FILE",
    "PCBR_MLFLOW_PASSWORD_FILE",
    "PCBR_MONITOR_PASSWORD_FILE",
    "PCBR_API_ADMIN_TOKEN_FILE",
}
FILE_PATHS = {
    "PCBR_BUILDCORES_CATALOG_FILE",
    "PCBR_CATALOG_READINESS_ARTIFACT_FILE",
    "PCBR_GOVERNED_OFFERS_FILE",
    "PCBR_REVIEWED_MAPPING_FILE",
    "PCBR_REVIEW_EVIDENCE_FILE",
}
_LEGACY_OFFERS_FILE_KEY = "PCBR_DYNACORE_OFFERS_FILE"
_GOVERNED_OFFERS_FILE_KEY = "PCBR_GOVERNED_OFFERS_FILE"
DIRECTORY_PATHS = {
    "PCBR_PIPELINE_DATA_DIR",
    "PCBR_PIPELINE_OPERATIONS_DIR",
    "PCBR_PIPELINE_ARTIFACTS_DIR",
    "PCBR_MLFLOW_ARTIFACTS_DIR",
    "PCBR_BACKUP_DIR",
    "PCBR_SERVING_RELEASE_DIR",
}
PORT_VALUES = {
    "PCBR_WEB_PORT",
    "PCBR_API_PORT",
    "PCBR_DAGSTER_PORT",
    "PCBR_MLFLOW_PORT",
    "PCBR_PROMETHEUS_PORT",
}
FORBIDDEN_CLEAR_TEXT = {
    "POSTGRES_PASSWORD",
    "DATABASE_URL",
    "MLFLOW_BACKEND_STORE_URI",
    "DATA_SOURCE_PASS",
    "PCBR_API_ADMIN_TOKEN",
}
PLACEHOLDER_PATTERN = re.compile(r"change[_-]?me|example\.com|<[^>]+>|your[_-]", re.IGNORECASE)
IMAGE_PATTERN = re.compile(r"^\S+@sha256:[0-9a-f]{64}$", re.IGNORECASE)
IDENTIFIER_PATTERN = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")
UNSAFE_VERSION_PATTERN = re.compile(r"development|demo|untrained|unknown|change[_-]?me", re.I)
_SERVING_MANIFEST_SCHEMA = "pc-build-recommender.serving-release.v3"
_SERVING_MANIFEST_FIELDS = {
    "schema_version",
    "catalog_data_version",
    "catalog",
    "catalog_inputs",
    "embedding",
    "entity_resolution",
    "retrieval",
    "ranker",
    "ranker_promotion",
    "performance",
    "content_sha256",
}


class ValidationResult(TypedDict):
    status: str
    environment_file: str
    checked_values: int
    errors: list[str]
    warnings: list[str]


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite value {value}")


def _parse_env(path: Path) -> tuple[dict[str, str], list[str]]:
    values: dict[str, str] = {}
    errors: list[str] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        if "=" not in line:
            errors.append(f"line {line_number}: expected KEY=VALUE")
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            errors.append(f"line {line_number}: invalid environment key {key!r}")
            continue
        if key in values:
            errors.append(f"line {line_number}: duplicate key {key}")
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key] = value
    return values, errors


def _resolve_path(env_file: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = env_file.parent / candidate
    return candidate.resolve()


def _https_origin(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme == "https" and bool(parsed.netloc) and not parsed.path.rstrip("/")


def _normalize_deprecated_offer_path(
    values: dict[str, str],
    warnings: list[str],
) -> None:
    """Accept the old deployment key while making canonical precedence explicit."""

    legacy_value = values.pop(_LEGACY_OFFERS_FILE_KEY, None)
    if legacy_value is None:
        return
    if values.get(_GOVERNED_OFFERS_FILE_KEY):
        warnings.append(
            f"{_LEGACY_OFFERS_FILE_KEY} is deprecated and ignored because "
            f"{_GOVERNED_OFFERS_FILE_KEY} is set"
        )
        return
    values[_GOVERNED_OFFERS_FILE_KEY] = legacy_value
    warnings.append(
        f"{_LEGACY_OFFERS_FILE_KEY} is deprecated; rename it to {_GOVERNED_OFFERS_FILE_KEY}"
    )


def _validate_paths(
    env_file: Path, values: dict[str, str], errors: list[str], warnings: list[str]
) -> None:
    for key in sorted(SECRET_PATHS | FILE_PATHS | DIRECTORY_PATHS):
        value = values.get(key)
        if not value:
            errors.append(f"missing required path {key}")
            continue
        path = _resolve_path(env_file, value)
        if key in DIRECTORY_PATHS:
            if not path.is_dir():
                errors.append(f"{key} does not name an existing directory: {path}")
            elif key == "PCBR_BACKUP_DIR" and not os.access(path, os.W_OK):
                errors.append(f"{key} is not writable: {path}")
            continue
        if not path.is_file():
            errors.append(f"{key} does not name an existing file: {path}")
            continue
        if key in SECRET_PATHS:
            secret = path.read_text(encoding="utf-8").strip()
            if len(secret) < 24:
                errors.append(f"{key} must contain at least 24 non-whitespace characters")
            if "\n" in secret or "\r" in secret:
                errors.append(f"{key} must contain exactly one secret value")
            if os.name != "nt":
                mode = stat.S_IMODE(path.stat().st_mode)
                if mode & 0o077:
                    errors.append(f"{key} must not be group/world accessible (mode {mode:o})")
            else:
                warnings.append(
                    f"{key}: verify the Windows ACL grants read access only to the deploy identity"
                )


def _json_artifact(
    env_file: Path,
    values: dict[str, str],
    key: str,
    errors: list[str],
) -> dict[str, object] | None:
    value = values.get(key)
    if not value:
        return None
    path = _resolve_path(env_file, value)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        errors.append(f"{key} must contain valid UTF-8 JSON: {error}")
        return None
    if not isinstance(payload, dict):
        errors.append(f"{key} must contain a JSON object")
        return None
    return payload


def _validate_pinned_file(
    env_file: Path,
    values: dict[str, str],
    *,
    environment_key: str,
    reference: object,
    label: str,
    errors: list[str],
) -> None:
    """Verify a file-mounted catalogue input before Compose starts containers."""

    if not isinstance(reference, dict) or set(reference) != {"size_bytes", "sha256"}:
        errors.append(f"serving manifest requires an exact {label} content reference")
        return
    expected_size = reference.get("size_bytes")
    expected_sha256 = reference.get("sha256")
    if type(expected_size) is not int or expected_size < 0:
        errors.append(f"serving manifest {label} size_bytes must be non-negative")
        return
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        errors.append(f"serving manifest {label} sha256 must be lowercase hexadecimal")
        return
    configured_path = values.get(environment_key)
    if not configured_path:
        return
    path = _resolve_path(env_file, configured_path)
    if not path.is_file():
        return
    if path.stat().st_size != expected_size:
        errors.append(f"serving manifest {label} size does not match {environment_key}")
        return
    if sha256_file(path) != expected_sha256:
        errors.append(f"serving manifest {label} SHA-256 does not match {environment_key}")


def _validate_release_artifacts(
    env_file: Path,
    values: dict[str, str],
    errors: list[str],
) -> None:
    reviewed = _json_artifact(env_file, values, "PCBR_REVIEWED_MAPPING_FILE", errors)
    if reviewed is not None:
        if reviewed.get("schema_version") != "pc-build-recommender.reviewed-mappings.v1":
            errors.append("PCBR_REVIEWED_MAPPING_FILE has an unsupported schema_version")
        if not isinstance(reviewed.get("mappings"), list):
            errors.append("PCBR_REVIEWED_MAPPING_FILE requires a mappings array")

    artifact = _json_artifact(
        env_file,
        values,
        "PCBR_CATALOG_READINESS_ARTIFACT_FILE",
        errors,
    )
    if artifact is None:
        return
    data_version = values.get("PCBR_API_DATA_VERSION")
    readiness = artifact.get("readiness")
    if artifact.get("data_version") != data_version:
        errors.append("catalogue readiness artifact data_version must equal PCBR_API_DATA_VERSION")
    if artifact.get("database_upserted") is not False:
        errors.append("catalogue readiness artifact must be a read-only import report")
    if not isinstance(readiness, dict):
        errors.append("catalogue readiness artifact requires a readiness object")
        return
    if readiness.get("data_version") != data_version:
        errors.append("catalogue readiness data_version must equal PCBR_API_DATA_VERSION")
    if readiness.get("production_ready") is not True:
        errors.append("catalogue readiness artifact must record production_ready=true")
    blockers = readiness.get("production_blockers")
    if not isinstance(blockers, list) or blockers:
        errors.append("catalogue readiness artifact must contain an empty blocker list")


def _validate_serving_manifest(
    env_file: Path,
    values: dict[str, str],
    errors: list[str],
) -> None:
    release_value = values.get("PCBR_SERVING_RELEASE_DIR")
    if not release_value:
        return
    manifest_path = _resolve_path(env_file, release_value) / "serving-manifest.json"
    if not manifest_path.is_file():
        errors.append(
            f"PCBR_SERVING_RELEASE_DIR must contain serving-manifest.json: {manifest_path}"
        )
        return
    try:
        payload = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            parse_constant=_reject_nonfinite_json,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        errors.append(f"serving manifest must contain strict UTF-8 JSON: {error}")
        return
    if not isinstance(payload, dict):
        errors.append("serving manifest must contain a JSON object")
        return
    if set(payload) != _SERVING_MANIFEST_FIELDS:
        errors.append("serving manifest has an incomplete or unsupported field set")
        return
    if payload.get("schema_version") != _SERVING_MANIFEST_SCHEMA:
        errors.append("serving manifest has an unsupported schema_version")
    stored_hash = payload.get("content_sha256")
    unhashed = dict(payload)
    unhashed.pop("content_sha256", None)
    actual_hash = hashlib.sha256(
        json.dumps(
            unhashed,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    if stored_hash != actual_hash:
        errors.append("serving manifest content_sha256 verification failed")
    if stored_hash != values.get("PCBR_API_SERVING_MANIFEST_SHA256"):
        errors.append("serving manifest does not match PCBR_API_SERVING_MANIFEST_SHA256")
    if payload.get("catalog_data_version") != values.get("PCBR_API_DATA_VERSION"):
        errors.append("serving manifest catalogue version must equal PCBR_API_DATA_VERSION")
    _validate_pinned_file(
        env_file,
        values,
        environment_key="PCBR_BUILDCORES_CATALOG_FILE",
        reference=payload.get("catalog"),
        label="catalog",
        errors=errors,
    )
    catalog_inputs = payload.get("catalog_inputs")
    if not isinstance(catalog_inputs, dict) or set(catalog_inputs) != {
        "offers",
        "reviewed_mappings",
        "review_evidence",
    }:
        errors.append("serving manifest requires exact catalog_inputs references")
    else:
        _validate_pinned_file(
            env_file,
            values,
            environment_key="PCBR_GOVERNED_OFFERS_FILE",
            reference=catalog_inputs.get("offers"),
            label="governed offers",
            errors=errors,
        )
        _validate_pinned_file(
            env_file,
            values,
            environment_key="PCBR_REVIEWED_MAPPING_FILE",
            reference=catalog_inputs.get("reviewed_mappings"),
            label="reviewed mappings",
            errors=errors,
        )
        _validate_pinned_file(
            env_file,
            values,
            environment_key="PCBR_REVIEW_EVIDENCE_FILE",
            reference=catalog_inputs.get("review_evidence"),
            label="review evidence",
            errors=errors,
        )
    embedding = payload.get("embedding")
    encoder_bundle = embedding.get("encoder_bundle") if isinstance(embedding, dict) else None
    if not isinstance(encoder_bundle, dict) or set(encoder_bundle) != {
        "path",
        "sha256",
        "file_count",
        "size_bytes",
    }:
        errors.append("serving manifest requires an exact embedding.encoder_bundle reference")
    else:
        encoder_digest = values.get("PCBR_API_SEMANTIC_ENCODER_BUNDLE_SHA256", "")
        expected_relative_path = f"encoders/{encoder_digest}"
        if encoder_bundle.get("path") != expected_relative_path:
            errors.append(
                f"serving manifest encoder bundle path must equal {expected_relative_path}"
            )
        if encoder_bundle.get("sha256") != encoder_digest:
            errors.append(
                "serving manifest encoder bundle does not match "
                "PCBR_API_SEMANTIC_ENCODER_BUNDLE_SHA256"
            )
        file_count = encoder_bundle.get("file_count")
        size_bytes = encoder_bundle.get("size_bytes")
        if type(file_count) is not int or file_count < 1:
            errors.append("serving manifest encoder bundle file_count must be positive")
        elif type(size_bytes) is not int or size_bytes < 1:
            errors.append("serving manifest encoder bundle size_bytes must be positive")
        else:
            try:
                validate_encoder_bundle(
                    manifest_path.parent / expected_relative_path,
                    expected_sha256=encoder_digest,
                    expected_file_count=file_count,
                    expected_size_bytes=size_bytes,
                )
            except (EncoderBundleValidationError, OSError, ValueError) as error:
                errors.append(f"semantic encoder bundle failed validation: {error}")
    ranker = payload.get("ranker")
    if not isinstance(ranker, dict) or ranker.get("ranker_version") != values.get(
        "PCBR_API_RANKING_MODEL_VERSION"
    ):
        errors.append("serving manifest ranker version must equal PCBR_API_RANKING_MODEL_VERSION")


def validate(env_file: Path) -> ValidationResult:
    values, errors = _parse_env(env_file)
    warnings: list[str] = []
    _normalize_deprecated_offer_path(values, warnings)

    for key in sorted(REQUIRED_VALUES | IMAGE_VALUES):
        if not values.get(key):
            errors.append(f"missing required value {key}")
    for key in sorted(FORBIDDEN_CLEAR_TEXT & values.keys()):
        errors.append(f"{key} must not be stored in the production env file; use a secret file")

    for key, value in sorted(values.items()):
        if PLACEHOLDER_PATTERN.search(value):
            errors.append(f"{key} still contains a placeholder value")

    for key in sorted(IMAGE_VALUES):
        value = values.get(key, "")
        if value and not IMAGE_PATTERN.fullmatch(value):
            errors.append(f"{key} must be an immutable image@sha256:<64 hex> reference")

    for key in sorted(PORT_VALUES):
        port_value = values.get(key)
        if port_value is None:
            errors.append(f"missing required port {key}")
            continue
        try:
            port = int(port_value)
        except ValueError:
            errors.append(f"{key} must be an integer")
        else:
            if not 1 <= port <= 65535:
                errors.append(f"{key} must be between 1 and 65535")

    bounded_integer_values = {
        "PCBR_API_MAX_REQUEST_BODY_BYTES": (1024, 16 * 1024 * 1024),
        "PCBR_API_BUILD_GENERATION_MAX_CONCURRENCY": (1, 16),
        "PCBR_API_BUILD_GENERATION_MAX_QUEUE_SIZE": (0, 256),
        "PCBR_API_PIPELINE_OPERATIONS_WINDOW_HOURS": (1, 24 * 31),
    }
    for key, (minimum, maximum) in bounded_integer_values.items():
        raw_value = values.get(key)
        if raw_value is None:
            continue
        try:
            parsed_value = int(raw_value)
        except ValueError:
            errors.append(f"{key} must be an integer")
        else:
            if not minimum <= parsed_value <= maximum:
                errors.append(f"{key} must be between {minimum} and {maximum}")
    timeout_value = values.get("PCBR_API_BUILD_GENERATION_QUEUE_TIMEOUT_SECONDS")
    if timeout_value is not None:
        try:
            parsed_timeout = float(timeout_value)
        except ValueError:
            errors.append("PCBR_API_BUILD_GENERATION_QUEUE_TIMEOUT_SECONDS must be numeric")
        else:
            if not math.isfinite(parsed_timeout) or not 0 < parsed_timeout <= 60:
                errors.append(
                    "PCBR_API_BUILD_GENERATION_QUEUE_TIMEOUT_SECONDS must be greater than 0 "
                    "and at most 60"
                )

    for key in (
        "PCBR_POSTGRES_ADMIN_USER",
        "PCBR_DATABASE_NAME",
        "PCBR_MIGRATOR_USER",
        "PCBR_APP_USER",
        "PCBR_DAGSTER_DATABASE",
        "PCBR_DAGSTER_USER",
        "PCBR_MLFLOW_DATABASE",
        "PCBR_MLFLOW_USER",
        "PCBR_MONITOR_USER",
    ):
        value = values.get(key, "")
        if value and not IDENTIFIER_PATTERN.fullmatch(value):
            errors.append(f"{key} must be a lowercase PostgreSQL identifier")

    if values.get("PCBR_API_ENVIRONMENT") != "production":
        errors.append("PCBR_API_ENVIRONMENT must equal production")
    if values.get("PCBR_API_SERVICE_MODE") != "processed_catalog":
        errors.append("PCBR_API_SERVICE_MODE must equal processed_catalog")
    if values.get("PCBR_API_DOCS_ENABLED", "").casefold() != "false":
        errors.append("PCBR_API_DOCS_ENABLED must be false")
    if values.get("PCBR_API_SERVING_MANIFEST_PATH") != ("/run/pcbr-serving/serving-manifest.json"):
        errors.append("PCBR_API_SERVING_MANIFEST_PATH must use the read-only serving release mount")
    manifest_sha256 = values.get("PCBR_API_SERVING_MANIFEST_SHA256", "")
    if len(manifest_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in manifest_sha256
    ):
        errors.append("PCBR_API_SERVING_MANIFEST_SHA256 must be a lowercase SHA-256 digest")
    encoder_bundle_sha256 = values.get("PCBR_API_SEMANTIC_ENCODER_BUNDLE_SHA256", "")
    if len(encoder_bundle_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in encoder_bundle_sha256
    ):
        errors.append("PCBR_API_SEMANTIC_ENCODER_BUNDLE_SHA256 must be a lowercase SHA-256 digest")
    if values.get("PCBR_BIND_ADDRESS") not in {"127.0.0.1", "::1"}:
        errors.append("PCBR_BIND_ADDRESS must be loopback; public traffic belongs at TLS ingress")

    for key in ("PCBR_PUBLIC_WEB_URL", "PCBR_PUBLIC_API_URL"):
        value = values.get(key, "")
        if value and not _https_origin(value):
            errors.append(f"{key} must be an HTTPS origin without a path")

    try:
        origins = json.loads(values.get("PCBR_API_CORS_ORIGINS", ""))
    except json.JSONDecodeError:
        errors.append("PCBR_API_CORS_ORIGINS must be a JSON array")
    else:
        if not isinstance(origins, list) or not origins:
            errors.append("PCBR_API_CORS_ORIGINS must be a non-empty JSON array")
        elif any(not isinstance(origin, str) or not _https_origin(origin) for origin in origins):
            errors.append("every CORS origin must be an HTTPS origin without a path")
        elif "*" in origins:
            errors.append("wildcard CORS is forbidden")
        if values.get("PCBR_PUBLIC_WEB_URL") not in origins:
            errors.append("PCBR_API_CORS_ORIGINS must include PCBR_PUBLIC_WEB_URL")

    for key in (
        "PCBR_API_DATA_VERSION",
        "PCBR_API_RANKING_MODEL_VERSION",
        "PCBR_API_COMPATIBILITY_RULE_VERSION",
        "PCBR_API_SOLVER_VERSION",
    ):
        value = values.get(key, "")
        if value and UNSAFE_VERSION_PATTERN.search(value):
            errors.append(f"{key} must identify a reviewed immutable artifact or baseline")

    _validate_paths(env_file, values, errors, warnings)
    _validate_release_artifacts(env_file, values, errors)
    _validate_serving_manifest(env_file, values, errors)
    return {
        "status": "valid" if not errors else "invalid",
        "environment_file": str(env_file.resolve()),
        "checked_values": len(values),
        "errors": errors,
        "warnings": sorted(set(warnings)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    args = parser.parse_args()
    if not args.env_file.is_file():
        print(json.dumps({"status": "invalid", "errors": ["environment file not found"]}))
        return 2
    result = validate(args.env_file)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "valid" else 2


if __name__ == "__main__":
    sys.exit(main())
