from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.create_production_contract_fixture import create_production_contract_fixture
from scripts.validate_production_env import _parse_env, validate


def _write_valid_environment(tmp_path: Path) -> Path:
    return create_production_contract_fixture(
        template=Path(".env.production.example"),
        fixture_root=tmp_path / "fixture",
        env_file=tmp_path / ".env.production",
    )


def test_valid_production_environment_passes(tmp_path: Path) -> None:
    result = validate(_write_valid_environment(tmp_path))

    assert result["status"] == "valid"
    assert result["errors"] == []


def test_contract_fixture_refuses_to_overwrite_existing_root(tmp_path: Path) -> None:
    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir()

    with pytest.raises(ValueError, match="fixture root must not already exist"):
        create_production_contract_fixture(
            template=Path(".env.production.example"),
            fixture_root=fixture_root,
            env_file=tmp_path / ".env.production",
        )


@pytest.mark.parametrize(
    ("key", "invalid_value", "expected_error"),
    [
        ("PCBR_API_MAX_REQUEST_BODY_BYTES", "1023", "between 1024 and 16777216"),
        ("PCBR_API_MAX_REQUEST_BODY_BYTES", "16777217", "between 1024 and 16777216"),
        ("PCBR_API_BUILD_GENERATION_MAX_CONCURRENCY", "0", "between 1 and 16"),
        ("PCBR_API_BUILD_GENERATION_MAX_CONCURRENCY", "17", "between 1 and 16"),
        ("PCBR_API_BUILD_GENERATION_MAX_QUEUE_SIZE", "-1", "between 0 and 256"),
        ("PCBR_API_BUILD_GENERATION_MAX_QUEUE_SIZE", "257", "between 0 and 256"),
        (
            "PCBR_API_PIPELINE_OPERATIONS_WINDOW_HOURS",
            "0",
            "between 1 and 744",
        ),
        (
            "PCBR_API_PIPELINE_OPERATIONS_WINDOW_HOURS",
            "745",
            "between 1 and 744",
        ),
        (
            "PCBR_API_BUILD_GENERATION_QUEUE_TIMEOUT_SECONDS",
            "0",
            "greater than 0 and at most 60",
        ),
        (
            "PCBR_API_BUILD_GENERATION_QUEUE_TIMEOUT_SECONDS",
            "61",
            "greater than 0 and at most 60",
        ),
    ],
)
def test_production_resource_bounds_fail_closed(
    tmp_path: Path,
    key: str,
    invalid_value: str,
    expected_error: str,
) -> None:
    env_file = _write_valid_environment(tmp_path)
    content = env_file.read_text(encoding="utf-8")
    lines = [
        f"{key}={invalid_value}" if line.startswith(f"{key}=") else line
        for line in content.splitlines()
    ]
    env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = validate(env_file)

    assert result["status"] == "invalid"
    assert any(expected_error in error for error in result["errors"])


def test_deprecated_offer_file_key_is_accepted_with_warning(tmp_path: Path) -> None:
    env_file = _write_valid_environment(tmp_path)
    content = env_file.read_text(encoding="utf-8").replace(
        "PCBR_GOVERNED_OFFERS_FILE=",
        "PCBR_DYNACORE_OFFERS_FILE=",
    )
    env_file.write_text(content, encoding="utf-8")

    result = validate(env_file)

    assert result["status"] == "valid"
    assert any(
        "PCBR_DYNACORE_OFFERS_FILE is deprecated" in warning for warning in result["warnings"]
    )


def test_canonical_offer_file_key_wins_over_deprecated_key(tmp_path: Path) -> None:
    env_file = _write_valid_environment(tmp_path)
    env_file.write_text(
        env_file.read_text(encoding="utf-8")
        + "PCBR_DYNACORE_OFFERS_FILE=C:/ignored/missing.jsonl\n",
        encoding="utf-8",
    )

    result = validate(env_file)

    assert result["status"] == "valid"
    assert any("deprecated and ignored" in warning for warning in result["warnings"])


def test_example_fails_closed_with_placeholders_and_missing_paths() -> None:
    result = validate(Path(".env.production.example"))

    assert result["status"] == "invalid"
    assert any("placeholder" in error for error in result["errors"])
    assert any("does not name an existing" in error for error in result["errors"])


def test_cleartext_database_url_and_wildcard_cors_are_rejected(tmp_path: Path) -> None:
    env_file = _write_valid_environment(tmp_path)
    content = env_file.read_text(encoding="utf-8")
    content = content.replace(
        'PCBR_API_CORS_ORIGINS=["https://pcbr.invalid"]',
        'PCBR_API_CORS_ORIGINS=["*"]',
    )
    env_file.write_text(
        content
        + "DATABASE_URL=postgresql://user:password@database/app\n"
        + "PCBR_API_ADMIN_TOKEN=must-not-be-stored-in-an-env-file\n",
        encoding="utf-8",
    )

    result = validate(env_file)

    assert result["status"] == "invalid"
    assert any("DATABASE_URL must not be stored" in error for error in result["errors"])
    assert any("PCBR_API_ADMIN_TOKEN must not be stored" in error for error in result["errors"])
    assert any("CORS" in error for error in result["errors"])


def test_readiness_artifact_version_mismatch_is_rejected(tmp_path: Path) -> None:
    env_file = _write_valid_environment(tmp_path)
    values, parse_errors = _parse_env(env_file)
    assert not parse_errors
    readiness_path = Path(values["PCBR_CATALOG_READINESS_ARTIFACT_FILE"])
    payload = json.loads(readiness_path.read_text(encoding="utf-8"))
    payload["data_version"] = "different-data-version"
    readiness_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate(env_file)

    assert result["status"] == "invalid"
    assert any("data_version" in error for error in result["errors"])


def test_semantic_encoder_bundle_tampering_is_rejected(tmp_path: Path) -> None:
    env_file = _write_valid_environment(tmp_path)
    values, parse_errors = _parse_env(env_file)
    assert not parse_errors
    bundle_file = (
        Path(values["PCBR_SERVING_RELEASE_DIR"])
        / "encoders"
        / values["PCBR_API_SEMANTIC_ENCODER_BUNDLE_SHA256"]
        / "modules.json"
    )
    bundle_file.write_bytes(bundle_file.read_bytes() + b"tampered")

    result = validate(env_file)

    assert result["status"] == "invalid"
    assert any("semantic encoder bundle failed validation" in error for error in result["errors"])


def test_serving_manifest_tampering_is_rejected_by_preflight(tmp_path: Path) -> None:
    env_file = _write_valid_environment(tmp_path)
    values, parse_errors = _parse_env(env_file)
    assert not parse_errors
    manifest_path = Path(values["PCBR_SERVING_RELEASE_DIR"]) / "serving-manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["ranker"]["ranker_version"] = "tampered-ranker"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate(env_file)

    assert result["status"] == "invalid"
    assert any("content_sha256 verification failed" in error for error in result["errors"])


def test_review_evidence_tampering_is_rejected_by_preflight(tmp_path: Path) -> None:
    env_file = _write_valid_environment(tmp_path)
    values, parse_errors = _parse_env(env_file)
    assert not parse_errors
    review_evidence_path = Path(values["PCBR_REVIEW_EVIDENCE_FILE"])
    review_evidence_path.write_text("[]\n", encoding="utf-8")

    result = validate(env_file)

    assert result["status"] == "invalid"
    assert any("review evidence SHA-256" in error for error in result["errors"])


def test_non_ready_release_artifact_is_rejected(tmp_path: Path) -> None:
    env_file = _write_valid_environment(tmp_path)
    values, parse_errors = _parse_env(env_file)
    assert not parse_errors
    readiness_path = Path(values["PCBR_CATALOG_READINESS_ARTIFACT_FILE"])
    payload = json.loads(readiness_path.read_text(encoding="utf-8"))
    payload["readiness"]["production_ready"] = False
    payload["readiness"]["production_blockers"] = ["rights missing"]
    readiness_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate(env_file)

    assert result["status"] == "invalid"
    assert any("production_ready" in error for error in result["errors"])
