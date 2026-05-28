from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.loadtest.profile import (
    PROFILE_SCHEMA_VERSION,
    REMOTE_CONFIRMATION_VALUE,
    LoadProfileError,
    load_profile,
    normalise_target_url,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _profile_payload() -> dict[str, object]:
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "profile_name": "fixture-profile",
        "reportability": "development_only",
        "search": {
            "method": "POST",
            "path": "/v1/products/search",
            "body": {"query": "GPU", "limit": 20},
            "expected_statuses": [200],
            "task_weight": 4,
        },
        "build": {
            "method": "POST",
            "path": "/v1/builds/generate",
            "body": {"budget_sgd": 2000},
            "expected_statuses": [200, 429],
            "task_weight": 1,
        },
    }


def _write_profile(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_checked_in_development_profile_has_bounded_search_and_build_tasks() -> None:
    profile = load_profile(REPOSITORY_ROOT / "scripts" / "loadtest" / "development-profile.json")

    assert profile.profile_name == "development-demo-api-mix"
    assert profile.reportability == "development_only"
    assert profile.search.request_name == "/v1/products/search"
    assert profile.build.request_name == "/v1/builds/generate"
    assert profile.search.task_weight == 4
    assert profile.build.task_weight == 1
    assert len(profile.sha256) == 64


def test_profile_rejects_remote_path_and_unknown_fields(tmp_path: Path) -> None:
    payload = _profile_payload()
    search = payload["search"]
    assert isinstance(search, dict)
    search["path"] = "https://outside.example.test/v1/products/search"
    with pytest.raises(LoadProfileError, match="absolute API path"):
        load_profile(_write_profile(tmp_path, payload))

    payload = _profile_payload()
    build = payload["build"]
    assert isinstance(build, dict)
    build["path"] = "/v1/interactions"
    with pytest.raises(LoadProfileError, match="must be '/v1/builds/generate'"):
        load_profile(_write_profile(tmp_path, payload))

    payload = _profile_payload()
    payload["unreviewed_headers"] = {"Authorization": "secret"}
    with pytest.raises(LoadProfileError, match="fields do not match"):
        load_profile(_write_profile(tmp_path, payload))


def test_target_validation_allows_loopback_but_requires_https_and_confirmation_remotely() -> None:
    assert normalise_target_url("http://127.0.0.1:8000/") == "http://127.0.0.1:8000"
    assert normalise_target_url("https://localhost") == "https://localhost"

    with pytest.raises(LoadProfileError, match="must use HTTPS"):
        normalise_target_url("http://load.example.test")
    with pytest.raises(LoadProfileError, match="PCBR_LOAD_CONFIRM"):
        normalise_target_url("https://load.example.test")

    assert normalise_target_url(
        "https://load.example.test",
        confirmation=REMOTE_CONFIRMATION_VALUE,
    ) == "https://load.example.test"
