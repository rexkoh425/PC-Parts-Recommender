from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from scripts import import_catalog_release
from scripts.verify_production_release_contract import (
    _ALLOWED_IMPORT_FLAGS,
    verify_compose_contract,
    verify_powershell_sequence,
)


def _volume(source: str, target: str) -> dict[str, object]:
    return {"source": source, "target": target, "read_only": True}


def _valid_contract() -> dict[str, object]:
    data_version = "processed-release-version"
    command = [
        "python",
        "-m",
        "scripts.import_catalog_release",
        "--buildcores",
        "/run/pcbr-release/buildcores.jsonl",
        "--offers",
        "/run/pcbr-release/governed-offers.jsonl",
        "--reviewed-mappings",
        "/run/pcbr-release/reviewed-mappings.json",
        "--review-evidence",
        "/run/pcbr-release/review-evidence.jsonl",
        "--serving-manifest",
        "/run/pcbr-serving/serving-manifest.json",
        "--serving-manifest-sha256",
        "a" * 64,
        "--source-registry",
        "/run/pcbr-source/source-registry.yaml",
        "--source-trust-root-sha256",
        "b" * 64,
        "--readiness-artifact",
        "/run/pcbr-release/catalog-readiness.json",
        "--expected-data-version",
        data_version,
    ]
    release_volumes = [
        _volume("/host/buildcores", "/run/pcbr-release/buildcores.jsonl"),
        _volume("/host/offers", "/run/pcbr-release/governed-offers.jsonl"),
        _volume("/host/reviewed", "/run/pcbr-release/reviewed-mappings.json"),
        _volume("/host/review-evidence", "/run/pcbr-release/review-evidence.jsonl"),
        _volume("/host/readiness", "/run/pcbr-release/catalog-readiness.json"),
        _volume("/host/serving-release", "/run/pcbr-serving"),
        _volume("/host/current-source-registry", "/run/pcbr-source/source-registry.yaml"),
    ]
    api_volumes = [
        _volume("/host/buildcores", "/run/pcbr-data/buildcores.jsonl"),
        _volume("/host/offers", "/run/pcbr-data/governed-offers.jsonl"),
        _volume("/host/reviewed", "/run/pcbr-data/reviewed-mappings.json"),
        _volume("/host/review-evidence", "/run/pcbr-data/review-evidence.jsonl"),
        _volume("/host/serving-release", "/run/pcbr-serving"),
        _volume("/host/current-source-registry", "/run/pcbr-source/source-registry.yaml"),
    ]
    dagster_volumes = [
        _volume("/host/current-source-registry", "/app/config/source_registry.yaml"),
    ]
    return {
        "services": {
            "migrate": {},
            "catalog-release": {
                "command": command,
                "depends_on": {"migrate": {"condition": "service_completed_successfully"}},
                "environment": {"PCBR_RUN_MIGRATIONS": "false"},
                "volumes": release_volumes,
            },
            "api": {
                "depends_on": {"catalog-release": {"condition": "service_completed_successfully"}},
                "environment": {
                    "PCBR_API_DATA_VERSION": data_version,
                    "PCBR_API_BUILDCORES_CATALOG_PATH": "/run/pcbr-data/buildcores.jsonl",
                    "PCBR_API_GOVERNED_OFFERS_PATH": ("/run/pcbr-data/governed-offers.jsonl"),
                    "PCBR_API_REVIEWED_MAPPING_PATH": "/run/pcbr-data/reviewed-mappings.json",
                    "PCBR_API_REVIEW_EVIDENCE_PATH": "/run/pcbr-data/review-evidence.jsonl",
                    "PCBR_API_SERVING_MANIFEST_PATH": ("/run/pcbr-serving/serving-manifest.json"),
                    "PCBR_API_SERVING_MANIFEST_SHA256": "a" * 64,
                    "PCBR_API_SOURCE_REGISTRY_PATH": "/run/pcbr-source/source-registry.yaml",
                    "PCBR_API_SOURCE_TRUST_ROOT_SHA256": "b" * 64,
                    "PCBR_API_SEMANTIC_ENCODER_BUNDLE_PATH": (
                        "/run/pcbr-serving/encoders/" + "c" * 64
                    ),
                    "PCBR_API_SEMANTIC_ENCODER_BUNDLE_SHA256": "c" * 64,
                },
                "volumes": api_volumes,
            },
            "dagster-code": {
                "environment": {
                    "SOURCE_REGISTRY_PATH": "/app/config/source_registry.yaml",
                },
                "volumes": dagster_volumes,
            },
        }
    }


def test_rendered_contract_enforces_migrate_release_api_sequence() -> None:
    assert verify_compose_contract(_valid_contract()) == []


def test_release_contract_flag_allowlist_matches_the_importer_parser() -> None:
    parser_flags = {
        option
        for action in import_catalog_release._parser()._actions
        for option in action.option_strings
        if option.startswith("--") and option != "--help"
    }

    assert parser_flags == _ALLOWED_IMPORT_FLAGS


def test_api_cannot_bypass_failed_catalog_release() -> None:
    document = deepcopy(_valid_contract())
    services = document["services"]
    assert isinstance(services, dict)
    api = services["api"]
    assert isinstance(api, dict)
    api["depends_on"] = {"migrate": {"condition": "service_completed_successfully"}}

    errors = verify_compose_contract(document)

    assert any("api must depend" in error for error in errors)


def test_catalog_release_rejects_flags_not_supported_by_the_dedicated_importer() -> None:
    document = deepcopy(_valid_contract())
    services = document["services"]
    assert isinstance(services, dict)
    release = services["catalog-release"]
    assert isinstance(release, dict)
    command = release["command"]
    assert isinstance(command, list)
    command.append("--require-production-ready")

    errors = verify_compose_contract(document)

    assert any("unsupported importer flags" in error for error in errors)


def test_catalog_release_requires_the_importable_module_entrypoint() -> None:
    document = deepcopy(_valid_contract())
    services = document["services"]
    assert isinstance(services, dict)
    release = services["catalog-release"]
    assert isinstance(release, dict)
    command = release["command"]
    assert isinstance(command, list)
    command[:3] = ["python", "scripts/import_catalog_release.py", "--buildcores"]

    errors = verify_compose_contract(document)

    assert any("as a module" in error for error in errors)


def test_api_requires_read_only_content_addressed_serving_release() -> None:
    document = deepcopy(_valid_contract())
    services = document["services"]
    assert isinstance(services, dict)
    api = services["api"]
    assert isinstance(api, dict)
    volumes = api["volumes"]
    assert isinstance(volumes, list)
    api["volumes"] = [
        volume
        for volume in volumes
        if not isinstance(volume, dict) or volume.get("target") != "/run/pcbr-serving"
    ]
    environment = api["environment"]
    assert isinstance(environment, dict)
    environment.pop("PCBR_API_SERVING_MANIFEST_PATH")

    errors = verify_compose_contract(document)

    assert any("serving release directory" in error for error in errors)
    assert any("serving-manifest.json" in error for error in errors)


def test_dagster_code_requires_the_same_read_only_current_registry() -> None:
    document = deepcopy(_valid_contract())
    services = document["services"]
    assert isinstance(services, dict)
    dagster_code = services["dagster-code"]
    assert isinstance(dagster_code, dict)
    dagster_code["volumes"] = [
        _volume("/host/stale-registry", "/app/config/source_registry.yaml")
    ]

    errors = verify_compose_contract(document)

    assert any("dagster-code current source registry source differs" in error for error in errors)


def test_dagster_code_registry_mount_must_be_read_only_and_match_environment() -> None:
    document = deepcopy(_valid_contract())
    services = document["services"]
    assert isinstance(services, dict)
    dagster_code = services["dagster-code"]
    assert isinstance(dagster_code, dict)
    volumes = dagster_code["volumes"]
    assert isinstance(volumes, list)
    volume = volumes[0]
    assert isinstance(volume, dict)
    volume["read_only"] = False
    environment = dagster_code["environment"]
    assert isinstance(environment, dict)
    environment["SOURCE_REGISTRY_PATH"] = "/app/config/unmounted.yaml"

    errors = verify_compose_contract(document)

    assert any("mount must be read-only" in error for error in errors)
    assert any("must read the mounted" in error for error in errors)


def test_api_requires_content_addressed_semantic_encoder_bundle() -> None:
    document = deepcopy(_valid_contract())
    services = document["services"]
    assert isinstance(services, dict)
    api = services["api"]
    assert isinstance(api, dict)
    environment = api["environment"]
    assert isinstance(environment, dict)
    environment["PCBR_API_SEMANTIC_ENCODER_BUNDLE_PATH"] = (
        "/run/pcbr-serving/encoders/not-the-pinned-digest"
    )

    errors = verify_compose_contract(document)

    assert any("semantic encoder bundle path" in error for error in errors)


def test_powershell_deploy_core_sequence_is_explicit() -> None:
    script = "\n".join(
        (
            'Invoke-Compose "up" "migrate"',
            'Invoke-Compose "up" "catalog-release"',
            'Invoke-Compose "up" "-d" "api" "web"',
        )
    )

    assert verify_powershell_sequence(script) == []


def test_ci_production_contract_installs_and_uses_shared_v4_fixture() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    production_job = workflow.split("  production-contract:", maxsplit=1)[1]

    assert "astral-sh/setup-uv@v6" in production_job
    assert "uv sync --locked" in production_job
    assert "python -m scripts.create_production_contract_fixture" in production_job
    assert "uv run python scripts/validate_production_env.py" in production_job
    assert "serving-release.v1" not in production_job
