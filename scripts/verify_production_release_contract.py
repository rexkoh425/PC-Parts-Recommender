"""Verify the rendered production release dependency and artifact contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_IMPORT_COMMAND = "scripts/import_catalog_release.py"
_REQUIRED_FLAGS = frozenset(
    {
        "--buildcores",
        "--offers",
        "--reviewed-mappings",
        "--review-evidence",
        "--serving-manifest",
        "--serving-manifest-sha256",
        "--readiness-artifact",
        "--expected-data-version",
    }
)


def _dependency_condition(service: dict[str, Any], dependency: str) -> str | None:
    depends_on = service.get("depends_on")
    if not isinstance(depends_on, dict):
        return None
    configuration = depends_on.get(dependency)
    if isinstance(configuration, dict):
        condition = configuration.get("condition")
        return condition if isinstance(condition, str) else None
    return None


def _volumes_by_target(service: dict[str, Any]) -> dict[str, dict[str, Any]]:
    volumes = service.get("volumes")
    if not isinstance(volumes, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for volume in volumes:
        if not isinstance(volume, dict):
            continue
        target = volume.get("target")
        if isinstance(target, str):
            result[target] = volume
    return result


def verify_compose_contract(document: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    services = document.get("services")
    if not isinstance(services, dict):
        return ["rendered Compose document requires a services object"]
    migrate = services.get("migrate")
    release = services.get("catalog-release")
    api = services.get("api")
    if not isinstance(migrate, dict):
        errors.append("migrate service is missing")
    if not isinstance(release, dict):
        errors.append("catalog-release service is missing")
    if not isinstance(api, dict):
        errors.append("api service is missing")
    if errors:
        return errors

    assert isinstance(release, dict)
    assert isinstance(api, dict)
    if _dependency_condition(release, "migrate") != "service_completed_successfully":
        errors.append("catalog-release must depend on successful migrate completion")
    if _dependency_condition(api, "catalog-release") != "service_completed_successfully":
        errors.append("api must depend on successful catalog-release completion")

    command = release.get("command")
    command_tokens = [str(token) for token in command] if isinstance(command, list) else []
    if _IMPORT_COMMAND not in command_tokens:
        errors.append("catalog-release must execute the pinned catalogue release importer")
    missing_flags = sorted(_REQUIRED_FLAGS - set(command_tokens))
    if missing_flags:
        errors.append("catalog-release command is missing flags: " + ", ".join(missing_flags))

    release_environment = release.get("environment")
    if (
        not isinstance(release_environment, dict)
        or str(release_environment.get("PCBR_RUN_MIGRATIONS")).casefold() != "false"
    ):
        errors.append("catalog-release must not rerun migrations")

    release_targets = _volumes_by_target(release)
    api_targets = _volumes_by_target(api)
    shared_targets = {
        "/run/pcbr-release/buildcores.jsonl": "/run/pcbr-data/buildcores.jsonl",
        "/run/pcbr-release/governed-offers.jsonl": ("/run/pcbr-data/governed-offers.jsonl"),
        "/run/pcbr-release/reviewed-mappings.json": ("/run/pcbr-data/reviewed-mappings.json"),
        "/run/pcbr-release/review-evidence.jsonl": ("/run/pcbr-data/review-evidence.jsonl"),
    }
    for release_target, api_target in shared_targets.items():
        release_volume = release_targets.get(release_target)
        api_volume = api_targets.get(api_target)
        if release_volume is None or api_volume is None:
            errors.append(f"release/API shared artifact mount is missing: {release_target}")
            continue
        if release_volume.get("source") != api_volume.get("source"):
            errors.append(f"release/API artifact sources differ: {release_target}")
        if not release_volume.get("read_only") or not api_volume.get("read_only"):
            errors.append(f"release/API artifact mounts must be read-only: {release_target}")
    readiness_volume = release_targets.get("/run/pcbr-release/catalog-readiness.json")
    if readiness_volume is None or not readiness_volume.get("read_only"):
        errors.append("catalog-release requires a read-only readiness artifact mount")
    release_serving_volume = release_targets.get("/run/pcbr-serving")
    serving_release_volume = api_targets.get("/run/pcbr-serving")
    if release_serving_volume is None or serving_release_volume is None:
        errors.append("catalog-release and API require the same serving release directory mount")
    elif release_serving_volume.get("source") != serving_release_volume.get("source"):
        errors.append("catalog-release and API serving release sources differ")
    elif not release_serving_volume.get("read_only") or not serving_release_volume.get("read_only"):
        errors.append("catalog-release and API serving release mounts must be read-only")

    api_environment = api.get("environment")
    if not isinstance(api_environment, dict):
        errors.append("api environment is missing")
    else:
        expected_paths = {
            "PCBR_API_BUILDCORES_CATALOG_PATH": "/run/pcbr-data/buildcores.jsonl",
            "PCBR_API_GOVERNED_OFFERS_PATH": "/run/pcbr-data/governed-offers.jsonl",
            "PCBR_API_REVIEWED_MAPPING_PATH": "/run/pcbr-data/reviewed-mappings.json",
            "PCBR_API_REVIEW_EVIDENCE_PATH": "/run/pcbr-data/review-evidence.jsonl",
            "PCBR_API_SERVING_MANIFEST_PATH": ("/run/pcbr-serving/serving-manifest.json"),
        }
        for key, expected in expected_paths.items():
            if api_environment.get(key) != expected:
                errors.append(f"api must consume the release artifact at {expected}")
        manifest_sha256 = api_environment.get("PCBR_API_SERVING_MANIFEST_SHA256")
        if (
            not isinstance(manifest_sha256, str)
            or len(manifest_sha256) != 64
            or any(character not in "0123456789abcdef" for character in manifest_sha256)
        ):
            errors.append("api must pin the serving manifest with a lowercase SHA-256")
        encoder_bundle_sha256 = api_environment.get("PCBR_API_SEMANTIC_ENCODER_BUNDLE_SHA256")
        if (
            not isinstance(encoder_bundle_sha256, str)
            or len(encoder_bundle_sha256) != 64
            or any(character not in "0123456789abcdef" for character in encoder_bundle_sha256)
        ):
            errors.append("api must pin the semantic encoder bundle with a lowercase SHA-256")
        expected_encoder_path = f"/run/pcbr-serving/encoders/{encoder_bundle_sha256}"
        if api_environment.get("PCBR_API_SEMANTIC_ENCODER_BUNDLE_PATH") != (expected_encoder_path):
            errors.append(
                "api semantic encoder bundle path must be content-addressed inside the "
                "read-only serving release mount"
            )
        if "--expected-data-version" in command_tokens:
            index = command_tokens.index("--expected-data-version")
            command_version = command_tokens[index + 1] if index + 1 < len(command_tokens) else None
            if command_version != api_environment.get("PCBR_API_DATA_VERSION"):
                errors.append("catalog-release and API data versions differ")
        if "--serving-manifest-sha256" in command_tokens:
            index = command_tokens.index("--serving-manifest-sha256")
            release_digest = command_tokens[index + 1] if index + 1 < len(command_tokens) else None
            if release_digest != manifest_sha256:
                errors.append("catalog-release and API serving manifest digests differ")
    return errors


def verify_powershell_sequence(script: str) -> list[str]:
    migrate = 'Invoke-Compose "up" "migrate"'
    release = 'Invoke-Compose "up" "catalog-release"'
    api = 'Invoke-Compose "up" "-d" "api" "web"'
    positions = (script.find(migrate), script.find(release), script.find(api))
    if any(position < 0 for position in positions):
        return ["DeployCore must explicitly run migrate, catalog-release, then API/web"]
    if not positions[0] < positions[1] < positions[2]:
        return ["DeployCore release order must be migrate -> catalog-release -> API/web"]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compose-json",
        type=argparse.FileType("r", encoding="utf-8"),
        default=sys.stdin,
    )
    parser.add_argument(
        "--production-script",
        type=Path,
        default=Path("scripts/production.ps1"),
    )
    args = parser.parse_args()
    try:
        document = json.load(args.compose_json)
    except json.JSONDecodeError as error:
        print(json.dumps({"status": "invalid", "errors": [f"invalid Compose JSON: {error}"]}))
        return 2
    if not isinstance(document, dict):
        errors = ["rendered Compose document must be a JSON object"]
    else:
        errors = verify_compose_contract(document)
    errors.extend(verify_powershell_sequence(args.production_script.read_text(encoding="utf-8")))
    print(
        json.dumps(
            {"status": "valid" if not errors else "invalid", "errors": errors},
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
