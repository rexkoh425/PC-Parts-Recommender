"""Create an isolated, non-production fixture for production contract validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pc_build_recommender.evaluation.manifest import sha256_file, json_sha256
from pc_build_recommender.retrieval import inspect_encoder_bundle
from scripts.validate_production_env import _parse_env, validate


def create_production_contract_fixture(
    *,
    template: Path,
    fixture_root: Path,
    env_file: Path,
) -> Path:
    """Materialize a fixture that satisfies the current production preflight contract."""

    template = template.resolve()
    fixture_root = fixture_root.resolve()
    env_file = env_file.resolve()
    if not template.is_file():
        raise ValueError(f"production environment template does not exist: {template}")
    if fixture_root.exists():
        raise ValueError(f"fixture root must not already exist: {fixture_root}")

    values, parse_errors = _parse_env(template)
    if parse_errors:
        raise ValueError(f"production environment template is invalid: {parse_errors}")

    fixture_root.mkdir(parents=True)
    image_digest = "b" * 64
    for key in (key for key in values if key.endswith("_IMAGE")):
        values[key] = f"registry.invalid/pcbr/{key.casefold()}@sha256:{image_digest}"
    values.update(
        {
            "PCBR_DEPLOYMENT_ID": "20260722T120000Z-cafebabe",
            "PCBR_PUBLIC_WEB_URL": "https://pcbr.invalid",
            "PCBR_PUBLIC_API_URL": "https://api.pcbr.invalid",
            "PCBR_API_CORS_ORIGINS": '["https://pcbr.invalid"]',
            "PCBR_API_DATA_VERSION": "catalog-20260722-sha256-cafebabe",
            "PCBR_API_RANKING_MODEL_VERSION": "deterministic-baseline-v1",
        }
    )

    for key in (
        key
        for key in values
        if key.endswith("_PASSWORD_FILE") or key == "PCBR_API_ADMIN_TOKEN_FILE"
    ):
        secret = fixture_root / f"{key.casefold()}.txt"
        secret.write_text(f"contract-only-{key}-0123456789abcdef", encoding="utf-8")
        secret.chmod(0o600)
        values[key] = str(secret)

    for key in (
        "PCBR_BUILDCORES_CATALOG_FILE",
        "PCBR_GOVERNED_OFFERS_FILE",
        "PCBR_REVIEW_EVIDENCE_FILE",
    ):
        data_file = fixture_root / f"{key.casefold()}.jsonl"
        data_file.write_text("{}\n", encoding="utf-8")
        values[key] = str(data_file)

    reviewed_mapping = fixture_root / "reviewed-mappings.json"
    reviewed_mapping.write_text(
        json.dumps(
            {
                "schema_version": "pc-build-recommender.reviewed-mappings.v1",
                "mappings": [],
            }
        ),
        encoding="utf-8",
    )
    values["PCBR_REVIEWED_MAPPING_FILE"] = str(reviewed_mapping)

    readiness = fixture_root / "catalog-readiness.json"
    readiness.write_text(
        json.dumps(
            {
                "data_version": values["PCBR_API_DATA_VERSION"],
                "database_upserted": False,
                "readiness": {
                    "data_version": values["PCBR_API_DATA_VERSION"],
                    "production_ready": True,
                    "production_blockers": [],
                },
            }
        ),
        encoding="utf-8",
    )
    values["PCBR_CATALOG_READINESS_ARTIFACT_FILE"] = str(readiness)

    serving_release = fixture_root / "serving-release"
    serving_release.mkdir()
    encoder_staging = serving_release / "encoders" / "staging"
    encoder_staging.mkdir(parents=True)
    (encoder_staging / "modules.json").write_text("[]\n", encoding="utf-8")
    encoder_identity = inspect_encoder_bundle(encoder_staging)
    encoder_bundle = encoder_staging.with_name(encoder_identity.sha256)
    encoder_staging.rename(encoder_bundle)

    serving_manifest: dict[str, object] = {
        "schema_version": "pc-build-recommender.serving-release.v3",
        "catalog_data_version": values["PCBR_API_DATA_VERSION"],
        "catalog": {
            "size_bytes": Path(values["PCBR_BUILDCORES_CATALOG_FILE"]).stat().st_size,
            "sha256": sha256_file(values["PCBR_BUILDCORES_CATALOG_FILE"]),
        },
        "catalog_inputs": {
            "offers": {
                "size_bytes": Path(values["PCBR_GOVERNED_OFFERS_FILE"]).stat().st_size,
                "sha256": sha256_file(values["PCBR_GOVERNED_OFFERS_FILE"]),
            },
            "reviewed_mappings": {
                "size_bytes": Path(values["PCBR_REVIEWED_MAPPING_FILE"]).stat().st_size,
                "sha256": sha256_file(values["PCBR_REVIEWED_MAPPING_FILE"]),
            },
            "review_evidence": {
                "size_bytes": Path(values["PCBR_REVIEW_EVIDENCE_FILE"]).stat().st_size,
                "sha256": sha256_file(values["PCBR_REVIEW_EVIDENCE_FILE"]),
            },
        },
        "embedding": {
            "encoder_bundle": {
                "path": f"encoders/{encoder_identity.sha256}",
                "sha256": encoder_identity.sha256,
                "file_count": encoder_identity.file_count,
                "size_bytes": encoder_identity.size_bytes,
            }
        },
        "entity_resolution": {},
        "retrieval": {},
        "ranker": {"ranker_version": values["PCBR_API_RANKING_MODEL_VERSION"]},
        "ranker_promotion": {},
        "performance": [],
    }
    serving_manifest["content_sha256"] = json_sha256(serving_manifest)
    (serving_release / "serving-manifest.json").write_text(
        json.dumps(serving_manifest),
        encoding="utf-8",
    )
    values["PCBR_SERVING_RELEASE_DIR"] = str(serving_release)
    values["PCBR_API_SERVING_MANIFEST_SHA256"] = str(serving_manifest["content_sha256"])
    values["PCBR_API_SEMANTIC_ENCODER_BUNDLE_SHA256"] = encoder_identity.sha256

    for key in (
        "PCBR_PIPELINE_DATA_DIR",
        "PCBR_PIPELINE_OPERATIONS_DIR",
        "PCBR_PIPELINE_ARTIFACTS_DIR",
        "PCBR_MLFLOW_ARTIFACTS_DIR",
        "PCBR_BACKUP_DIR",
    ):
        directory = fixture_root / key.casefold()
        directory.mkdir()
        values[key] = str(directory)

    env_file.parent.mkdir(parents=True, exist_ok=True)
    env_file.write_text(
        "\n".join(f"{key}={value}" for key, value in values.items()) + "\n",
        encoding="utf-8",
    )
    result = validate(env_file)
    if result["status"] != "valid":
        raise ValueError(f"generated production contract fixture is invalid: {result['errors']}")
    return env_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=Path(".env.production.example"))
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    args = parser.parse_args()
    output = create_production_contract_fixture(
        template=args.template,
        fixture_root=args.fixture_root,
        env_file=args.env_file,
    )
    print(json.dumps({"status": "created", "environment_file": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
