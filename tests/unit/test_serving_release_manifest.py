from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypedDict

import numpy as np
import pytest
from numpy.typing import NDArray
from services.api import serving_release
from sqlalchemy.orm import sessionmaker

from pc_build_recommender.application import ServingConfigurationError
from pc_build_recommender.evaluation.manifest import sha256_file, sha256_json
from pc_build_recommender.ranking import RankerPromotionPolicy
from pc_build_recommender.retrieval import (
    SentenceTransformerEmbeddingEncoder,
    embedding_encoder_fingerprint,
    inspect_encoder_bundle,
)


def _write(root: Path, relative: str, content: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _reference(root: Path, path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    unhashed = dict(payload)
    unhashed.pop("content_sha256", None)
    payload["content_sha256"] = sha256_json(unhashed)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _release_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, dict[str, Path], dict[str, str]]:
    root = tmp_path / "release"
    root.mkdir()
    catalog = _write(tmp_path, "catalog.jsonl", '{"product_id":"cpu-1"}\n')
    offers = _write(tmp_path, "offers.jsonl", '{"listing_id":"offer-1"}\n')
    reviewed = _write(tmp_path, "reviewed.json", '{"mappings":[]}\n')
    review_evidence = _write(tmp_path, "review-evidence.jsonl", "")
    files = {
        "embedding": _write(root, "embedding/manifest.json", "{}"),
        "retrieval": _write(root, "retrieval/comparison.json", "{}"),
        "ranker_model": _write(root, "ranker/ranker.txt", "ranker-model"),
        "ranker_manifest": _write(
            root,
            "ranker/ranker.txt.artifact-manifest.json",
            "{}",
        ),
        "ranking_bundle_manifest": _write(
            root, "ranker/evaluation/manifest.json", "{}"
        ),
        "performance": _write(root, "performance/gpu-ai/artifact_manifest.json", "{}"),
        "er_metadata": _write(root, "entity-resolution/model/metadata.json", "{}"),
        "er_model": _write(root, "entity-resolution/model/model.txt", "er-model"),
        "er_evidence": _write(
            root,
            "entity-resolution/model/serving_evidence.json",
            "{}",
        ),
        "er_evaluation": _write(root, "entity-resolution/evaluation.json", "{}"),
        "er_policy": _write(root, "entity-resolution/policy.json", "{}"),
        "er_rights": _write(root, "entity-resolution/rights.json", "{}"),
        "source_raw": _write(root, "source-input/raw-snapshot.csv", "fixture-source-row\n"),
        "source_rejections": _write(root, "source-input/rejections.jsonl", ""),
        "source_registry": _write(
            root,
            "source-input/source-registry.yaml",
            "schema_version: pc-build-recommender.source-registry.v1\n",
        ),
    }
    source_manifest_staging = _write(
        root,
        "source-releases/fixture/staging/manifest.json",
        '{"fixture":"signed-source-release"}\n',
    )
    source_manifest_sha256 = sha256_file(source_manifest_staging)
    source_manifest_directory = source_manifest_staging.parent.with_name(source_manifest_sha256)
    source_manifest_staging.parent.rename(source_manifest_directory)
    files["source_manifest"] = source_manifest_directory / "manifest.json"
    encoder_staging_file = _write(root, "encoders/staging/modules.json", "[]\n")
    encoder_identity = inspect_encoder_bundle(encoder_staging_file.parent)
    encoder_bundle = encoder_staging_file.parent.with_name(encoder_identity.sha256)
    encoder_staging_file.parent.rename(encoder_bundle)
    files["encoder_bundle_file"] = encoder_bundle / "modules.json"
    er_identity = {
        "artifact_core_sha256": "1" * 64,
        "model_file_sha256": sha256_file(files["er_model"]),
        "metadata_sha256": sha256_file(files["er_metadata"]),
        "calibrator_sha256": "2" * 64,
        "serving_evidence_sha256": sha256_file(files["er_evidence"]),
        "model_release_sha256": "3" * 64,
        "evaluation_sha256": sha256_file(files["er_evaluation"]),
        "policy_sha256": sha256_file(files["er_policy"]),
        "rights_sha256": sha256_file(files["er_rights"]),
        "binding_sha256": "4" * 64,
    }
    encoder = SentenceTransformerEmbeddingEncoder(
        "sentence-transformers/all-MiniLM-L6-v2",
        revision="0123456789abcdef",
        device="cpu",
    )
    payload: dict[str, Any] = {
        "schema_version": serving_release.SERVING_RELEASE_SCHEMA_VERSION,
        "catalog_data_version": "catalog-v1",
        "catalog": {
            "size_bytes": catalog.stat().st_size,
            "sha256": sha256_file(catalog),
        },
        "catalog_inputs": {
            "offers": {
                "size_bytes": offers.stat().st_size,
                "sha256": sha256_file(offers),
            },
            "reviewed_mappings": {
                "size_bytes": reviewed.stat().st_size,
                "sha256": sha256_file(reviewed),
            },
            "review_evidence": {
                "size_bytes": review_evidence.stat().st_size,
                "sha256": sha256_file(review_evidence),
            },
        },
        "source_release": {
            "manifest": _reference(root, files["source_manifest"]),
            "raw_snapshot": _reference(root, files["source_raw"]),
            "rejections": _reference(root, files["source_rejections"]),
            "current_source_registry": {
                "size_bytes": files["source_registry"].stat().st_size,
                "sha256": sha256_file(files["source_registry"]),
            },
            "expected_trust_root_sha256": "5" * 64,
        },
        "embedding": {
            "artifact_manifest": _reference(root, files["embedding"]),
            "data_version": "embedding-data-v1",
            "index_version": "embedding-index-v1",
            "embedding_model": encoder.model_name,
            "encoder_revision": encoder.revision,
            "encoder_fingerprint": embedding_encoder_fingerprint(encoder),
            "dataset_content_hash": "d" * 64,
            "manifest_schema_version": "product-embedding-index-manifest-v1",
            "device": "cpu",
            "batch_size": 32,
            "rrf_k": 60,
            "encoder_bundle": {
                "path": encoder_bundle.relative_to(root).as_posix(),
                "sha256": encoder_identity.sha256,
                "file_count": encoder_identity.file_count,
                "size_bytes": encoder_identity.size_bytes,
            },
        },
        "retrieval": {
            "comparison_report": _reference(root, files["retrieval"]),
            "evaluation_model": "hybrid",
        },
        "entity_resolution": {
            "metadata": _reference(root, files["er_metadata"]),
            "model": _reference(root, files["er_model"]),
            "serving_evidence": _reference(root, files["er_evidence"]),
            "evaluation": _reference(root, files["er_evaluation"]),
            "policy": _reference(root, files["er_policy"]),
            "rights": _reference(root, files["er_rights"]),
            "model_version": "er-lightgbm-release-v1",
            **er_identity,
        },
        "ranker": {
            "model": _reference(root, files["ranker_model"]),
            "artifact_manifest": _reference(root, files["ranker_manifest"]),
            "ranker_version": "ltr-v4",
        },
        "ranker_promotion": {
            "evaluation_bundle_manifest": _reference(
                root, files["ranking_bundle_manifest"]
            ),
            "bundle_manifest_sha256": "e" * 64,
            "ledger_identity_sha256": "6" * 64,
            "evaluator_source_sha256": "7" * 64,
            "dependency_lock_sha256": "8" * 64,
            "policy": RankerPromotionPolicy().to_dict(),
        },
        "performance": [
            {
                "artifact_manifest": _reference(root, files["performance"]),
                "route": "gpu/local_ai",
                "model_version": "performance-v1",
            }
        ],
    }
    manifest = root / "serving-manifest.json"
    _write_manifest(manifest, payload)
    files["catalog"] = catalog
    files["offers"] = offers
    files["reviewed"] = reviewed
    files["review_evidence"] = review_evidence
    return manifest, catalog, offers, reviewed, files, er_identity


class _EncoderArguments(TypedDict):
    expected_encoder_bundle_path: Path
    expected_encoder_bundle_sha256: str


def _encoder_arguments(manifest: Path) -> _EncoderArguments:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    bundle = payload["embedding"]["encoder_bundle"]
    return {
        "expected_encoder_bundle_path": manifest.parent / bundle["path"],
        "expected_encoder_bundle_sha256": bundle["sha256"],
    }


def _source_arguments(manifest: Path) -> dict[str, object]:
    return {
        "current_source_registry_path": manifest.parent
        / "source-input"
        / "source-registry.yaml",
        "expected_source_trust_root_sha256": "5" * 64,
    }


def _patch_loaders(
    monkeypatch: pytest.MonkeyPatch,
    er_identity: dict[str, str],
) -> tuple[Any, Any, Any, Any]:
    ranker = object()
    performance_artifact = SimpleNamespace(
        config=SimpleNamespace(category="gpu", workload="local_ai"),
        model_version="performance-v1",
    )
    active_models = object()
    entity_resolution_release = SimpleNamespace(
        runtime=SimpleNamespace(model_version="er-lightgbm-release-v1"),
        policy=object(),
        identity=SimpleNamespace(**er_identity),
    )
    source_release = SimpleNamespace()

    def verify_source_release(**kwargs: object) -> object:
        source_release.manifest_path = Path(str(kwargs["manifest_path"])).resolve()
        source_release.manifest_sha256 = kwargs["expected_manifest_sha256"]
        source_release.records_path = Path(str(kwargs["records"])).resolve()
        source_release.authority_expires_at = datetime(2099, 1, 1, tzinfo=UTC)
        return source_release

    monkeypatch.setattr(
        serving_release,
        "verify_awin_production_batch_release",
        verify_source_release,
    )
    monkeypatch.setattr(
        serving_release,
        "validate_embedding_artifact",
        lambda *_args, **_kwargs: SimpleNamespace(
            vectors=np.zeros((1, 384), dtype=np.float32),
            products=(
                SimpleNamespace(
                    product_id="prod-1",
                    category=SimpleNamespace(value="gpu"),
                    brand="Fixture",
                ),
            ),
            search_documents=("fixture gpu",),
        ),
    )

    class _OfflineModel:
        @staticmethod
        def encode(_texts: list[str], **_kwargs: object) -> NDArray[np.float32]:
            values = np.zeros((1, 384), dtype=np.float32)
            values[0, 0] = 1.0
            return values

    def load_offline_model(
        encoder: SentenceTransformerEmbeddingEncoder,
    ) -> _OfflineModel:
        if not encoder.local_files_only:
            raise AssertionError("production encoder must be offline-only")
        if encoder.model_path is None or not encoder.model_path.is_dir():
            raise AssertionError("production encoder must load a local bundle")
        encoder.dimension = 384
        return _OfflineModel()

    monkeypatch.setattr(
        SentenceTransformerEmbeddingEncoder,
        "_load",
        load_offline_model,
    )

    class _RankerLoader:
        @staticmethod
        def load(_path: Path) -> object:
            return ranker

    monkeypatch.setattr(serving_release, "LambdaMARTRanker", _RankerLoader)
    monkeypatch.setattr(
        serving_release,
        "load_performance_artifact",
        lambda _path: performance_artifact,
    )
    monkeypatch.setattr(
        serving_release,
        "validate_promoted_serving_models",
        lambda **_kwargs: active_models,
    )
    monkeypatch.setattr(
        serving_release,
        "load_entity_resolution_release",
        lambda *_args, **_kwargs: entity_resolution_release,
    )
    return ranker, performance_artifact, active_models, entity_resolution_release


def test_qualifying_content_addressed_release_composes_all_runtime_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, catalog, offers, reviewed, _, er_identity = _release_fixture(tmp_path)
    ranker, performance, active_models, er_release = _patch_loaders(monkeypatch, er_identity)

    release = serving_release.load_production_serving_release(
        manifest,
        catalog_path=catalog,
        offers_path=offers,
        reviewed_mappings_path=reviewed,
        review_evidence_path=tmp_path / "review-evidence.jsonl",
        session_factory=sessionmaker(),
        expected_catalog_data_version="catalog-v1",
        expected_ranker_version="ltr-v4",
        expected_manifest_sha256=json.loads(manifest.read_text())["content_sha256"],
        **_source_arguments(manifest),
        **_encoder_arguments(manifest),
    )

    assert release.ranker is ranker
    assert release.performance_artifacts == (performance,)
    assert release.active_models is active_models
    assert release.retriever.retrieval_model_version == (
        "bm25-okapi+pgvector-cosine+rrf-v1@embedding-index-v1"
    )
    assert release.manifest_sha256 == json.loads(manifest.read_text())["content_sha256"]
    assert release.catalog_release.entity_resolution is er_release
    assert release.semantic_encoder_ready is True
    assert (
        release.encoder_bundle.path
        == Path(_encoder_arguments(manifest)["expected_encoder_bundle_path"]).resolve()
    )


def test_catalog_import_and_api_bootstrap_resolve_identical_er_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, catalog, offers, reviewed, _, er_identity = _release_fixture(tmp_path)
    _, _, _, er_release = _patch_loaders(monkeypatch, er_identity)
    manifest_sha256 = json.loads(manifest.read_text(encoding="utf-8"))["content_sha256"]

    import_release = serving_release.load_production_catalog_release(
        manifest,
        catalog_path=catalog,
        offers_path=offers,
        reviewed_mappings_path=reviewed,
        review_evidence_path=tmp_path / "review-evidence.jsonl",
        expected_catalog_data_version="catalog-v1",
        expected_manifest_sha256=manifest_sha256,
        **_source_arguments(manifest),
    )
    api_release = serving_release.load_production_serving_release(
        manifest,
        catalog_path=catalog,
        offers_path=offers,
        reviewed_mappings_path=reviewed,
        review_evidence_path=tmp_path / "review-evidence.jsonl",
        session_factory=sessionmaker(),
        expected_catalog_data_version="catalog-v1",
        expected_ranker_version="ltr-v4",
        expected_manifest_sha256=manifest_sha256,
        **_source_arguments(manifest),
        **_encoder_arguments(manifest),
    ).catalog_release

    assert import_release.manifest_sha256 == api_release.manifest_sha256 == manifest_sha256
    assert import_release.catalog_path == api_release.catalog_path == catalog.resolve()
    assert import_release.offers_path == api_release.offers_path == offers.resolve()
    assert (
        import_release.reviewed_mappings_path
        == api_release.reviewed_mappings_path
        == reviewed.resolve()
    )
    assert (
        import_release.review_evidence_path
        == api_release.review_evidence_path
        == (tmp_path / "review-evidence.jsonl").resolve()
    )
    assert import_release.entity_resolution is api_release.entity_resolution is er_release
    assert import_release.source_release is api_release.source_release
    assert import_release.source_release.records_path == offers.resolve()
    assert (
        import_release.source_release.manifest_sha256
        == json.loads(manifest.read_text(encoding="utf-8"))["source_release"]["manifest"][
            "sha256"
        ]
    )
    assert (
        import_release.entity_resolution.identity.binding_sha256
        == api_release.entity_resolution.identity.binding_sha256
        == er_identity["binding_sha256"]
    )


@pytest.mark.parametrize(
    "artifact_name",
    [
        "catalog",
        "offers",
        "reviewed",
        "review_evidence",
        "source_manifest",
        "source_raw",
        "source_rejections",
        "source_registry",
        "embedding",
        "retrieval",
        "ranker_model",
        "ranker_manifest",
        "ranking_bundle_manifest",
        "performance",
        "er_metadata",
        "er_model",
        "er_evidence",
        "er_evaluation",
        "er_policy",
        "er_rights",
        "encoder_bundle_file",
    ],
)
def test_release_fails_closed_when_any_pinned_artifact_is_tampered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_name: str,
) -> None:
    manifest, catalog, offers, reviewed, files, er_identity = _release_fixture(tmp_path)
    _patch_loaders(monkeypatch, er_identity)
    files[artifact_name].write_bytes(files[artifact_name].read_bytes() + b"tampered")

    with pytest.raises(
        ServingConfigurationError,
        match="size does not match|content hash does not match",
    ):
        serving_release.load_production_serving_release(
            manifest,
            catalog_path=catalog,
            offers_path=offers,
            reviewed_mappings_path=reviewed,
            review_evidence_path=tmp_path / "review-evidence.jsonl",
            session_factory=sessionmaker(),
            expected_catalog_data_version="catalog-v1",
            expected_ranker_version="ltr-v4",
            expected_manifest_sha256=json.loads(manifest.read_text())["content_sha256"],
            **_source_arguments(manifest),
            **_encoder_arguments(manifest),
        )


def test_release_fails_closed_on_manifest_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, catalog, offers, reviewed, _, er_identity = _release_fixture(tmp_path)
    _patch_loaders(monkeypatch, er_identity)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["ranker"]["ranker_version"] = "tampered"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ServingConfigurationError, match="content hash verification failed"):
        serving_release.load_production_serving_release(
            manifest,
            catalog_path=catalog,
            offers_path=offers,
            reviewed_mappings_path=reviewed,
            review_evidence_path=tmp_path / "review-evidence.jsonl",
            session_factory=sessionmaker(),
            expected_catalog_data_version="catalog-v1",
            expected_ranker_version="ltr-v4",
            expected_manifest_sha256=payload["content_sha256"],
            **_source_arguments(manifest),
            **_encoder_arguments(manifest),
        )


def test_source_release_verification_receives_exact_governed_offers_and_pins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, catalog, offers, reviewed, files, er_identity = _release_fixture(tmp_path)
    _patch_loaders(monkeypatch, er_identity)
    calls: list[dict[str, object]] = []
    verified = SimpleNamespace(
        manifest_path=files["source_manifest"].resolve(),
        manifest_sha256=sha256_file(files["source_manifest"]),
        authority_expires_at=datetime(2099, 1, 1, tzinfo=UTC),
    )

    def capture_verification(**kwargs: object) -> object:
        calls.append(kwargs)
        return verified

    monkeypatch.setattr(
        serving_release,
        "verify_awin_production_batch_release",
        capture_verification,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    release = serving_release.load_production_catalog_release(
        manifest,
        catalog_path=catalog,
        offers_path=offers,
        reviewed_mappings_path=reviewed,
        review_evidence_path=tmp_path / "review-evidence.jsonl",
        expected_catalog_data_version="catalog-v1",
        expected_manifest_sha256=payload["content_sha256"],
        **_source_arguments(manifest),
    )

    assert release.source_release is verified
    assert calls == [
        {
            "manifest_path": files["source_manifest"].resolve(),
            "expected_manifest_sha256": payload["source_release"]["manifest"]["sha256"],
            "expected_trust_root_sha256": "5" * 64,
            "current_source_registry": files["source_registry"].resolve(),
            "raw_snapshot": files["source_raw"].resolve(),
            "records": offers.resolve(),
            "rejections": files["source_rejections"].resolve(),
        }
    ]


def test_source_release_failure_blocks_catalogue_admission_after_outer_rehash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, catalog, offers, reviewed, _, er_identity = _release_fixture(tmp_path)
    _patch_loaders(monkeypatch, er_identity)

    def reject_release(**_kwargs: object) -> object:
        raise serving_release.AuthorizedBatchReleaseError(
            "accepted records do not match signed batch"
        )

    monkeypatch.setattr(
        serving_release,
        "verify_awin_production_batch_release",
        reject_release,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    _write_manifest(manifest, payload)

    with pytest.raises(
        ServingConfigurationError,
        match="authorized source release failed validation: accepted records",
    ):
        serving_release.load_production_catalog_release(
            manifest,
            catalog_path=catalog,
            offers_path=offers,
            reviewed_mappings_path=reviewed,
            review_evidence_path=tmp_path / "review-evidence.jsonl",
            expected_catalog_data_version="catalog-v1",
            expected_manifest_sha256=payload["content_sha256"],
            **_source_arguments(manifest),
        )


def test_source_trust_root_must_match_independent_operator_pin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, catalog, offers, reviewed, _, er_identity = _release_fixture(tmp_path)
    _patch_loaders(monkeypatch, er_identity)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    source_arguments = _source_arguments(manifest)
    source_arguments["expected_source_trust_root_sha256"] = "0" * 64

    with pytest.raises(ServingConfigurationError, match="independent operator pin"):
        serving_release.load_production_catalog_release(
            manifest,
            catalog_path=catalog,
            offers_path=offers,
            reviewed_mappings_path=reviewed,
            review_evidence_path=tmp_path / "review-evidence.jsonl",
            expected_catalog_data_version="catalog-v1",
            expected_manifest_sha256=payload["content_sha256"],
            **source_arguments,
        )


def test_source_release_path_escape_is_rejected_even_when_outer_manifest_is_rehashed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, catalog, offers, reviewed, _, er_identity = _release_fixture(tmp_path)
    _patch_loaders(monkeypatch, er_identity)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["source_release"]["raw_snapshot"]["path"] = "../outside.csv"
    _write_manifest(manifest, payload)

    with pytest.raises(ServingConfigurationError, match="relative and confined"):
        serving_release.load_production_catalog_release(
            manifest,
            catalog_path=catalog,
            offers_path=offers,
            reviewed_mappings_path=reviewed,
            review_evidence_path=tmp_path / "review-evidence.jsonl",
            expected_catalog_data_version="catalog-v1",
            expected_manifest_sha256=payload["content_sha256"],
            **_source_arguments(manifest),
        )


def test_missing_source_release_is_rejected_even_when_outer_manifest_is_rehashed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, catalog, offers, reviewed, _, er_identity = _release_fixture(tmp_path)
    _patch_loaders(monkeypatch, er_identity)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload.pop("source_release")
    _write_manifest(manifest, payload)

    with pytest.raises(ServingConfigurationError, match="manifest fields do not match"):
        serving_release.load_production_catalog_release(
            manifest,
            catalog_path=catalog,
            offers_path=offers,
            reviewed_mappings_path=reviewed,
            review_evidence_path=tmp_path / "review-evidence.jsonl",
            expected_catalog_data_version="catalog-v1",
            expected_manifest_sha256=payload["content_sha256"],
            **_source_arguments(manifest),
        )


def test_legacy_v1_manifest_is_rejected_even_when_rehashed_and_pinned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, catalog, offers, reviewed, _, er_identity = _release_fixture(tmp_path)
    _patch_loaders(monkeypatch, er_identity)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["schema_version"] = "pc-build-recommender.serving-release.v1"
    _write_manifest(manifest, payload)

    with pytest.raises(ServingConfigurationError, match="unsupported serving manifest schema"):
        serving_release.load_production_serving_release(
            manifest,
            catalog_path=catalog,
            offers_path=offers,
            reviewed_mappings_path=reviewed,
            review_evidence_path=tmp_path / "review-evidence.jsonl",
            session_factory=sessionmaker(),
            expected_catalog_data_version="catalog-v1",
            expected_ranker_version="ltr-v4",
            expected_manifest_sha256=payload["content_sha256"],
            **_source_arguments(manifest),
            **_encoder_arguments(manifest),
        )


def test_release_rejects_path_escape_even_when_manifest_is_rehashed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, catalog, offers, reviewed, _, er_identity = _release_fixture(tmp_path)
    _patch_loaders(monkeypatch, er_identity)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["retrieval"]["comparison_report"]["path"] = "../outside.json"
    _write_manifest(manifest, payload)

    with pytest.raises(ServingConfigurationError, match="relative and confined"):
        serving_release.load_production_serving_release(
            manifest,
            catalog_path=catalog,
            offers_path=offers,
            reviewed_mappings_path=reviewed,
            review_evidence_path=tmp_path / "review-evidence.jsonl",
            session_factory=sessionmaker(),
            expected_catalog_data_version="catalog-v1",
            expected_ranker_version="ltr-v4",
            expected_manifest_sha256=payload["content_sha256"],
            **_source_arguments(manifest),
            **_encoder_arguments(manifest),
        )


def test_release_rejects_a_valid_but_unpinned_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, catalog, offers, reviewed, _, er_identity = _release_fixture(tmp_path)
    _patch_loaders(monkeypatch, er_identity)

    with pytest.raises(ServingConfigurationError, match="operator-pinned"):
        serving_release.load_production_serving_release(
            manifest,
            catalog_path=catalog,
            offers_path=offers,
            reviewed_mappings_path=reviewed,
            review_evidence_path=tmp_path / "review-evidence.jsonl",
            session_factory=sessionmaker(),
            expected_catalog_data_version="catalog-v1",
            expected_ranker_version="ltr-v4",
            expected_manifest_sha256="0" * 64,
            **_source_arguments(manifest),
            **_encoder_arguments(manifest),
        )


def test_release_rejects_an_encoder_bundle_not_pinned_by_the_operator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, catalog, offers, reviewed, _, er_identity = _release_fixture(tmp_path)
    _patch_loaders(monkeypatch, er_identity)
    arguments = _encoder_arguments(manifest)
    arguments["expected_encoder_bundle_sha256"] = "0" * 64

    with pytest.raises(ServingConfigurationError, match="operator-pinned SHA-256"):
        serving_release.load_production_serving_release(
            manifest,
            catalog_path=catalog,
            offers_path=offers,
            reviewed_mappings_path=reviewed,
            review_evidence_path=tmp_path / "review-evidence.jsonl",
            session_factory=sessionmaker(),
            expected_catalog_data_version="catalog-v1",
            expected_ranker_version="ltr-v4",
            expected_manifest_sha256=json.loads(manifest.read_text())["content_sha256"],
            **_source_arguments(manifest),
            **arguments,
        )


def test_release_rejects_encoder_bundle_count_drift_in_rehashed_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, catalog, offers, reviewed, _, er_identity = _release_fixture(tmp_path)
    _patch_loaders(monkeypatch, er_identity)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["embedding"]["encoder_bundle"]["file_count"] += 1
    _write_manifest(manifest, payload)

    with pytest.raises(ServingConfigurationError, match="file count does not match"):
        serving_release.load_production_serving_release(
            manifest,
            catalog_path=catalog,
            offers_path=offers,
            reviewed_mappings_path=reviewed,
            review_evidence_path=tmp_path / "review-evidence.jsonl",
            session_factory=sessionmaker(),
            expected_catalog_data_version="catalog-v1",
            expected_ranker_version="ltr-v4",
            expected_manifest_sha256=payload["content_sha256"],
            **_source_arguments(manifest),
            **_encoder_arguments(manifest),
        )


def test_release_fails_closed_without_the_mounted_encoder_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, catalog, offers, reviewed, files, er_identity = _release_fixture(tmp_path)
    _patch_loaders(monkeypatch, er_identity)
    files["encoder_bundle_file"].parent.rename(tmp_path / "detached-encoder-bundle")

    with pytest.raises(ServingConfigurationError, match="bundle path is unavailable"):
        serving_release.load_production_serving_release(
            manifest,
            catalog_path=catalog,
            offers_path=offers,
            reviewed_mappings_path=reviewed,
            review_evidence_path=tmp_path / "review-evidence.jsonl",
            session_factory=sessionmaker(),
            expected_catalog_data_version="catalog-v1",
            expected_ranker_version="ltr-v4",
            expected_manifest_sha256=json.loads(manifest.read_text())["content_sha256"],
            **_source_arguments(manifest),
            **_encoder_arguments(manifest),
        )


def test_release_fails_closed_when_encoder_warmup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, catalog, offers, reviewed, _, er_identity = _release_fixture(tmp_path)
    _patch_loaders(monkeypatch, er_identity)

    def fail_warmup(
        _encoder: SentenceTransformerEmbeddingEncoder,
        *,
        expected_dimension: int,
    ) -> int:
        raise RuntimeError(f"warmup probe failed for dimension {expected_dimension}")

    monkeypatch.setattr(SentenceTransformerEmbeddingEncoder, "warmup", fail_warmup)

    with pytest.raises(ServingConfigurationError, match="warmup probe failed"):
        serving_release.load_production_serving_release(
            manifest,
            catalog_path=catalog,
            offers_path=offers,
            reviewed_mappings_path=reviewed,
            review_evidence_path=tmp_path / "review-evidence.jsonl",
            session_factory=sessionmaker(),
            expected_catalog_data_version="catalog-v1",
            expected_ranker_version="ltr-v4",
            expected_manifest_sha256=json.loads(manifest.read_text())["content_sha256"],
            **_source_arguments(manifest),
            **_encoder_arguments(manifest),
        )
