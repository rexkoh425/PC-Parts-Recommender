from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts import import_catalog_release
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from pc_build_recommender.application import ServingConfigurationError
from pc_build_recommender.evaluation.manifest import sha256_file


def test_release_sessions_cannot_commit_past_the_outer_release_transaction() -> None:
    engine = create_engine("sqlite:///:memory:")
    try:
        with engine.begin() as connection:
            connection.execute(text("CREATE TABLE release_probe (value INTEGER NOT NULL)"))

        with (
            pytest.raises(RuntimeError, match="final identity failed"),
            engine.begin() as connection,
        ):
            factory = import_catalog_release._release_session_factory(connection)
            with import_catalog_release.session_scope(factory) as session:
                session.execute(text("INSERT INTO release_probe (value) VALUES (1)"))
            # The vector repository owns a normal Session/transaction over the same bound
            # connection. Its inner commit must also remain subordinate to the release.
            with Session(connection) as vector_session, vector_session.begin():
                vector_session.execute(text("INSERT INTO release_probe (value) VALUES (2)"))
            # The durable identity verifier opens a read-only session and closes it without an
            # explicit commit. That close must leave the outer transaction authoritative too.
            with factory() as identity_session:
                assert identity_session.scalar(text("SELECT COUNT(*) FROM release_probe")) == 2
            assert connection.scalar(text("SELECT COUNT(*) FROM release_probe")) == 2
            raise RuntimeError("final identity failed")

        with engine.connect() as connection:
            assert connection.scalar(text("SELECT COUNT(*) FROM release_probe")) == 0
    finally:
        engine.dispose()


def _embedding_manifest(
    root: Path,
    *,
    artifact_manifest: Path,
    artifact: SimpleNamespace,
) -> dict[str, object]:
    return {
        "artifact_manifest": {
            "path": artifact_manifest.relative_to(root).as_posix(),
            "size_bytes": artifact_manifest.stat().st_size,
            "sha256": sha256_file(artifact_manifest),
        },
        "data_version": artifact.data_version,
        "index_version": artifact.index_version,
        "embedding_model": artifact.embedding_model,
        "encoder_revision": artifact.manifest["encoder"]["model_revision"],
        "encoder_fingerprint": artifact.encoder_fingerprint,
        "dataset_content_hash": artifact.dataset_content_hash,
        "manifest_schema_version": artifact.manifest["schema_version"],
        "device": "cpu",
        "batch_size": 32,
        "rrf_k": 60,
        "encoder_bundle": {
            "path": "encoders/pinned",
            "sha256": "f" * 64,
            "file_count": 1,
            "size_bytes": 1,
        },
    }


def test_pinned_embedding_artifact_is_derived_from_verified_serving_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "release"
    artifact_manifest = root / "embedding" / "manifest.json"
    artifact_manifest.parent.mkdir(parents=True)
    artifact_manifest.write_text("{}\n", encoding="utf-8")
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text("{}\n", encoding="utf-8")
    artifact = SimpleNamespace(
        data_version="embedding-data-v1",
        index_version="embedding-index-v1",
        embedding_model="fixture-encoder",
        encoder_fingerprint="a" * 64,
        dataset_content_hash="b" * 64,
        manifest={
            "schema_version": "product-embedding-index-manifest-v1",
            "encoder": {"model_revision": "revision-1"},
        },
    )
    release = SimpleNamespace(
        manifest_path=root / "serving-manifest.json",
        catalog_path=catalog,
        manifest={
            "embedding": _embedding_manifest(
                root,
                artifact_manifest=artifact_manifest,
                artifact=artifact,
            )
        },
    )
    calls: list[tuple[Path, Path]] = []

    def validate(catalog_path: Path, artifact_dir: Path) -> SimpleNamespace:
        calls.append((catalog_path, artifact_dir))
        return artifact

    monkeypatch.setattr(import_catalog_release, "validate_embedding_artifact", validate)

    assert import_catalog_release._pinned_embedding_artifact(release) is artifact
    assert calls == [(catalog, artifact_manifest.parent)]


def test_pinned_embedding_artifact_rejects_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "release"
    artifact_manifest = root / "embedding" / "manifest.json"
    artifact_manifest.parent.mkdir(parents=True)
    artifact_manifest.write_text("{}\n", encoding="utf-8")
    artifact = SimpleNamespace(
        data_version="artifact-data-v1",
        index_version="embedding-index-v1",
        embedding_model="fixture-encoder",
        encoder_fingerprint="a" * 64,
        dataset_content_hash="b" * 64,
        manifest={
            "schema_version": "product-embedding-index-manifest-v1",
            "encoder": {"model_revision": "revision-1"},
        },
    )
    configured = _embedding_manifest(
        root,
        artifact_manifest=artifact_manifest,
        artifact=artifact,
    )
    configured["data_version"] = "different-data-v1"
    release = SimpleNamespace(
        manifest_path=root / "serving-manifest.json",
        catalog_path=tmp_path / "catalog.jsonl",
        manifest={"embedding": configured},
    )
    monkeypatch.setattr(
        import_catalog_release,
        "validate_embedding_artifact",
        lambda *_args: artifact,
    )

    with pytest.raises(
        ServingConfigurationError,
        match="identity does not match.*data_version",
    ):
        import_catalog_release._pinned_embedding_artifact(release)


def test_release_sequence_is_non_destructive_and_finishes_with_exact_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    product = SimpleNamespace(product_id="product-1")
    listing = SimpleNamespace(listing_id="listing-1")
    data = SimpleNamespace(
        products=(product,),
        listings=(listing,),
        stats=SimpleNamespace(data_version="catalog-v1"),
    )
    artifact = SimpleNamespace(
        products=(product,),
        data_version="embedding-data-v1",
        index_version="embedding-index-v1",
        embedding_model="fixture-encoder",
        dataset_content_hash="b" * 64,
        product_count=1,
    )
    release = SimpleNamespace(
        manifest_sha256="c" * 64,
        source_release=SimpleNamespace(
            manifest_sha256="d" * 64,
            source_name="fixture-source",
            raw_snapshot_sha256="e" * 64,
            processed_run_sha256="f" * 64,
            authority_expires_at=SimpleNamespace(isoformat=lambda: "2099-01-01T00:00:00+00:00"),
            accepted_count=1,
            rejected_count=0,
        ),
    )

    monkeypatch.setattr(
        import_catalog_release,
        "load_production_catalog_release",
        lambda *_args, **_kwargs: events.append("load-release") or release,
    )
    monkeypatch.setattr(
        import_catalog_release,
        "_pinned_embedding_artifact",
        lambda _release: events.append("validate-embedding") or artifact,
    )
    monkeypatch.setattr(
        import_catalog_release,
        "_load_release_data",
        lambda _release, **_kwargs: events.append("load-data") or data,
    )
    monkeypatch.setattr(
        import_catalog_release,
        "_verify_readiness_artifact",
        lambda *_args, **_kwargs: events.append("verify-readiness"),
    )

    connection = object()

    class FakeEngine:
        @contextmanager
        def begin(self) -> Iterator[object]:
            events.append("release-transaction-begin")
            try:
                yield connection
            except Exception:
                events.append("release-transaction-rollback")
                raise
            else:
                events.append("release-transaction-commit")

        def dispose(self) -> None:
            events.append("dispose")

    engine = FakeEngine()
    monkeypatch.setattr(
        import_catalog_release,
        "create_db_engine",
        lambda _url: events.append("engine") or engine,
    )

    class FakeStore:
        def __init__(
            self,
            actual_engine: object,
            session_factory: object | None = None,
        ) -> None:
            assert actual_engine is engine
            self.session_factory = session_factory or object()

        def verify_schema(self) -> None:
            events.append("schema")

        def verify_no_unexpected_catalog_ids(self, **_kwargs: object) -> None:
            events.append("stale-preflight")

        def verify_catalog_identity(self, **kwargs: object) -> None:
            assert kwargs["canonical_products"] == data.products
            assert kwargs["retailer_listings"] == data.listings
            events.append("exact-identity")

    monkeypatch.setattr(import_catalog_release, "SqlAlchemyDurableStore", FakeStore)
    release_factory = object()

    def fake_release_session_factory(actual_connection: object) -> object:
        assert actual_connection is connection
        events.append("bind-release-transaction")
        return release_factory

    monkeypatch.setattr(
        import_catalog_release,
        "_release_session_factory",
        fake_release_session_factory,
    )

    @contextmanager
    def fake_session_scope(actual_factory: object) -> Iterator[object]:
        assert actual_factory is release_factory
        yield object()

    monkeypatch.setattr(import_catalog_release, "session_scope", fake_session_scope)
    monkeypatch.setattr(
        import_catalog_release,
        "seed_processed_catalog",
        lambda *_args: events.append("catalog-upsert"),
    )

    class FakeVectorResult:
        def to_dict(self) -> dict[str, int]:
            return {"product_count": 1}

    class FakeVectorRepository:
        def __init__(self, actual_connection: object) -> None:
            assert actual_connection is connection

        def import_artifact(
            self,
            actual_artifact: object,
            *,
            batch_size: int,
            reconcile_stale_provenance: bool,
        ) -> FakeVectorResult:
            assert actual_artifact is artifact
            assert batch_size == 100
            assert reconcile_stale_provenance is False
            events.append("vector-import")
            return FakeVectorResult()

    monkeypatch.setattr(
        import_catalog_release,
        "PostgresVectorCatalogRepository",
        FakeVectorRepository,
    )

    report = import_catalog_release.run_catalog_release(
        buildcores=tmp_path / "catalog.jsonl",
        offers=tmp_path / "offers.jsonl",
        reviewed_mappings=tmp_path / "mappings.json",
        review_evidence=tmp_path / "evidence.jsonl",
        serving_manifest=tmp_path / "serving-manifest.json",
        serving_manifest_sha256="c" * 64,
        source_registry=tmp_path / "source-registry.yaml",
        source_trust_root_sha256="d" * 64,
        readiness_artifact=tmp_path / "readiness.json",
        expected_data_version="catalog-v1",
        database_url="postgresql+psycopg://release.invalid/pcbr",
        batch_size=100,
        max_line_bytes=1024,
    )

    assert events == [
        "load-release",
        "validate-embedding",
        "load-data",
        "verify-readiness",
        "engine",
        "schema",
        "stale-preflight",
        "release-transaction-begin",
        "bind-release-transaction",
        "catalog-upsert",
        "vector-import",
        "exact-identity",
        "release-transaction-commit",
        "dispose",
    ]
    assert report["identity_verification"] == "exact"
    assert report["atomic_release_transaction"] is True
    assert report["destructive_reconciliation_performed"] is False


def test_stale_preflight_blocks_all_catalogue_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    product = SimpleNamespace(product_id="product-1")
    data = SimpleNamespace(
        products=(product,),
        listings=(),
        stats=SimpleNamespace(data_version="catalog-v1"),
    )
    artifact = SimpleNamespace(products=(product,))
    release = SimpleNamespace()
    monkeypatch.setattr(
        import_catalog_release,
        "load_production_catalog_release",
        lambda *_args, **_kwargs: release,
    )
    monkeypatch.setattr(
        import_catalog_release,
        "_pinned_embedding_artifact",
        lambda _release: artifact,
    )
    monkeypatch.setattr(
        import_catalog_release,
        "_load_release_data",
        lambda _release, **_kwargs: data,
    )
    monkeypatch.setattr(
        import_catalog_release,
        "_verify_readiness_artifact",
        lambda *_args, **_kwargs: None,
    )

    class FakeEngine:
        def dispose(self) -> None:
            events.append("dispose")

    monkeypatch.setattr(import_catalog_release, "create_db_engine", lambda _url: FakeEngine())

    class FakeStore:
        session_factory = object()

        def __init__(self, _engine: object) -> None:
            return None

        def verify_schema(self) -> None:
            return None

        def verify_no_unexpected_catalog_ids(self, **_kwargs: object) -> None:
            raise RuntimeError("stale rows require reconciliation")

    monkeypatch.setattr(import_catalog_release, "SqlAlchemyDurableStore", FakeStore)
    monkeypatch.setattr(
        import_catalog_release,
        "seed_processed_catalog",
        lambda *_args: events.append("catalog-upsert"),
    )
    monkeypatch.setattr(
        import_catalog_release,
        "PostgresVectorCatalogRepository",
        lambda _engine: events.append("vector-import"),
    )

    with pytest.raises(RuntimeError, match="stale rows"):
        import_catalog_release.run_catalog_release(
            buildcores=tmp_path / "catalog.jsonl",
            offers=tmp_path / "offers.jsonl",
            reviewed_mappings=tmp_path / "mappings.json",
            review_evidence=tmp_path / "evidence.jsonl",
            serving_manifest=tmp_path / "serving-manifest.json",
            serving_manifest_sha256="c" * 64,
            source_registry=tmp_path / "source-registry.yaml",
            source_trust_root_sha256="d" * 64,
            readiness_artifact=tmp_path / "readiness.json",
            expected_data_version="catalog-v1",
            database_url="postgresql+psycopg://release.invalid/pcbr",
            batch_size=100,
            max_line_bytes=1024,
        )

    assert events == ["dispose"]
