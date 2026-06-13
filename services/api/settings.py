"""Runtime configuration for the HTTP service."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApiRuntimeSettings(BaseSettings):
    """Settings are environment-driven and intentionally contain no secrets."""

    model_config = SettingsConfigDict(
        env_prefix="PCBR_API_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    app_name: str = "PC Build Recommender API"
    environment: str = "development"
    service_mode: Literal["demo", "processed_catalog"] = "demo"
    log_level: str = "INFO"
    docs_enabled: bool = True
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"]
    )

    data_version: str = "demo-seed-2026-07-22"
    ranking_model_version: str = "deterministic-baseline-v1"
    compatibility_rule_version: str = "compat_v2"
    solver_version: str = "in-memory-baseline-v1"
    stale_after_hours: int = Field(default=24, ge=1, le=24 * 30)

    request_id_header: str = "X-Request-ID"
    max_request_body_bytes: int = Field(
        default=1024 * 1024,
        ge=1024,
        le=16 * 1024 * 1024,
    )
    build_generation_max_concurrency: int = Field(default=1, ge=1, le=16)
    build_generation_max_queue_size: int = Field(default=8, ge=0, le=256)
    build_generation_queue_timeout_seconds: float = Field(default=2.0, gt=0, le=60)
    build_share_ttl_hours: int = Field(default=24 * 30, ge=1, le=24 * 365)
    admin_token: SecretStr | None = Field(default=None)
    admin_token_file: Path | None = None
    pipeline_operations_path: Path | None = None
    pipeline_operations_window_hours: int = Field(default=24 * 7, ge=1, le=24 * 31)

    storage_backend: Literal["auto", "memory", "database"] = "auto"
    database_url: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("PCBR_API_DATABASE_URL", "DATABASE_URL"),
    )

    buildcores_catalog_path: Path | None = None
    governed_offers_path: Path | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "PCBR_API_GOVERNED_OFFERS_PATH",
            "governed_offers_path",
            "PCBR_API_DYNACORE_OFFERS_PATH",
            "dynacore_offers_path",
        ),
    )
    reviewed_mapping_path: Path | None = None
    review_evidence_path: Path | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "PCBR_API_REVIEW_EVIDENCE_PATH",
            "review_evidence_path",
        ),
    )
    entity_resolution_evaluation_path: Path | None = None
    serving_manifest_path: Path | None = None
    serving_manifest_sha256: str | None = None
    semantic_encoder_bundle_path: Path | None = None
    semantic_encoder_bundle_sha256: str | None = None
    allow_development_catalog: bool = False
    performance_artifact_paths: list[Path] = Field(default_factory=list)
    allow_unpromoted_performance_models: bool = False

    @model_validator(mode="after")
    def processed_catalog_paths_are_explicit(self) -> ApiRuntimeSettings:
        if self.admin_token is not None and self.admin_token_file is not None:
            raise ValueError("configure only one of admin_token or admin_token_file")
        if self.admin_token_file is not None:
            try:
                token = self.admin_token_file.read_text(encoding="utf-8").strip()
            except OSError as error:
                raise ValueError("admin_token_file could not be read") from error
            if len(token) < 24:
                raise ValueError(
                    "admin_token_file must contain at least 24 non-whitespace characters"
                )
            self.admin_token = SecretStr(token)
        elif self.admin_token is not None and len(self.admin_token.get_secret_value().strip()) < 24:
            raise ValueError("admin_token must contain at least 24 non-whitespace characters")
        if self.service_mode == "processed_catalog" and (
            self.buildcores_catalog_path is None or self.governed_offers_path is None
        ):
            raise ValueError(
                "processed_catalog mode requires buildcores_catalog_path and governed_offers_path"
            )
        development_environments = {
            "development",
            "dev",
            "local",
            "test",
            "testing",
        }
        if self.allow_development_catalog and self.service_mode != "processed_catalog":
            raise ValueError("allow_development_catalog applies only to processed_catalog mode")
        if self.serving_manifest_path is not None and self.service_mode != "processed_catalog":
            raise ValueError("serving_manifest_path applies only to processed_catalog mode")
        if self.review_evidence_path is not None and self.service_mode != "processed_catalog":
            raise ValueError("review_evidence_path applies only to processed_catalog mode")
        if (self.serving_manifest_path is None) != (self.serving_manifest_sha256 is None):
            raise ValueError(
                "serving_manifest_path and serving_manifest_sha256 must be configured together"
            )
        if (self.semantic_encoder_bundle_path is None) != (
            self.semantic_encoder_bundle_sha256 is None
        ):
            raise ValueError(
                "semantic_encoder_bundle_path and semantic_encoder_bundle_sha256 "
                "must be configured together"
            )
        if (
            self.semantic_encoder_bundle_path is not None
            and self.service_mode != "processed_catalog"
        ):
            raise ValueError("semantic_encoder_bundle_path applies only to processed_catalog mode")
        if self.serving_manifest_sha256 is not None and (
            len(self.serving_manifest_sha256) != 64
            or any(
                character not in "0123456789abcdef" for character in self.serving_manifest_sha256
            )
        ):
            raise ValueError("serving_manifest_sha256 must be a lowercase SHA-256 digest")
        if self.semantic_encoder_bundle_sha256 is not None and (
            len(self.semantic_encoder_bundle_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.semantic_encoder_bundle_sha256
            )
        ):
            raise ValueError("semantic_encoder_bundle_sha256 must be a lowercase SHA-256 digest")
        if self.performance_artifact_paths and self.service_mode != "processed_catalog":
            raise ValueError("performance artifacts apply only to processed_catalog mode")
        if self.allow_unpromoted_performance_models and not self.performance_artifact_paths:
            raise ValueError(
                "allow_unpromoted_performance_models requires performance_artifact_paths"
            )
        if (
            self.allow_unpromoted_performance_models
            and self.environment.casefold() not in development_environments
        ):
            raise ValueError("unpromoted performance models are development/test-only")
        if (
            self.allow_development_catalog
            and self.environment.casefold() not in development_environments
        ):
            raise ValueError("allow_development_catalog is forbidden outside development/test")
        if self.storage_backend == "database" and self.database_url is None:
            raise ValueError("database storage requires DATABASE_URL or PCBR_API_DATABASE_URL")
        if self.requires_durable_storage:
            if self.admin_token is not None and self.admin_token_file is None:
                raise ValueError(
                    "non-development administrator tokens must be supplied through admin_token_file"
                )
            if self.entity_resolution_evaluation_path is not None:
                raise ValueError(
                    "non-development entity-resolution evaluation must come from the immutable "
                    "serving manifest"
                )
            if self.performance_artifact_paths:
                raise ValueError(
                    "non-development serving loads performance models only from the immutable "
                    "serving manifest"
                )
            if self.storage_backend == "memory":
                raise ValueError(
                    "non-development processed_catalog mode cannot use in-memory storage"
                )
            if self.database_url is None:
                raise ValueError(
                    "non-development processed_catalog mode requires durable DATABASE_URL storage"
                )
            if self.serving_manifest_path is None:
                raise ValueError(
                    "non-development processed_catalog mode requires serving_manifest_path "
                    "and serving_manifest_sha256"
                )
            if self.semantic_encoder_bundle_path is None:
                raise ValueError(
                    "non-development processed_catalog mode requires "
                    "semantic_encoder_bundle_path and semantic_encoder_bundle_sha256"
                )
            if self.review_evidence_path is None:
                raise ValueError(
                    "non-development processed_catalog mode requires a pinned review_evidence_path"
                )
            database_url = self.database_url.get_secret_value().casefold()
            if not database_url.startswith(("postgresql://", "postgresql+")):
                raise ValueError("production durable storage must use PostgreSQL")
        self.validate_http_exposure()
        return self

    def validate_http_exposure(self) -> None:
        """Fail closed on developer-only HTTP surfaces outside development/test."""

        if self.is_development_environment:
            return
        if self.docs_enabled:
            raise ValueError("API documentation must be disabled outside development/test")
        if any(origin.strip() == "*" for origin in self.cors_origins):
            raise ValueError("wildcard CORS origins are forbidden outside development/test")

    @property
    def is_development_environment(self) -> bool:
        return self.environment.casefold() in {
            "development",
            "dev",
            "local",
            "test",
            "testing",
        }

    @property
    def requires_durable_storage(self) -> bool:
        return self.service_mode == "processed_catalog" and not self.is_development_environment

    @property
    def uses_database_storage(self) -> bool:
        if self.storage_backend == "database":
            return True
        if self.storage_backend == "memory":
            return False
        return self.requires_durable_storage

    @property
    def dynacore_offers_path(self) -> Path | None:
        """Deprecated read-only alias for callers migrating to governed_offers_path."""

        return self.governed_offers_path
