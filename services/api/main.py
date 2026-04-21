"""FastAPI application factory and default ASGI entry point."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from services.api.core_service import create_processed_catalog_service
from services.api.errors import install_exception_handlers
from services.api.middleware import (
    BuildGenerationAdmissionController,
    BuildGenerationAdmissionMiddleware,
    RequestBodyLimitMiddleware,
    RequestContextMiddleware,
    configure_logging,
)
from services.api.routers import admin, builds, compatibility, health, interactions, products
from services.api.service import InMemoryRecommendationService, RecommendationApplication
from services.api.settings import ApiRuntimeSettings


def create_app(
    settings: ApiRuntimeSettings | None = None,
    service: RecommendationApplication | None = None,
) -> FastAPI:
    runtime_settings = settings or ApiRuntimeSettings()
    runtime_settings.validate_http_exposure()
    if service is not None:
        application_service = service
    elif runtime_settings.service_mode == "processed_catalog":
        application_service = create_processed_catalog_service(runtime_settings)
        runtime_settings = application_service.settings
    else:
        if runtime_settings.environment.casefold() not in {
            "development",
            "dev",
            "local",
            "test",
            "testing",
        }:
            raise RuntimeError(
                "The controlled demo service is development-only; configure "
                "PCBR_API_SERVICE_MODE=processed_catalog for a non-development runtime."
            )
        application_service = InMemoryRecommendationService(runtime_settings)
    configure_logging(runtime_settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            await health.refresh_freshness_metrics(application_service)
            yield
        finally:
            await application_service.close()

    app = FastAPI(
        title=runtime_settings.app_name,
        version="0.1.0",
        docs_url="/docs" if runtime_settings.docs_enabled else None,
        redoc_url="/redoc" if runtime_settings.docs_enabled else None,
        openapi_url="/openapi.json" if runtime_settings.docs_enabled else None,
        lifespan=lifespan,
    )
    app.state.settings = runtime_settings
    app.state.application_service = application_service
    build_generation_admission = BuildGenerationAdmissionController(
        max_concurrency=runtime_settings.build_generation_max_concurrency,
        max_queue_size=runtime_settings.build_generation_max_queue_size,
        queue_timeout_seconds=runtime_settings.build_generation_queue_timeout_seconds,
    )
    app.state.build_generation_admission = build_generation_admission

    # Middleware executes in reverse registration order. Request context remains outermost,
    # CORS wraps middleware-generated errors, and resource controls run before request parsing.
    app.add_middleware(
        BuildGenerationAdmissionMiddleware,
        controller=build_generation_admission,
    )
    app.add_middleware(
        RequestBodyLimitMiddleware,
        max_body_bytes=runtime_settings.max_request_body_bytes,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=runtime_settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=[
            "Content-Type",
            "Accept",
            runtime_settings.request_id_header,
            "X-PCBR-Admin-Token",
        ],
        expose_headers=[
            runtime_settings.request_id_header,
            "X-Data-Version",
            "X-Ranking-Model",
            "X-Compatibility-Rule-Version",
            "X-Solver-Version",
            "Retry-After",
        ],
    )
    app.add_middleware(RequestContextMiddleware, settings=runtime_settings)
    install_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(builds.router)
    app.include_router(products.router)
    app.include_router(compatibility.router)
    app.include_router(interactions.router)
    app.include_router(admin.router)
    return app


app = create_app()
