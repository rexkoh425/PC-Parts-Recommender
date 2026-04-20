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

# TODO: rest of this module still to come.
