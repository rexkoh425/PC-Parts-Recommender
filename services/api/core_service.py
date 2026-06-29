"""FastAPI adapter for the real catalog-backed recommendation application."""

from __future__ import annotations

import asyncio
from collections.abc import Collection, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, Literal
from uuid import uuid4

from pc_build_recommender.application import (
    ActiveServingModels,
    ApplicationBuildGenerationResponse,
    ApplicationError,
    ApplicationServices,
    CatalogIntegrityError,
    EmptyCatalogError,
    ReplacementMode,
    RequestConflictError,
    ResultNotFoundError,
    SearchProductResult,
    ServingConfigurationError,
    create_application_services,
)
from pc_build_recommender.catalog import (
    CatalogReadinessReport,
    InMemoryCatalogReader,
    ProcessedCatalogData,
    load_processed_catalog,
    validate_production_readiness,
)
from pc_build_recommender.compatibility import AUTHORITATIVE_COMPATIBILITY_POLICY
from pc_build_recommender.domain import (
    BuildGenerationRequest as DomainBuildRequest,
)
from pc_build_recommender.domain import (
    BuildPreferences as DomainPreferences,
)
from pc_build_recommender.domain import (
    BuildProfile as DomainProfile,
)
from pc_build_recommender.domain import (
    BuildRecommendation as DomainBuild,
)
from pc_build_recommender.domain import (
    BuildRequirements as DomainRequirements,
)
from pc_build_recommender.domain import (
    ComponentCategory as DomainCategory,
)
from pc_build_recommender.domain import (
    ExistingComponent,
    InteractionType,
    SearchQuery,
    WorkloadPreference,
)
from pc_build_recommender.domain import (
    InteractionEvent as DomainInteractionEvent,
)
from pc_build_recommender.domain import (
    WorkloadName as DomainWorkload,
)
from pc_build_recommender.performance_models import (
    PerformanceModelArtifact,
    load_performance_artifact,
)
from pc_build_recommender.pipeline_operations import summarize_pipeline_operations
from pc_build_recommender.pricing import PriceObservation as HistoricalPriceObservation
from pc_build_recommender.ranking import ProductRanker
from pc_build_recommender.retrieval import ProductRetriever, StructuredFilters
from services.api.durability import DurableStorageError, SqlAlchemyDurableStore
from services.api.errors import ApiError
from services.api.models import (
    AdminMappingQueue,
    AdminMissingField,
    AdminOperationsResponse,
    AdminPipelineOperations,
    AdminPriceFreshness,
    BenchmarkObservation,
    BuildComponent,
    BuildGenerationStatus,
    BuildProfile,
    BuildShareCreated,
    BuildShareRevoked,
    BuildSummary,
    CanonicalInteractionEvent,
    CatalogueCoverage,
    CatalogueReadinessSummary,
    CompatibilityCheck,
    CompatibilityCheckRequest,
    CompatibilityCheckResponse,
    CompatibilityStatus,
    ComponentCategory,
    ExplanationItem,
    FreshnessResponse,
    GenerateBuildsRequest,
    GenerateBuildsResponse,
    InfeasibilityExplanation,
    InfeasibilityReason,
    InteractionAccepted,
    InvalidProductSearchCursor,
    PerformanceBenchmarkEvidence,
    PerformanceSignal,
    PriceObservation,
    ProductBenchmarksResponse,
    ProductDetail,
    ProductFacetCount,
    ProductPricesResponse,
    ProductReviewsResponse,
    ProductSearchFacets,
    ProductSearchItem,
    ProductSearchPagination,
    ProductSearchRequest,
    ProductSearchResponse,
    PublicBuildShare,
    PublicBuildSnapshot,
    ReplacementCandidate,
    ReplacementRequest,
    ReplacementResponse,
    RevokeBuildShareRequest,
    SolverProfileOutcome,
    SolverStatus,
    SourceAttribution,
    SourceReference,
    SuggestedRelaxation,
    encode_product_search_cursor,
    product_search_identity,
    resolve_product_search_page,
)
from services.api.models import (
    ReviewEvidence as ApiReviewEvidence,
)
from services.api.pricing import summarize_price_history
from services.api.public_shares import public_build_snapshot
from services.api.service import (
    RecommendationApplication,
    interaction_event_id,
    same_canonical_interaction,
)
from services.api.serving_release import (
    load_production_serving_release,
    production_catalog_policy_from_entity_resolution,
)
from services.api.settings import ApiSettings

_MAX_PRICE_OBSERVATIONS_RETURNED = 365
_REVIEW_POSITIVE_SENTIMENT_THRESHOLD = 0.25
_REVIEW_NEGATIVE_SENTIMENT_THRESHOLD = -0.25

_API_TO_DOMAIN_CATEGORY = {
    ComponentCategory.CPU: DomainCategory.CPU,
    ComponentCategory.GPU: DomainCategory.GPU,
    ComponentCategory.MOTHERBOARD: DomainCategory.MOTHERBOARD,
    ComponentCategory.MEMORY: DomainCategory.MEMORY,
    ComponentCategory.STORAGE: DomainCategory.STORAGE,
    ComponentCategory.PSU: DomainCategory.POWER_SUPPLY,
    ComponentCategory.COOLER: DomainCategory.COOLER,
    ComponentCategory.CASE: DomainCategory.CASE,
}
_DOMAIN_TO_API_CATEGORY = {value: key for key, value in _API_TO_DOMAIN_CATEGORY.items()}
_PROFILE_ORDER = (
    DomainProfile.BEST_OVERALL,
    DomainProfile.BEST_VALUE,
    DomainProfile.HIGHEST_PERFORMANCE,
    DomainProfile.MOST_UPGRADEABLE,
    DomainProfile.LOWEST_POWER,
)


def _processed_catalog_freshness(
    processed_data: ProcessedCatalogData,
    *,
    catalogue_stale_after_hours: int,
    price_stale_after_hours: int,
    now: datetime,
) -> tuple[
    datetime | None,
    datetime | None,
    Literal["fresh", "stale", "degraded"],
    Literal["fresh", "stale", "degraded"],
    Literal["fresh", "stale", "degraded"],
    tuple[str, ...],
]:
    """Measure catalogue and price recency without inferring missing evidence as fresh."""

    def measure_dimension(
        values: Sequence[object],
        *,
        dimension: Literal["Catalogue", "Price"],
        timestamp_label: str,
        missing_message: str,
        stale_after_hours: int,
    ) -> tuple[datetime | None, Literal["fresh", "stale", "degraded"], tuple[str, ...]]:
        valid: list[datetime] = []
        invalid_count = 0
        future_count = 0
        for value in values:
            if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
                invalid_count += 1
                continue
            normalised = value.astimezone(UTC)
            if normalised > now:
                future_count += 1
            valid.append(normalised)

        watermark = min(valid, default=None)
        blockers: list[str] = []
        if not values:
            blockers.append(missing_message)
        if invalid_count:
            blockers.append(
                f"{dimension} freshness cannot be verified because {invalid_count} "
                "timestamp value(s) are missing or not timezone-aware."
            )
        if future_count:
            blockers.append(
                f"{dimension} freshness cannot be verified because {future_count} "
                "timestamp value(s) are in the future."
            )
        if blockers:
            return watermark, "degraded", tuple(blockers)

        # Every catalogue product and every listing's latest price observation must be
        # current. A single fresh row must not hide stale required serving evidence.
        assert watermark is not None
        if (now - watermark).total_seconds() / 3600 > stale_after_hours:
            return (
                watermark,
                "stale",
                (
                    f"{dimension} data is stale: {timestamp_label} exceeds "
                    f"stale_after_hours={stale_after_hours}.",
                ),
            )
        return watermark, "fresh", ()

    catalog_values = tuple(
        getattr(item, "updated_at", None) for item in processed_data.products
    )
    price_snapshots = tuple(getattr(processed_data, "price_snapshots", ()))
    price_values_by_listing: dict[str, list[object]] = {}
    for index, snapshot in enumerate(price_snapshots):
        listing_id = str(getattr(snapshot, "listing_id", f"__snapshot_{index}"))
        price_values_by_listing.setdefault(listing_id, []).append(
            getattr(snapshot, "observed_at", None)
        )
    expected_listing_ids = {
        str(getattr(listing, "listing_id", ""))
        for listing in getattr(processed_data, "listings", ())
        if str(getattr(listing, "listing_id", ""))
    }
    price_values: list[object] = []
    for listing_id in sorted(set(price_values_by_listing) | expected_listing_ids):
        observations = price_values_by_listing.get(listing_id, [])
        valid_observations = [
            value
            for value in observations
            if isinstance(value, datetime)
            and value.tzinfo is not None
            and value.utcoffset() is not None
        ]
        if len(valid_observations) != len(observations) or not observations:
            price_values.append(None)
        if valid_observations:
            price_values.append(max(valid_observations))

    catalog_watermark, catalogue_status, catalogue_blockers = measure_dimension(
        catalog_values,
        dimension="Catalogue",
        timestamp_label="last_catalog_update",
        missing_message="Catalogue freshness cannot be verified because no products are loaded.",
        stale_after_hours=catalogue_stale_after_hours,
    )
    price_watermark, price_status, price_blockers = measure_dimension(
        tuple(price_values),
        dimension="Price",
        timestamp_label="prices_updated_at",
        missing_message="Price freshness cannot be verified because no price snapshots are loaded.",
        stale_after_hours=price_stale_after_hours,
    )
    blockers = (*catalogue_blockers, *price_blockers)
    status: Literal["fresh", "stale", "degraded"]
    if "degraded" in {catalogue_status, price_status}:
        status = "degraded"
    elif "stale" in {catalogue_status, price_status}:
        status = "stale"
    else:
        status = "fresh"
    return (
        catalog_watermark,
        price_watermark,
        catalogue_status,
        price_status,
        status,
        blockers,
    )


def _normalise_source_authority_expiry(expires_at: object) -> datetime | None:
    """Normalise trustworthy expiry evidence and reject missing or naive values."""

    if (
        not isinstance(expires_at, datetime)
        or expires_at.tzinfo is None
        or expires_at.utcoffset() is None
    ):
        return None
    return expires_at.astimezone(UTC)


def _source_authority_is_active(expires_at: object, *, now: datetime) -> bool:
    """Return whether independently verified source authority is still active."""

    normalised_expiry = _normalise_source_authority_expiry(expires_at)
    return normalised_expiry is not None and normalised_expiry > now


def _api_category(value: DomainCategory) -> ComponentCategory:
    return _DOMAIN_TO_API_CATEGORY[value]


def _review_sentiment_label(value: float) -> Literal["positive", "neutral", "negative"]:
    """Convert a stored scalar sentiment without inventing a mixed label."""

    if value >= _REVIEW_POSITIVE_SENTIMENT_THRESHOLD:
        return "positive"
    if value <= _REVIEW_NEGATIVE_SENTIMENT_THRESHOLD:
        return "negative"
    return "neutral"


def _api_status(value: object) -> CompatibilityStatus:
    raw = getattr(value, "value", value)
    return CompatibilityStatus(str(raw).casefold())


def _optimizer_status(response: ApplicationBuildGenerationResponse) -> SolverStatus:
    return SolverStatus(response.optimizer_status.value.upper())


def _optimizer_version(
    response: ApplicationBuildGenerationResponse, services: ApplicationServices
) -> str:
    return response.optimizer_version or services.versions.optimizer_version


def _domain_request(request: GenerateBuildsRequest) -> DomainBuildRequest:
    minimum_vram = request.requirements.minimum_gpu_vram_gb
    profiles = request.requested_profiles or [
        BuildProfile(profile.value) for profile in _PROFILE_ORDER[: request.max_builds]
    ]
    return DomainBuildRequest(
        budget_sgd=Decimal(str(request.budget_sgd)),
        workloads=[
            WorkloadPreference(name=DomainWorkload(item.name.value), weight=item.weight)
            for item in request.workloads
        ],
        existing_products=[
            ExistingComponent(
                category=_API_TO_DOMAIN_CATEGORY[item.category],
                product_id=item.product_id,
            )
            for item in request.existing_products
        ],
        requirements=DomainRequirements(
            minimum_gpu_vram_gb=minimum_vram,
            minimum_memory_gb=request.requirements.minimum_memory_gb,
            storage_gb=request.requirements.storage_gb,
            wifi_required=request.requirements.wifi_required,
            case_size=request.requirements.case_size,
            in_stock_only=request.requirements.in_stock_only,
        ),
        preferences=DomainPreferences(
            noise=request.preferences.noise,
            upgradeability=request.preferences.upgradeability,
            power_efficiency=request.preferences.power_efficiency,
            preferred_brands=request.preferences.preferred_brands,
            excluded_brands=request.preferences.excluded_brands,
        ),
        performance_target=request.performance_target,
        requested_profiles=[DomainProfile(profile.value) for profile in profiles],
    )


def _source_url(product: Any) -> str | None:
    provenance = getattr(product, "provenance", ())
    return provenance[0].source_url if provenance else None


_BUILDCORES_ODC_BY_URI = "https://opendatacommons.org/licenses/by/1-0/"
_BUILDCORES_PUBLIC_ATTRIBUTION = (
    "Contains information from BuildCores OpenDB, made available under the "
    "ODC Attribution License v1.0."
)


def _attribution_terms(provenance: Any) -> tuple[str | None, str | None]:
    """Return a precise public notice only for the recognized licensed source contract."""

    source_name = str(provenance.source_name).casefold()
    licence_note = str(provenance.licence_or_access_note).casefold()
    if "buildcores" in source_name and "odc-by" in licence_note:
        return _BUILDCORES_PUBLIC_ATTRIBUTION, _BUILDCORES_ODC_BY_URI
    return None, None


def _source_attributions(products: Sequence[Any]) -> list[SourceAttribution]:
    """Return stable, visible source notices without silently dropping licence terms."""

    unique: dict[tuple[str, str, str, str | None, str | None, datetime], SourceAttribution] = {}
    for product in products:
        for provenance in getattr(product, "provenance", ()):
            attribution_notice, licence_url = _attribution_terms(provenance)
            item = SourceAttribution(
                source_name=provenance.source_name,
                source_url=provenance.source_url,
                licence_or_access_note=provenance.licence_or_access_note,
                attribution_notice=attribution_notice,
                licence_url=licence_url,
                retrieved_at=provenance.retrieved_at,
            )
            unique[
                (
                    item.source_name,
                    item.source_url,
                    item.licence_or_access_note,
                    item.attribution_notice,
                    item.licence_url,
                    item.retrieved_at,
                )
            ] = item
    return [
        unique[key]
        for key in sorted(
            unique,
            key=lambda item: (item[0].casefold(), item[1], item[2], item[3], item[4], item[5]),
        )
    ]


def _public_catalogue_readiness(
    report: CatalogReadinessReport | None,
    *,
    blockers: Sequence[str],
) -> CatalogueReadinessSummary | None:
    """Project the release gate into a bounded response safe for public clients."""

    if not isinstance(report, CatalogReadinessReport):
        return None
    return CatalogueReadinessSummary(
        products_by_category=dict(report.products_by_category),
        compatibility_ready_products_by_category=dict(
            report.compatibility_ready_products_by_category
        ),
        matched_listings_by_category=dict(report.matched_listings_by_category),
        in_stock_listings_by_category=dict(report.in_stock_listings_by_category),
        offer_count=report.offer_count,
        mapping_rate=report.mapping_rate,
        has_complete_priced_coverage=report.has_complete_priced_coverage,
        has_complete_in_stock_coverage=report.has_complete_in_stock_coverage,
        product_provenance_complete_count=report.product_provenance_complete_count,
        offer_provenance_complete_count=report.offer_provenance_complete_count,
        offer_rights_production_valid_count=report.offer_rights_production_valid_count,
        rights_territory=report.rights_territory,
        entity_resolution_model_version=report.entity_resolution_model_version,
        entity_resolution_model_production_authorized=(
            report.entity_resolution_model_production_authorized
        ),
        production_ready=not blockers,
        production_blockers=list(blockers),
    )


class CoreRecommendationService(RecommendationApplication):
    """Transport adapter over the real retrieval/ranking/CP-SAT application."""

    def __init__(
        self,
        settings: ApiSettings,
        services: ApplicationServices,
        reader: InMemoryCatalogReader,
        processed_data: ProcessedCatalogData,
        durable_store: SqlAlchemyDurableStore | None = None,
        semantic_encoder_ready: bool | None = None,
        release_artifacts_verified: bool = False,
        source_authority_expires_at: datetime | None = None,
    ) -> None:
        self.services = services
        self.reader = reader
        self.processed_data = processed_data
        self._durable_store = durable_store
        self._semantic_encoder_ready = (
            bool(semantic_encoder_ready)
            if semantic_encoder_ready is not None
            else not settings.requires_durable_storage
        )
        if release_artifacts_verified:
            self._release_artifact_verification: Literal[
                "verified", "development_unverified", "not_verified"
            ] = "verified"
        elif settings.is_development_environment:
            self._release_artifact_verification = "development_unverified"
        else:
            self._release_artifact_verification = "not_verified"
        self._source_authority_expires_at = source_authority_expires_at
        self.settings = settings.model_copy(
            update={
                "data_version": services.versions.data_version,
                "ranking_model_version": services.versions.ranking_model,
                "compatibility_rule_version": services.versions.rule_version,
                "solver_version": services.versions.optimizer_version,
            }
        )
        self._mutation_lock = asyncio.Lock()
        self._interactions: list[tuple[str, CanonicalInteractionEvent, datetime]] = []
        self._generated_at_by_request: dict[str, datetime] = {}

    async def ready(self) -> bool:
        return all(value == "ready" for value in (await self.readiness_checks()).values())

    async def close(self) -> None:
        durable_store = getattr(self, "_durable_store", None)
        if durable_store is not None:
            await asyncio.to_thread(durable_store.engine.dispose)

    def _release_artifact_status(
        self,
    ) -> Literal["verified", "development_unverified", "not_verified"]:
        """Return the bounded serving-release verification state.

        Production eligibility must never be inferred from catalogue contents alone.  The
        immutable serving release is a separate trust boundary and therefore remains an
        explicit input to readiness and every decision-serving operation.
        """

        return getattr(
            self,
            "_release_artifact_verification",
            (
                "development_unverified"
                if self.settings.is_development_environment
                else "not_verified"
            ),
        )

    async def _require_production_operation_eligible(
        self,
        operation: Literal["generate", "search", "replace"],
    ) -> None:
        """Fail closed when a production decision would use ineligible evidence.

        Startup and readiness probes are advisory to load balancers; they do not prevent a
        caller from reaching an already-running worker.  Re-evaluate the live freshness and
        source-authority contract at the point of use so an expiry or stale watermark takes
        effect without a restart.
        """

        if not self.settings.requires_durable_storage:
            return
        evidence = await self.freshness()
        if evidence.production_ready:
            return
        raise ApiError(
            status_code=503,
            code="production_serving_not_ready",
            message=(
                "Production recommendations are temporarily unavailable because the active "
                "serving evidence is not eligible."
            ),
            details={
                "operation": operation,
                "freshness_status": evidence.status,
                "catalogue_status": evidence.catalogue_status,
                "price_status": evidence.price_status,
                "release_artifact_verification": evidence.release_artifact_verification,
                "source_authority_expires_at": evidence.source_authority_expires_at,
                "readiness_blockers": evidence.readiness_blockers,
                "retryable": True,
            },
        )

    async def readiness_checks(self) -> dict[str, Literal["ready", "not_ready"]]:
        categories = {product.category for product in self.processed_data.products}
        all_categories = set(DomainCategory)
        report = self.processed_data.readiness
        now = datetime.now(UTC)
        _, _, _, _, freshness_status, _ = _processed_catalog_freshness(
            self.processed_data,
            catalogue_stale_after_hours=self.settings.catalogue_stale_after_hours,
            price_stale_after_hours=self.settings.price_stale_after_hours,
            now=now,
        )
        freshness_ready = freshness_status == "fresh"
        release_artifacts_verified = self._release_artifact_status() == "verified"
        production_ready = (
            report is not None
            and not report.blockers()
            and freshness_ready
            and release_artifacts_verified
        )
        application_catalog = getattr(self.services, "catalog", None)
        authority_policy = getattr(
            application_catalog,
            "compatibility_evidence_policy",
            None,
        )
        authority_ready = authority_policy != AUTHORITATIVE_COMPATIBILITY_POLICY or bool(
            getattr(application_catalog, "has_authoritative_compatibility_coverage", False)
        )
        durable_store = getattr(self, "_durable_store", None)
        if durable_store is None:
            durable_storage_ready = not self.settings.requires_durable_storage
        else:
            durable_storage_ready = await asyncio.to_thread(durable_store.is_ready)
        semantic_encoder_ready = getattr(
            self,
            "_semantic_encoder_ready",
            not self.settings.requires_durable_storage,
        )
        source_authority_expires_at = _normalise_source_authority_expiry(
            getattr(self, "_source_authority_expires_at", None)
        )
        source_authority_ready = not self.settings.requires_durable_storage or (
            _source_authority_is_active(source_authority_expires_at, now=now)
        )
        return {
            "catalogue": "ready" if self.processed_data.products else "not_ready",
            "category_coverage": "ready" if categories == all_categories else "not_ready",
            "priced_coverage": (
                "ready" if self.processed_data.stats.has_complete_priced_coverage else "not_ready"
            ),
            "in_stock_coverage": (
                "ready" if self.processed_data.stats.has_complete_in_stock_coverage else "not_ready"
            ),
            "catalogue_freshness": "ready" if freshness_ready else "not_ready",
            "production_catalog_policy": "ready" if production_ready else "not_ready",
            "compatibility_engine": (
                "ready"
                if self.services.generate_builds.compatibility_engine.rule_version
                == self.services.versions.rule_version
                else "not_ready"
            ),
            "compatibility_authority": "ready" if authority_ready else "not_ready",
            "optimizer": "ready" if self.services.versions.optimizer_version else "not_ready",
            "durable_storage": "ready" if durable_storage_ready else "not_ready",
            "semantic_encoder": ("ready" if semantic_encoder_ready else "not_ready"),
            "serving_release": "ready" if release_artifacts_verified else "not_ready",
            "source_authority": "ready" if source_authority_ready else "not_ready",
        }

    async def freshness(self) -> FreshnessResponse:
        if not self.processed_data.products:
            raise ApiError(
                status_code=503,
                code="catalog_unavailable",
                message="No verified catalogue products are loaded.",
            )
        now = datetime.now(UTC)
        (
            catalog_watermark,
            price_watermark,
            catalogue_status,
            price_status,
            freshness_status,
            freshness_blockers,
        ) = (
            _processed_catalog_freshness(
                self.processed_data,
                catalogue_stale_after_hours=self.settings.catalogue_stale_after_hours,
                price_stale_after_hours=self.settings.price_stale_after_hours,
                now=now,
            )
        )
        readiness = self.processed_data.readiness
        blockers = (
            list(readiness.blockers())
            if readiness is not None
            else ["A measured production-readiness report is unavailable."]
        )
        blockers.extend(item for item in freshness_blockers if item not in blockers)
        source_authority_expires_at = _normalise_source_authority_expiry(
            getattr(self, "_source_authority_expires_at", None)
        )
        if self.settings.requires_durable_storage and (
            not _source_authority_is_active(source_authority_expires_at, now=now)
        ):
            blockers.append(
                "Signed retailer-source authority is expired or its earliest expiry is unavailable."
            )
        application_catalog = getattr(self.services, "catalog", None)
        if getattr(
            application_catalog, "compatibility_evidence_policy", None
        ) == AUTHORITATIVE_COMPATIBILITY_POLICY and not getattr(
            application_catalog,
            "has_authoritative_compatibility_coverage",
            False,
        ):
            blockers.append(
                "Manufacturer-authoritative compatibility coverage is incomplete; "
                "community specifications cannot support production PASS decisions."
            )
        source_names = {
            provenance.source_name
            for product in self.processed_data.products
            for provenance in product.provenance
        }
        source_names.update(
            provenance.source_name for provenance in self.processed_data.listing_provenance
        )
        release_artifact_verification = self._release_artifact_status()
        if release_artifact_verification != "verified" and (
            self.settings.requires_durable_storage or not blockers
        ):
            blockers.append(
                "Immutable serving-release artifacts have not been independently verified."
            )
        return FreshnessResponse(
            data_version=self.services.versions.data_version,
            status=freshness_status,
            catalogue_status=catalogue_status,
            price_status=price_status,
            last_catalog_update=catalog_watermark,
            prices_updated_at=price_watermark,
            stale_after_hours=self.settings.price_stale_after_hours,
            catalogue_stale_after_hours=self.settings.catalogue_stale_after_hours,
            price_stale_after_hours=self.settings.price_stale_after_hours,
            source_count=len(source_names),
            product_count=self.processed_data.stats.product_count,
            listing_count=self.processed_data.stats.matched_listing_count,
            production_ready=not blockers,
            release_artifact_verification=release_artifact_verification,
            source_authority_expires_at=source_authority_expires_at,
            readiness_blockers=blockers,
            catalogue_readiness=_public_catalogue_readiness(
                readiness,
                blockers=blockers,
            ),
        )

    async def generate_builds(self, request: GenerateBuildsRequest) -> GenerateBuildsResponse:
        await self._require_production_operation_eligible("generate")
        domain_request = _domain_request(request)
        included = frozenset(
            item.product_id for item in request.existing_products if item.include_in_budget
        )
        request_id = f"req_{uuid4().hex}"
        async with self._mutation_lock:
            try:
                response = await asyncio.to_thread(
                    self.services.generate_builds.generate,
                    domain_request,
                    request_id=request_id,
                    included_existing_product_ids=included,
                )
            except RequestConflictError as error:
                raise ApiError(
                    status_code=409,
                    code="request_conflict",
                    message=str(error),
                ) from error
            except DurableStorageError as error:
                raise ApiError(
                    status_code=503,
                    code="durable_storage_unavailable",
                    message=str(error),
                ) from error
            except (EmptyCatalogError, CatalogIntegrityError) as error:
                raise ApiError(
                    status_code=503,
                    code="catalog_unavailable",
                    message=str(error),
                ) from error
            except (ApplicationError, ValueError) as error:
                raise ApiError(
                    status_code=422,
                    code="invalid_generation_request",
                    message=str(error),
                ) from error
        stored_at: datetime | None = None
        owned_product_ids: frozenset[str] | None = None
        if getattr(self, "_durable_store", None) is not None:
            try:
                stored = await asyncio.to_thread(
                    self.services.results.require_generation, response.request_id
                )
            except DurableStorageError as error:
                raise ApiError(
                    status_code=503,
                    code="durable_storage_unavailable",
                    message=str(error),
                ) from error
            stored_at = stored.stored_at
            owned_product_ids = stored.owned_product_ids
        return self._generation_response(
            response,
            domain_request,
            generated_at=stored_at,
            owned_product_ids=owned_product_ids,
        )

    async def get_request_builds(self, request_id: str) -> GenerateBuildsResponse:
        try:
            generation = await asyncio.to_thread(self.services.results.get_generation, request_id)
        except DurableStorageError as error:
            raise ApiError(
                status_code=503,
                code="durable_storage_unavailable",
                message=str(error),
            ) from error
        if generation is None:
            raise ApiError(
                status_code=404,
                code="request_not_found",
                message=f"No request exists with ID '{request_id}'.",
            )
        return self._generation_response(
            generation.response,
            generation.request,
            generated_at=generation.stored_at,
            owned_product_ids=generation.owned_product_ids,
        )

    async def get_build(self, build_id: str) -> BuildSummary:
        try:
            generation = await asyncio.to_thread(
                self.services.results.generation_for_build, build_id
            )
        except DurableStorageError as error:
            raise ApiError(
                status_code=503,
                code="durable_storage_unavailable",
                message=str(error),
            ) from error
        except (ResultNotFoundError, KeyError) as error:
            raise ApiError(
                status_code=404,
                code="build_not_found",
                message=f"No build exists with ID '{build_id}'.",
            ) from error
        build = next(item for item in generation.response.builds if item.build_id == build_id)
        return self._build_summary(
            build,
            generation.request,
            generation.response,
            generated_at=generation.stored_at,
            owned_product_ids=generation.owned_product_ids,
        )

    async def admin_operations(self) -> AdminOperationsResponse:
        """Expose bounded aggregate operations data without leaking raw retailer records."""

        stats = self.processed_data.stats
        report = self.processed_data.readiness
        now = datetime.now(UTC)
        snapshots = self.processed_data.price_snapshots
        latest_snapshots_by_listing: dict[str, Any] = {}
        for index, snapshot in enumerate(snapshots):
            listing_id = str(getattr(snapshot, "listing_id", "")) or f"__snapshot_{index}"
            incumbent = latest_snapshots_by_listing.get(listing_id)
            if incumbent is None or snapshot.observed_at > incumbent.observed_at:
                latest_snapshots_by_listing[listing_id] = snapshot
        latest_snapshots = tuple(latest_snapshots_by_listing.values())
        newest = max((item.observed_at for item in latest_snapshots), default=None)
        stale_cutoff = now - timedelta(hours=self.settings.price_stale_after_hours)
        missing_fields: list[AdminMissingField] = []
        release_blockers: list[str]
        if report is None:
            release_blockers = [
                "No measured catalogue-readiness report is attached to this release."
            ]
        else:
            release_blockers = list(report.blockers())
            for category, groups in sorted(report.critical_field_present_by_category.items()):
                if category not in {item.value for item in ComponentCategory}:
                    continue
                product_count = report.products_by_category.get(category, 0)
                for field_group, present_count in sorted(groups.items()):
                    missing_count = max(0, product_count - present_count)
                    if missing_count:
                        missing_fields.append(
                            AdminMissingField(
                                category=ComponentCategory(category),
                                field_group=field_group,
                                missing_product_count=missing_count,
                                product_count=product_count,
                            )
                        )
        pipeline_summary = summarize_pipeline_operations(
            self.settings.pipeline_operations_path,
            now=now,
            window=timedelta(hours=self.settings.pipeline_operations_window_hours),
        )
        pipeline_operations = None
        pipeline_notes: list[str] = []
        pipeline_events_available = False
        if pipeline_summary is None:
            pipeline_notes.append(
                "Instrumented pipeline-operation receipts are not configured for this serving "
                "release. Dagster control-plane status is not inferred."
            )
        elif not pipeline_summary.available:
            pipeline_notes.append(
                "The configured pipeline-operation receipt mount is unavailable or invalid. "
                "Treat pipeline failure status as unknown."
            )
        else:
            pipeline_events_available = True
            pipeline_operations = AdminPipelineOperations(
                event_window_hours=self.settings.pipeline_operations_window_hours,
                event_count=pipeline_summary.event_count,
                succeeded_count=pipeline_summary.succeeded_count,
                failed_count=pipeline_summary.failed_count,
                latest_event_at=pipeline_summary.latest_event_at,
                latest_failure_at=pipeline_summary.latest_failure_at,
                invalid_receipt_count=pipeline_summary.invalid_receipt_count,
                truncated=pipeline_summary.truncated,
            )
            pipeline_notes.append(
                "Pipeline counters come from bounded receipts emitted by instrumented user-code "
                "only; inspect authenticated Dagster for scheduler, queue, or worker failures."
            )
            if pipeline_summary.invalid_receipt_count:
                pipeline_notes.append(
                    "One or more pipeline-operation receipts were invalid and excluded from "
                    "aggregate counters."
                )
            if pipeline_summary.truncated:
                pipeline_notes.append(
                    "The receipt reader reached its bounded event limit; older events may be "
                    "excluded from this response."
                )
        return AdminOperationsResponse(
            data_version=stats.data_version,
            generated_at=now,
            mode="processed_catalog",
            mapping_queue=AdminMappingQueue(
                offer_count=stats.offer_count,
                matched_count=stats.matched_listing_count,
                unmatched_count=stats.unmatched_offer_count,
                manual_review_count=stats.manual_review_count,
                rejected_conflict_count=stats.rejected_conflict_count,
                model_rejected_count=stats.model_rejected_count,
            ),
            price_freshness=AdminPriceFreshness(
                snapshot_count=len(latest_snapshots),
                newest_observed_at=newest,
                stale_snapshot_count=sum(
                    1 for snapshot in latest_snapshots if snapshot.observed_at < stale_cutoff
                ),
                stale_after_hours=self.settings.price_stale_after_hours,
            ),
            missing_critical_fields=missing_fields,
            release_blockers=release_blockers,
            pipeline_operations=pipeline_operations,
            pipeline_failure_events_available=pipeline_events_available,
            notes=[
                "Price freshness evaluates the latest observation for each distinct listing.",
                *pipeline_notes,
            ],
        )

    async def create_build_share(self, build_id: str) -> BuildShareCreated:
        """Create a revocable, immutable public share only on a durable serving backend."""

        durable_store = getattr(self, "_durable_store", None)
        if durable_store is None:
            raise ApiError(
                status_code=503,
                code="build_sharing_requires_durable_storage",
                message="Server-backed public build shares require durable storage.",
            )
        build = await self.get_build(build_id)
        expires_at = datetime.now(UTC) + timedelta(hours=self.settings.build_share_ttl_hours)
        try:
            created = await asyncio.to_thread(
                durable_store.create_build_share,
                build_id=build_id,
                snapshot=public_build_snapshot(build).model_dump(mode="json"),
                expires_at=expires_at,
            )
        except DurableStorageError as error:
            raise ApiError(
                status_code=503,
                code="durable_storage_unavailable",
                message=str(error),
            ) from error
        except ResultNotFoundError as error:
            raise ApiError(
                status_code=404,
                code="build_not_found",
                message=f"No build exists with ID '{build_id}'.",
            ) from error
        return BuildShareCreated(
            share_id=created.share_id,
            revocation_token=created.revocation_token,
            created_at=created.created_at,
            expires_at=created.expires_at,
        )

    async def get_build_share(self, share_id: str) -> PublicBuildShare:
        durable_store = getattr(self, "_durable_store", None)
        if durable_store is None:
            raise ApiError(
                status_code=503,
                code="build_sharing_requires_durable_storage",
                message="Server-backed public build shares require durable storage.",
            )
        try:
            stored = await asyncio.to_thread(durable_store.get_active_build_share, share_id)
        except DurableStorageError as error:
            raise ApiError(
                status_code=503,
                code="durable_storage_unavailable",
                message=str(error),
            ) from error
        if stored is None:
            raise ApiError(
                status_code=404,
                code="build_share_not_found",
                message="No active public build share exists with this ID.",
            )
        try:
            snapshot = PublicBuildSnapshot.model_validate(stored.snapshot)
        except ValueError as error:
            raise ApiError(
                status_code=503,
                code="durable_storage_unavailable",
                message="Stored public build share failed its response contract.",
            ) from error
        return PublicBuildShare(
            share_id=stored.share_id,
            created_at=stored.created_at,
            expires_at=stored.expires_at,
            snapshot=snapshot,
        )

    async def revoke_build_share(
        self, share_id: str, request: RevokeBuildShareRequest
    ) -> BuildShareRevoked:
        durable_store = getattr(self, "_durable_store", None)
        if durable_store is None:
            raise ApiError(
                status_code=503,
                code="build_sharing_requires_durable_storage",
                message="Server-backed public build shares require durable storage.",
            )
        try:
            revoked_at = await asyncio.to_thread(
                durable_store.revoke_build_share,
                share_id,
                request.revocation_token,
            )
        except DurableStorageError as error:
            raise ApiError(
                status_code=503,
                code="durable_storage_unavailable",
                message=str(error),
            ) from error
        if revoked_at is None:
            raise ApiError(
                status_code=404,
                code="build_share_not_found",
                message="No active public build share exists with this ID.",
            )
        return BuildShareRevoked(share_id=share_id, revoked_at=revoked_at)

    async def replace_component(
        self, build_id: str, request: ReplacementRequest
    ) -> ReplacementResponse:
        await self._require_production_operation_eligible("replace")
        try:
            prior = await asyncio.to_thread(self.services.results.generation_for_build, build_id)
            old_build = next(item for item in prior.response.builds if item.build_id == build_id)
        except DurableStorageError as error:
            raise ApiError(
                status_code=503,
                code="durable_storage_unavailable",
                message=str(error),
            ) from error
        except (ResultNotFoundError, KeyError) as error:
            raise ApiError(
                status_code=404,
                code="build_not_found",
                message=f"No build exists with ID '{build_id}'.",
            ) from error
        mode = ReplacementMode(request.mode)
        async with self._mutation_lock:
            try:
                response = await asyncio.to_thread(
                    self.services.replace_component.replace,
                    build_id,
                    category=_API_TO_DOMAIN_CATEGORY[request.category],
                    replacement_product_id=request.replacement_product_id,
                    mode=mode,
                    request_id=f"req_{uuid4().hex}",
                )
            except DurableStorageError as error:
                raise ApiError(
                    status_code=503,
                    code="durable_storage_unavailable",
                    message=str(error),
                ) from error
            except (ApplicationError, KeyError, ValueError) as error:
                raise ApiError(
                    status_code=409,
                    code="incompatible_replacement",
                    message=str(error),
                ) from error
        self._assert_conclusive(response)
        if not response.builds:
            raise ApiError(
                status_code=409,
                code="incompatible_replacement",
                message="Replacement did not produce a complete independently validated build.",
                details={"reasons": response.infeasibility_reasons},
            )
        new_build = response.builds[0]
        try:
            stored = await asyncio.to_thread(
                self.services.results.require_generation, response.request_id
            )
        except DurableStorageError as error:
            raise ApiError(
                status_code=503,
                code="durable_storage_unavailable",
                message=str(error),
            ) from error
        old_ids = {item.category: item.product_id for item in old_build.components}
        new_ids = {item.category: item.product_id for item in new_build.components}
        changed = [
            _api_category(category)
            for category in DomainCategory
            if old_ids.get(category) != new_ids.get(category)
        ]
        price_delta = float(new_build.total_price_sgd - old_build.total_price_sgd)
        workload_deltas = {
            workload.value: round(
                float(score) - float(old_build.workload_scores.get(workload, 0.0)), 6
            )
            for workload, score in new_build.workload_scores.items()
        }
        summary = self._build_summary(
            new_build,
            stored.request,
            response,
            generated_at=stored.stored_at,
            owned_product_ids=stored.owned_product_ids,
        )
        return ReplacementResponse(
            build=summary,
            changed_categories=changed,
            price_delta_sgd=price_delta,
            workload_score_deltas=workload_deltas,
            new_warnings=summary.warnings,
            data_version=response.data_version,
            ranking_model=response.ranking_model,
            rule_version=response.rule_version,
            solver_version=_optimizer_version(response, self.services),
            solver_status=_optimizer_status(response),
            solver_ran=response.optimizer_ran,
            solver_profile_statuses=[
                SolverProfileOutcome(
                    profile=BuildProfile(item.profile.value),
                    status=SolverStatus(item.status.value.upper()),
                    wall_time_seconds=item.wall_time_seconds,
                    objective_value=item.objective_value,
                )
                for item in response.optimizer_profile_statuses
            ],
            solver_validator_rejections=response.optimizer_validator_rejections,
        )

    async def search_products(self, request: ProductSearchRequest) -> ProductSearchResponse:
        await self._require_production_operation_eligible("search")
        try:
            requested_page = resolve_product_search_page(request)
        except InvalidProductSearchCursor as error:
            raise ApiError(
                status_code=422,
                code="invalid_pagination_cursor",
                message=str(error),
            ) from error

        # Retrieve the complete filtered candidate universe so ``total`` is not merely the
        # size of an arbitrary top-k window. The processed service currently uses the bounded
        # in-memory snapshot retriever; each category is capped by its measured snapshot size.
        categories = list(ComponentCategory)
        if request.compatible_with_build_id is not None and request.category is not None:
            categories = [request.category]
        all_results: list[SearchProductResult] = []
        retrieved_candidates = 0
        filtered_incompatible = 0
        filtered_unknown = 0
        query = request.query
        for category in categories:
            domain_category = _API_TO_DOMAIN_CATEGORY[category]
            category_capacity = sum(
                item.product.category is domain_category for item in self.services.catalog.items
            )
            if category_capacity == 0:
                continue
            try:
                outcome = await asyncio.to_thread(
                    self.services.search_products.search_with_outcome,
                    query,
                    category=domain_category,
                    filters=StructuredFilters(in_stock_only=request.in_stock_only),
                    top_k=category_capacity,
                    compatible_with_build_id=request.compatible_with_build_id,
                )
            except DurableStorageError as error:
                raise ApiError(
                    status_code=503,
                    code="durable_storage_unavailable",
                    message=str(error),
                ) from error
            except ApplicationError as error:
                raise ApiError(
                    status_code=404,
                    code="build_not_found",
                    message=str(error),
                ) from error
            all_results.extend(outcome.results)
            retrieved_candidates += outcome.retrieved_candidates
            filtered_incompatible += outcome.filtered_incompatible
            filtered_unknown += outcome.filtered_unknown
        all_results.sort(
            key=lambda item: (
                item.rank,
                item.product.canonical_name.casefold(),
                item.product_id,
            )
        )

        category_counts = {
            category: sum(_api_category(hit.product.category) is category for hit in all_results)
            for category in ComponentCategory
        }
        category_scoped = [
            hit
            for hit in all_results
            if request.category is None or _api_category(hit.product.category) is request.category
        ]
        brand_counts: dict[str, int] = {}
        for hit in category_scoped:
            brand_counts[hit.product.brand] = brand_counts.get(hit.product.brand, 0) + 1
        results = [
            hit
            for hit in category_scoped
            if request.brand is None or hit.product.brand.casefold() == request.brand.casefold()
        ]
        filtered_category = len(all_results) - len(category_scoped)
        filtered_brand = len(category_scoped) - len(results)

        total = len(results)
        page_size = request.effective_page_size
        total_pages = max(1, (total + page_size - 1) // page_size)
        page = min(requested_page, total_pages)
        offset = (page - 1) * page_size
        offer_display_permitted = self._price_history_is_permitted()
        products = []
        for hit in results[offset : offset + page_size]:
            status_value = getattr(hit, "compatibility_status", None)
            products.append(
                ProductSearchItem(
                    product_id=hit.product.product_id,
                    category=_api_category(hit.product.category),
                    canonical_name=hit.product.canonical_name,
                    brand=hit.product.brand,
                    model=hit.product.model,
                    lowest_price_sgd=(
                        float(hit.listing.total_price)
                        if offer_display_permitted and hit.listing is not None
                        else None
                    ),
                    stock_status=(
                        hit.listing.stock_status.value
                        if offer_display_permitted and hit.listing is not None
                        else None
                    ),
                    compatibility_status=(
                        _api_status(status_value) if status_value is not None else None
                    ),
                )
            )
        product_sources = {
            provenance.source_name
            for product in self.processed_data.products
            for provenance in product.provenance
        }
        product_sources.update(listing.retailer for listing in self.processed_data.listings)
        source_attributions = _source_attributions(self.processed_data.products)
        as_of = max(
            (
                *(product.updated_at for product in self.processed_data.products),
                *(listing.last_seen_at for listing in self.processed_data.listings),
            ),
            default=None,
        )
        query_id, structured_constraints = product_search_identity(
            request,
            data_version=self.services.versions.data_version,
            retrieval_model=self.services.versions.retrieval_model,
            rule_version=self.services.versions.rule_version,
        )
        durable_store = getattr(self, "_durable_store", None)
        if durable_store is not None:
            try:
                await asyncio.to_thread(
                    durable_store.save_search_query,
                    SearchQuery(
                        query_id=query_id,
                        raw_query=request.query,
                        structured_constraints=structured_constraints,
                    ),
                )
            except RequestConflictError as error:
                raise ApiError(
                    status_code=409,
                    code="search_query_identity_conflict",
                    message=str(error),
                ) from error
            except DurableStorageError as error:
                raise ApiError(
                    status_code=503,
                    code="durable_storage_unavailable",
                    message=str(error),
                ) from error
        return ProductSearchResponse(
            query_id=query_id,
            products=products,
            total=total,
            retrieved_candidates=retrieved_candidates,
            filtered_category=filtered_category,
            filtered_brand=filtered_brand,
            filtered_incompatible=filtered_incompatible,
            filtered_unknown=filtered_unknown,
            data_version=self.services.versions.data_version,
            retrieval_model=self.services.versions.retrieval_model,
            facets=ProductSearchFacets(
                categories=[
                    ProductFacetCount(value=category.value, count=category_counts[category])
                    for category in ComponentCategory
                    if category_counts[category] > 0
                ],
                brands=[
                    ProductFacetCount(value=brand, count=count)
                    for brand, count in sorted(
                        brand_counts.items(), key=lambda item: item[0].casefold()
                    )
                ],
            ),
            pagination=ProductSearchPagination(
                page=page,
                page_size=page_size,
                total_pages=total_pages,
                has_previous=page > 1,
                has_next=page < total_pages,
                previous_cursor=(
                    encode_product_search_cursor(request, page=page - 1) if page > 1 else None
                ),
                next_cursor=(
                    encode_product_search_cursor(request, page=page + 1)
                    if page < total_pages
                    else None
                ),
            ),
            coverage=CatalogueCoverage(
                canonical_products=self.processed_data.stats.product_count,
                retailer_listings=len(self.processed_data.listings),
                source_count=len(product_sources),
                category_count=len({product.category for product in self.processed_data.products}),
                as_of=as_of,
                scope_label="Versioned processed catalogue snapshot",
                source_attributions=source_attributions,
            ),
        )

    def _require_product(self, product_id: str) -> Any:
        product = self.reader.get_product(product_id)
        if product is None:
            raise ApiError(
                status_code=404,
                code="product_not_found",
                message=f"No product exists with ID '{product_id}'.",
            )
        return product

    def _eligible_current_listings(self, listings: Sequence[Any]) -> list[Any]:
        if not self._price_history_is_permitted():
            return []
        return [
            listing
            for listing in listings
            if listing.condition.value == "new" and listing.stock_status.value == "in_stock"
        ]

    def _price_history_is_permitted(self) -> bool:
        readiness = self.processed_data.readiness
        offer_count = int(getattr(readiness, "offer_count", 0))
        rights_valid_count = int(getattr(readiness, "offer_rights_production_valid_count", 0))
        return offer_count > 0 and rights_valid_count == offer_count

    async def get_product(self, product_id: str) -> ProductDetail:
        product = self._require_product(product_id)
        listings = self.reader.list_listings(product_id=product_id, limit=1000)
        eligible = self._eligible_current_listings(listings)
        listing = min(eligible, key=lambda item: item.total_price) if eligible else None
        return ProductDetail(
            product_id=product.product_id,
            category=_api_category(product.category),
            canonical_name=product.canonical_name,
            brand=product.brand,
            model=product.model,
            lowest_price_sgd=float(listing.total_price) if listing else None,
            stock_status=listing.stock_status.value if listing else None,
            manufacturer_part_number=product.manufacturer_part_number,
            attributes={
                **product.common_attributes.model_dump(mode="json"),
                **product.category_attributes.model_dump(mode="json"),
            },
            source_confidence=product.source_confidence,
            source_url=_source_url(product),
            source_attributions=_source_attributions((product,)),
            updated_at=product.updated_at,
            data_version=self.services.versions.data_version,
        )

    async def get_prices(self, product_id: str) -> ProductPricesResponse:
        self._require_product(product_id)
        history_is_permitted = self._price_history_is_permitted()
        if not history_is_permitted:
            # An offer can be retained for controlled data-quality and mapping
            # work without being licensed for public price display. Never let
            # the public product-price endpoint expose that retained evidence.
            return ProductPricesResponse(
                product_id=product_id,
                current_lowest_price_sgd=None,
                observations=[],
                price_intelligence=None,
                data_version=self.services.versions.data_version,
            )

        listings = self.reader.list_listings(product_id=product_id, limit=1000)
        eligible_ids = {listing.listing_id for listing in self._eligible_current_listings(listings)}
        observations: list[PriceObservation] = []
        historical_observations: list[HistoricalPriceObservation] = []
        for listing in listings:
            snapshots = self.reader.list_price_snapshots(listing.listing_id)
            for snapshot in snapshots:
                observations.append(
                    PriceObservation(
                        listing_id=listing.listing_id,
                        retailer=listing.retailer,
                        observed_at=snapshot.observed_at,
                        base_price_sgd=float(snapshot.base_price),
                        shipping_price_sgd=float(snapshot.shipping_price),
                        stock_status=snapshot.stock_status.value,
                        condition=listing.condition.value,
                        current_offer_eligible=listing.listing_id in eligible_ids,
                        listing_url=listing.listing_url,
                    )
                )
                if history_is_permitted and listing.condition.value == "new":
                    historical_observations.append(
                        HistoricalPriceObservation(
                            listing_id=listing.listing_id,
                            observed_at=snapshot.observed_at,
                            base_price=snapshot.base_price,
                            shipping_price=snapshot.shipping_price,
                            stock_status=snapshot.stock_status.value,
                            seller_name=listing.seller_name or "",
                            retailer=listing.retailer,
                            currency=listing.currency,
                            source_url=listing.listing_url,
                        )
                    )
        current = min(
            (item.total_price for item in listings if item.listing_id in eligible_ids),
            default=None,
        )
        return ProductPricesResponse(
            product_id=product_id,
            current_lowest_price_sgd=float(current) if current is not None else None,
            observations=sorted(
                observations,
                key=lambda item: (item.observed_at, item.listing_id),
                reverse=True,
            )[:_MAX_PRICE_OBSERVATIONS_RETURNED],
            price_intelligence=summarize_price_history(historical_observations),
            data_version=self.services.versions.data_version,
        )

    async def get_benchmarks(self, product_id: str) -> ProductBenchmarksResponse:
        self._require_product(product_id)
        benchmarks = self.reader.list_benchmarks(product_id)
        return ProductBenchmarksResponse(
            product_id=product_id,
            benchmarks=[
                BenchmarkObservation(
                    benchmark_name=item.benchmark_name,
                    workload=item.workload.value,
                    score=item.score,
                    unit=item.unit,
                    higher_is_better=item.higher_is_better,
                    basis="observed",
                    source_url=item.source_url,
                    observed_at=item.observed_at,
                )
                for item in benchmarks
            ],
            data_version=self.services.versions.data_version,
            performance_model_version=self.services.versions.performance_model,
        )

    async def get_reviews(self, product_id: str) -> ProductReviewsResponse:
        self._require_product(product_id)
        item = self.services.catalog.get(product_id)
        if item is None:
            raise ApiError(
                status_code=503,
                code="catalog_snapshot_mismatch",
                message="The product is absent from the active catalogue snapshot.",
            )
        return ProductReviewsResponse(
            product_id=product_id,
            evidence=[
                ApiReviewEvidence(
                    aspect=evidence.aspect,
                    sentiment=_review_sentiment_label(evidence.sentiment),
                    evidence_text=evidence.evidence_text,
                    source_url=evidence.source_url,
                    published_at=evidence.published_at,
                    confidence=evidence.confidence,
                )
                for evidence in item.review_evidence
            ],
            data_version=self.services.versions.data_version,
        )

    async def check_compatibility(
        self, request: CompatibilityCheckRequest
    ) -> CompatibilityCheckResponse:
        records: dict[str, Mapping[str, Any]] = {}
        checks: list[CompatibilityCheck] = []
        for item in request.components:
            category = _API_TO_DOMAIN_CATEGORY[item.category]
            if category.value in records:
                checks.append(
                    CompatibilityCheck(
                        rule_id="exactly-one-per-category",
                        status=CompatibilityStatus.FAIL,
                        message=f"Multiple {item.category.value} components were supplied.",
                        affected_categories=[item.category],
                    )
                )
                continue
            catalog_item = self.services.catalog.get(item.product_id or "")
            if catalog_item is None or catalog_item.product.category != category:
                checks.append(
                    CompatibilityCheck(
                        rule_id="verified-catalog-product",
                        status=CompatibilityStatus.UNKNOWN,
                        message="Compatibility requires a verified canonical product and category.",
                        affected_categories=[item.category],
                    )
                )
                continue
            records[category.value] = catalog_item.compatibility_record
        if not checks and len(records) == len(ComponentCategory):
            report = self.services.generate_builds.compatibility_engine.check_complete_build(
                records
            )
            checks = [self._compatibility_check(item) for item in report.results]
        elif len(records) != len(ComponentCategory):
            missing = [
                category
                for category in ComponentCategory
                if _API_TO_DOMAIN_CATEGORY[category].value not in records
            ]
            checks.append(
                CompatibilityCheck(
                    rule_id="complete-build",
                    status=CompatibilityStatus.UNKNOWN,
                    message="A complete decision requires all eight component categories.",
                    affected_categories=missing,
                )
            )
        status = self._aggregate_status(checks)
        return CompatibilityCheckResponse(
            status=status,
            is_feasible=status in {CompatibilityStatus.PASS, CompatibilityStatus.WARNING},
            checks=checks,
            rule_version=self.services.versions.rule_version,
            data_version=self.services.versions.data_version,
        )

    async def record_interaction(
        self, event: CanonicalInteractionEvent
    ) -> InteractionAccepted:
        event_id = interaction_event_id(event)
        accepted_at = datetime.now(UTC)
        domain_event = DomainInteractionEvent(
            event_id=event_id,
            session_id=event.session_id,
            user_id=event.user_id,
            query_id=event.query_id,
            product_id=event.product_id,
            build_id=event.build_id,
            event_type=InteractionType(event.event_type),
            rank_position=event.rank_position,
            model_version=event.model_version,
            data_version=event.data_version,
            rule_version=event.rule_version,
            metadata=event.metadata,
            impression_id=event.impression_id,
            trust_level=event.trust_level,
            idempotency_key_sha256=event.idempotency_key_sha256,
            idempotency_payload_sha256=event.idempotency_payload_sha256,
            created_at=accepted_at,
        )
        durable_store = getattr(self, "_durable_store", None)
        replayed = False
        if durable_store is None:
            existing = next(
                (item for item in self._interactions if item[0] == event_id),
                None,
            )
            if existing is not None:
                _, existing_event, existing_at = existing
                if not same_canonical_interaction(existing_event, event):
                    raise ApiError(
                        status_code=409,
                        code="interaction_idempotency_conflict",
                        message="Idempotency-Key was already used for another interaction.",
                    )
                accepted_at = existing_at
                replayed = True
            else:
                self._interactions.append((event_id, event, accepted_at))
        else:
            try:
                write = await asyncio.to_thread(durable_store.put_interaction, domain_event)
                accepted_at = write.event.created_at
                replayed = write.replayed
            except RequestConflictError as error:
                is_idempotency_conflict = "Idempotency-Key" in str(error)
                raise ApiError(
                    status_code=409,
                    code=(
                        "interaction_idempotency_conflict"
                        if is_idempotency_conflict
                        else "interaction_reference_conflict"
                    ),
                    message=str(error),
                ) from error
            except DurableStorageError as error:
                raise ApiError(
                    status_code=503,
                    code="durable_storage_unavailable",
                    message=str(error),
                ) from error
        return InteractionAccepted(
            event_id=event_id,
            accepted_at=accepted_at,
            data_version=event.data_version or self.services.versions.data_version,
            rule_version=event.rule_version or self.services.versions.rule_version,
            trust_level=event.trust_level,
            replayed=replayed,
        )

    def _generation_response(
        self,
        response: ApplicationBuildGenerationResponse,
        request: DomainBuildRequest,
        *,
        generated_at: datetime | None = None,
        owned_product_ids: Collection[str] | None = None,
    ) -> GenerateBuildsResponse:
        canonical_generated_at = self._generated_at_by_request.setdefault(
            response.request_id, generated_at or datetime.now(UTC)
        )
        self._assert_conclusive(response)
        solver_status = _optimizer_status(response)
        builds = [
            self._build_summary(
                item,
                request,
                response,
                generated_at=canonical_generated_at,
                owned_product_ids=owned_product_ids,
            )
            for item in response.builds
        ]
        if not builds:
            status = BuildGenerationStatus.INFEASIBLE
        elif len(builds) < len(request.requested_profiles):
            status = BuildGenerationStatus.PARTIAL
        else:
            status = BuildGenerationStatus.COMPLETE
        infeasibility = None
        if response.infeasibility_reasons:
            reasons: list[InfeasibilityReason] = []
            suggestions: list[SuggestedRelaxation] = []
            for reason in response.infeasibility_reasons:
                if reason.startswith("Suggested relaxation:"):
                    suggestions.append(
                        SuggestedRelaxation(
                            field_path="request",
                            current_value=None,
                            proposed_value=reason.removeprefix("Suggested relaxation:").strip(),
                            expected_effect="May expand the feasible candidate set.",
                        )
                    )
                else:
                    reasons.append(InfeasibilityReason(code="no_feasible_build", message=reason))
            infeasibility = InfeasibilityExplanation(
                reasons=reasons
                or [
                    InfeasibilityReason(
                        code="no_feasible_build",
                        message="No complete build satisfied every hard constraint.",
                    )
                ],
                suggested_relaxations=suggestions,
            )
        return GenerateBuildsResponse(
            request_id=response.request_id,
            status=status,
            generated_at=canonical_generated_at,
            data_version=response.data_version,
            ranking_model=response.ranking_model,
            retrieval_model=response.retrieval_model,
            performance_model=response.performance_model,
            rule_version=response.rule_version,
            solver_version=_optimizer_version(response, self.services),
            solver_status=solver_status,
            solver_ran=response.optimizer_ran,
            solver_profile_statuses=[
                SolverProfileOutcome(
                    profile=BuildProfile(item.profile.value),
                    status=SolverStatus(item.status.value.upper()),
                    wall_time_seconds=item.wall_time_seconds,
                    objective_value=item.objective_value,
                )
                for item in response.optimizer_profile_statuses
            ],
            solver_validator_rejections=response.optimizer_validator_rejections,
            builds=builds,
            infeasibility=infeasibility,
        )

    @staticmethod
    def _assert_conclusive(response: ApplicationBuildGenerationResponse) -> None:
        status = _optimizer_status(response)
        if status in {SolverStatus.MODEL_INVALID, SolverStatus.UNKNOWN}:
            raise ApiError(
                status_code=503,
                code="optimizer_not_conclusive",
                message=f"The optimizer returned {status.value}; no build was published.",
                details={
                    "solver_status": status.value,
                    "solver_ran": response.optimizer_ran,
                    "validator_rejections": response.optimizer_validator_rejections,
                    "profile_statuses": [
                        {
                            "profile": item.profile.value,
                            "status": item.status.value.upper(),
                        }
                        for item in response.optimizer_profile_statuses
                    ],
                },
            )
        if status is SolverStatus.INFEASIBLE and response.builds:
            raise ApiError(
                status_code=503,
                code="optimizer_contract_violation",
                message="The optimizer reported INFEASIBLE while returning build solutions.",
            )
        if not response.optimizer_ran and status in {
            SolverStatus.OPTIMAL,
            SolverStatus.FEASIBLE,
        }:
            raise ApiError(
                status_code=503,
                code="optimizer_contract_violation",
                message="A successful solver status was reported without optimizer execution.",
            )

    def _build_summary(
        self,
        build: DomainBuild,
        request: DomainBuildRequest,
        response: ApplicationBuildGenerationResponse,
        *,
        generated_at: datetime | None = None,
        owned_product_ids: Collection[str] | None = None,
    ) -> BuildSummary:
        existing_ids = (
            set(owned_product_ids)
            if owned_product_ids is not None
            else {item.product_id for item in request.existing_products}
        )
        components = []
        for selected in build.components:
            item = self.services.catalog.require(selected.product_id)
            listing = item.listing
            signals: list[PerformanceSignal] = []
            for signal in selected.performance_signals:
                workload = signal.workload.value
                observations = (
                    item.workload_benchmarks.get(workload, ()) if signal.basis == "observed" else ()
                )
                signals.append(
                    PerformanceSignal(
                        workload=workload,
                        metric=signal.metric,
                        value=(signal.score if signal.score is not None else signal.relative_score),
                        unit=signal.unit,
                        basis=signal.basis,
                        confidence=(None if signal.confidence == "observed" else signal.confidence),
                        decision=signal.decision,
                        model_version=signal.model_version,
                        observed_at=(
                            max(observation.observed_at for observation in observations)
                            if observations
                            else None
                        ),
                        sources=[
                            SourceReference(
                                label=source,
                                url=source,
                            )
                            for source in signal.supporting_sources
                        ],
                        supporting_benchmark_ids=signal.supporting_benchmark_ids,
                        benchmark_evidence=[
                            PerformanceBenchmarkEvidence(
                                benchmark_id=observation.benchmark_id,
                                benchmark_name=observation.benchmark_name,
                                benchmark_version=observation.benchmark_version,
                                resolution=observation.resolution,
                                preset=observation.preset,
                                operating_system=observation.operating_system,
                                driver_version=observation.driver_version,
                                source_url=observation.source_url,
                                observed_at=observation.observed_at,
                            )
                            for observation in observations
                        ],
                    )
                )
            components.append(
                BuildComponent(
                    category=_api_category(selected.category),
                    product_id=selected.product_id,
                    listing_id=selected.listing_id,
                    canonical_name=selected.canonical_name,
                    brand=item.product.brand,
                    retailer=listing.retailer if listing else None,
                    listing_url=listing.listing_url if listing else None,
                    price_sgd=float(selected.price_sgd),
                    already_owned=selected.product_id in existing_ids,
                    component_score=selected.component_score,
                    selection_reasons=[selected.selection_reason],
                    performance_signals=signals,
                )
            )
        checks = [self._domain_check(item) for item in build.compatibility_checks]
        warning_checks = [item for item in checks if item.status is CompatibilityStatus.WARNING]
        selected_prices = {item.category: item.price_sgd for item in build.components}
        alternatives = [
            ReplacementCandidate(
                product_id=item.product_id,
                canonical_name=item.canonical_name,
                category=_api_category(item.category),
                price_sgd=max(
                    0.0,
                    float(selected_prices[item.category] + item.price_delta_sgd),
                ),
                performance_delta=item.performance_delta,
                price_delta_sgd=float(item.price_delta_sgd),
                compatibility_status=None,
                reasons=[item.explanation],
            )
            for item in build.alternatives
        ]
        return BuildSummary(
            request_id=response.request_id,
            build_id=build.build_id,
            profile=BuildProfile(build.profile.value),
            total_price_sgd=float(build.total_price_sgd),
            overall_score=build.overall_score,
            estimated_peak_power_w=build.estimated_power_watts,
            workload_scores={key.value: value for key, value in build.workload_scores.items()},
            compatibility_status=build.compatibility_status.value,
            components=components,
            compatibility_checks=checks,
            warnings=warning_checks,
            explanation=[
                ExplanationItem(kind="compatibility", text=text) for text in build.explanation
            ],
            alternatives=alternatives,
            generated_at=generated_at
            or self._generated_at_by_request.setdefault(response.request_id, datetime.now(UTC)),
            data_version=response.data_version,
            ranking_model=response.ranking_model,
            rule_version=response.rule_version,
            solver_version=_optimizer_version(response, self.services),
            solver_status=_optimizer_status(response),
            solver_ran=response.optimizer_ran,
        )

    def _domain_check(self, check: Any) -> CompatibilityCheck:
        categories = {
            _api_category(self.services.catalog.require(product_id).product.category)
            for product_id in check.component_ids
            if self.services.catalog.get(product_id) is not None
        }
        return CompatibilityCheck(
            rule_id=check.rule_id or "compatibility-rule",
            status=_api_status(check.status),
            message=check.message,
            affected_categories=sorted(categories, key=lambda item: item.value),
        )

    def _compatibility_check(self, result: Any) -> CompatibilityCheck:
        categories: set[ComponentCategory] = set()

        def visit(value: Any) -> None:
            if isinstance(value, Mapping):
                product_id = value.get("product_id")
                if product_id and self.services.catalog.get(str(product_id)) is not None:
                    categories.add(
                        _api_category(
                            self.services.catalog.require(str(product_id)).product.category
                        )
                    )
                for nested in value.values():
                    visit(nested)
            elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
                for nested in value:
                    visit(nested)

        visit(result.evidence)
        return CompatibilityCheck(
            rule_id=result.rule_id,
            status=_api_status(result.status),
            message=result.message,
            affected_categories=sorted(categories, key=lambda item: item.value),
        )

    @staticmethod
    def _aggregate_status(checks: Sequence[CompatibilityCheck]) -> CompatibilityStatus:
        statuses = {item.status for item in checks}
        if CompatibilityStatus.FAIL in statuses:
            return CompatibilityStatus.FAIL
        if CompatibilityStatus.UNKNOWN in statuses:
            return CompatibilityStatus.UNKNOWN
        if CompatibilityStatus.WARNING in statuses:
            return CompatibilityStatus.WARNING
        return CompatibilityStatus.PASS


def create_processed_catalog_service(
    settings: ApiSettings,
    *,
    performance_artifacts: Sequence[PerformanceModelArtifact] | None = None,
    promoted_serving_models: ActiveServingModels | None = None,
    ranker: ProductRanker | None = None,
    retriever: ProductRetriever | None = None,
) -> CoreRecommendationService:
    """Bootstrap the real service from explicitly configured processed artifacts."""

    assert settings.buildcores_catalog_path is not None
    assert settings.governed_offers_path is not None
    release = None
    durable_store: SqlAlchemyDurableStore | None = None
    if settings.uses_database_storage:
        if settings.database_url is None:
            raise RuntimeError("database storage was selected without a database URL")
        durable_store = SqlAlchemyDurableStore.from_url(settings.database_url.get_secret_value())
        durable_store.verify_schema()
    if settings.requires_durable_storage:
        if any(
            value is not None
            for value in (performance_artifacts, promoted_serving_models, ranker, retriever)
        ):
            raise RuntimeError(
                "non-development serving components must come from the immutable serving manifest"
            )
        if settings.serving_manifest_path is None or settings.serving_manifest_sha256 is None:
            raise RuntimeError(
                "non-development processed_catalog mode requires a pinned serving_manifest_path"
            )
        if settings.source_registry_path is None or settings.source_trust_root_sha256 is None:
            raise RuntimeError(
                "non-development processed_catalog mode requires independent source authority pins"
            )
        if (
            settings.semantic_encoder_bundle_path is None
            or settings.semantic_encoder_bundle_sha256 is None
        ):
            raise RuntimeError(
                "non-development processed_catalog mode requires a pinned "
                "semantic_encoder_bundle_path"
            )
        if settings.reviewed_mapping_path is None:
            raise RuntimeError(
                "non-development processed_catalog mode requires reviewed_mapping_path"
            )
        if settings.review_evidence_path is None:
            raise RuntimeError(
                "non-development processed_catalog mode requires review_evidence_path"
            )
        if durable_store is None:
            raise RuntimeError("production serving release requires durable database storage")
        release = load_production_serving_release(
            settings.serving_manifest_path,
            catalog_path=settings.buildcores_catalog_path,
            offers_path=settings.governed_offers_path,
            reviewed_mappings_path=settings.reviewed_mapping_path,
            review_evidence_path=settings.review_evidence_path,
            current_source_registry_path=settings.source_registry_path,
            expected_source_trust_root_sha256=settings.source_trust_root_sha256,
            session_factory=durable_store.session_factory,
            expected_catalog_data_version=settings.data_version,
            expected_ranker_version=settings.ranking_model_version,
            expected_manifest_sha256=settings.serving_manifest_sha256,
            expected_encoder_bundle_path=settings.semantic_encoder_bundle_path,
            expected_encoder_bundle_sha256=settings.semantic_encoder_bundle_sha256,
        )
        er_release = release.catalog_release.entity_resolution
        production_policy = production_catalog_policy_from_entity_resolution(er_release.policy)
        data = load_processed_catalog(
            settings.buildcores_catalog_path,
            offer_path=settings.governed_offers_path,
            reviewed_mapping_path=settings.reviewed_mapping_path,
            review_evidence_path=settings.review_evidence_path,
            entity_resolution_evaluation_path=(
                release.catalog_release.entity_resolution_evaluation_path
            ),
            entity_resolution_runtime=er_release.runtime,
            entity_resolution_policy=er_release.policy,
            entity_resolution_binding_sha256=er_release.identity.binding_sha256,
            require_production_entity_resolution=True,
            production_policy=production_policy,
        )
    else:
        data = load_processed_catalog(
            settings.buildcores_catalog_path,
            offer_path=settings.governed_offers_path,
            reviewed_mapping_path=settings.reviewed_mapping_path,
            review_evidence_path=settings.review_evidence_path,
            entity_resolution_evaluation_path=(settings.entity_resolution_evaluation_path),
        )
    if data.readiness is None:
        raise RuntimeError("processed catalogue has no measured production-readiness report")
    if not settings.allow_development_catalog:
        validate_production_readiness(data.readiness)
    reader = InMemoryCatalogReader(data)
    if settings.requires_durable_storage and data.stats.data_version != settings.data_version:
        raise ServingConfigurationError(
            "processed catalogue version does not match PCBR_API_DATA_VERSION"
        )
    if durable_store is not None:
        durable_store.verify_catalog_identity(
            product_ids=(product.product_id for product in data.products),
            listing_ids=(listing.listing_id for listing in data.listings),
            canonical_products=(data.products if settings.requires_durable_storage else None),
            retailer_listings=(data.listings if settings.requires_durable_storage else None),
        )
    if settings.requires_durable_storage:
        assert release is not None
        loaded_performance_artifacts = release.performance_artifacts
        promoted_serving_models = release.active_models
        ranker = release.ranker
        retriever = release.retriever
    else:
        loaded_performance_artifacts = (
            tuple(performance_artifacts)
            if performance_artifacts is not None
            else tuple(
                load_performance_artifact(path) for path in settings.performance_artifact_paths
            )
        )
    services = create_application_services(
        reader,
        data_version=data.stats.data_version,
        compatibility_evidence_policy=AUTHORITATIVE_COMPATIBILITY_POLICY,
        require_promoted_models=not settings.is_development_environment,
        promoted_serving_models=promoted_serving_models,
        ranker=ranker,
        retriever=retriever,
        performance_artifacts=loaded_performance_artifacts,
        allow_unpromoted_performance_models=(settings.allow_unpromoted_performance_models),
        result_store=durable_store,
    )
    return CoreRecommendationService(
        settings,
        services,
        reader,
        data,
        durable_store=durable_store,
        semantic_encoder_ready=(release.semantic_encoder_ready if release is not None else None),
        release_artifacts_verified=release is not None,
        source_authority_expires_at=(
            getattr(release.catalog_release.source_release, "authority_expires_at", None)
            if release is not None
            else None
        ),
    )
