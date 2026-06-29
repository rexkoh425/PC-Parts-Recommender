"""Application-service boundary and a deterministic vertical-slice implementation.

The HTTP adapter depends only on :class:`RecommendationApplication`.  The in-memory service
keeps local development and contract tests useful while the SQL repositories, learned ranker,
and CP-SAT orchestration are integrated.  Its version labels deliberately say ``baseline`` and
``demo`` so its relative scores cannot be mistaken for trained-model evidence.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, Protocol
from uuid import uuid4

from pc_build_recommender.compatibility import (
    CompatibilityEngine as CoreCompatibilityEngine,
)
from pc_build_recommender.pricing import PriceObservation as HistoricalPriceObservation
from services.api.errors import ApiError
from services.api.models import (
    AdminOperationsResponse,
    BenchmarkObservation,
    BuildComponent,
    BuildGenerationStatus,
    BuildProfile,
    BuildShareCreated,
    BuildShareRevoked,
    BuildSummary,
    CanonicalInteractionEvent,
    CatalogueCoverage,
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
    ReplacementCandidate,
    ReplacementRequest,
    ReplacementResponse,
    RevokeBuildShareRequest,
    SolverStatus,
    SuggestedRelaxation,
    encode_product_search_cursor,
    product_search_identity,
    resolve_product_search_page,
)
from services.api.pricing import summarize_price_history
from services.api.public_shares import public_build_snapshot
from services.api.settings import ApiSettings


class RecommendationApplication(Protocol):
    """Interface consumed by FastAPI routers; persistence and ML stay behind this boundary."""

    async def generate_builds(self, request: GenerateBuildsRequest) -> GenerateBuildsResponse: ...

    async def get_request_builds(self, request_id: str) -> GenerateBuildsResponse: ...

    async def get_build(self, build_id: str) -> BuildSummary: ...

    async def admin_operations(self) -> AdminOperationsResponse: ...

    async def create_build_share(self, build_id: str) -> BuildShareCreated: ...

    async def get_build_share(self, share_id: str) -> PublicBuildShare: ...

    async def revoke_build_share(
        self, share_id: str, request: RevokeBuildShareRequest
    ) -> BuildShareRevoked: ...

    async def replace_component(
        self, build_id: str, request: ReplacementRequest
    ) -> ReplacementResponse: ...

    async def search_products(self, request: ProductSearchRequest) -> ProductSearchResponse: ...

    async def get_product(self, product_id: str) -> ProductDetail: ...

    async def get_prices(self, product_id: str) -> ProductPricesResponse: ...

    async def get_benchmarks(self, product_id: str) -> ProductBenchmarksResponse: ...

    async def get_reviews(self, product_id: str) -> ProductReviewsResponse: ...

    async def check_compatibility(
        self, request: CompatibilityCheckRequest
    ) -> CompatibilityCheckResponse: ...

    async def record_interaction(
        self, event: CanonicalInteractionEvent
    ) -> InteractionAccepted: ...

    async def freshness(self) -> FreshnessResponse: ...

    async def ready(self) -> bool: ...

    async def readiness_checks(self) -> dict[str, Literal["ready", "not_ready"]]: ...

    async def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ProductRecord:
    product_id: str
    category: ComponentCategory
    canonical_name: str
    brand: str
    model: str
    price_cents: int
    performance: float
    power_w: int
    attributes: Mapping[str, Any]
    source_url: str = "https://example.invalid/controlled-demo-catalog"


def interaction_event_id(event: CanonicalInteractionEvent) -> str:
    """Return a stable retry identity without retaining the caller's raw key."""

    if event.idempotency_key_sha256 is None:
        return f"evt_{uuid4().hex}"
    canonical = (
        f"pcbr-interaction-event-v1\x00{event.session_id}\x00"
        f"{event.idempotency_key_sha256}"
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"evt_{digest}"


def same_canonical_interaction(
    left: CanonicalInteractionEvent, right: CanonicalInteractionEvent
) -> bool:
    return left.model_dump(mode="json") == right.model_dump(mode="json")


def same_impression_interaction(
    left: CanonicalInteractionEvent, right: CanonicalInteractionEvent
) -> bool:
    """Compare one displayed-result action independently of retry-key choice."""

    if (
        left.impression_id is None
        or left.impression_id != right.impression_id
        or left.event_type != right.event_type
    ):
        return False
    excluded = {"idempotency_key_sha256", "idempotency_payload_sha256"}
    return left.model_dump(mode="json", exclude=excluded) == right.model_dump(
        mode="json", exclude=excluded
    )


@dataclass(frozen=True, slots=True)
class Template:
    profile: BuildProfile
    product_ids: tuple[str, ...]
    overall_score: float
    value_score: float
    upgradeability_score: float
    efficiency_score: float


@dataclass(frozen=True, slots=True)
class _InMemoryBuildShare:
    snapshot: PublicBuildShare
    revocation_token_digest: str


def _product(
    product_id: str,
    category: ComponentCategory,
    canonical_name: str,
    brand: str,
    model: str,
    price_sgd: int,
    performance: float,
    power_w: int,
    **attributes: Any,
) -> ProductRecord:
    return ProductRecord(
        product_id=product_id,
        category=category,
        canonical_name=canonical_name,
        brand=brand,
        model=model,
        price_cents=price_sgd * 100,
        performance=performance,
        power_w=power_w,
        attributes=attributes,
    )


def demo_catalog() -> dict[str, ProductRecord]:
    """Return controlled contract-test data, not a current-market price claim."""

    products = [
        _product(
            "cpu-amd-7600",
            ComponentCategory.CPU,
            "AMD Ryzen 5 7600",
            "AMD",
            "Ryzen 5 7600",
            249,
            73,
            88,
            socket="AM5",
            generation="Ryzen 7000",
            supported_chipsets=["B650", "X670"],
            peak_power_w=88,
        ),
        _product(
            "cpu-amd-7700",
            ComponentCategory.CPU,
            "AMD Ryzen 7 7700",
            "AMD",
            "Ryzen 7 7700",
            389,
            84,
            115,
            socket="AM5",
            generation="Ryzen 7000",
            supported_chipsets=["B650", "X670"],
            peak_power_w=115,
        ),
        _product(
            "cpu-amd-7900",
            ComponentCategory.CPU,
            "AMD Ryzen 9 7900",
            "AMD",
            "Ryzen 9 7900",
            529,
            94,
            165,
            socket="AM5",
            generation="Ryzen 7000",
            supported_chipsets=["B650", "X670"],
            peak_power_w=165,
        ),
        _product(
            "gpu-rtx-5060ti-16",
            ComponentCategory.GPU,
            "NVIDIA GeForce RTX 5060 Ti 16 GB",
            "NVIDIA",
            "GeForce RTX 5060 Ti 16 GB",
            699,
            78,
            180,
            vram_gb=16,
            length_mm=247,
            slot_width=2.5,
            board_power_w=180,
            power_connectors={"pcie_8_pin": 1},
        ),
        _product(
            "gpu-rx-7800xt-16",
            ComponentCategory.GPU,
            "AMD Radeon RX 7800 XT 16 GB",
            "AMD",
            "Radeon RX 7800 XT 16 GB",
            749,
            82,
            263,
            vram_gb=16,
            length_mm=280,
            slot_width=2.5,
            board_power_w=263,
            power_connectors={"pcie_8_pin": 2},
        ),
        _product(
            "gpu-rtx-4070tis-16",
            ComponentCategory.GPU,
            "NVIDIA GeForce RTX 4070 Ti SUPER 16 GB",
            "NVIDIA",
            "GeForce RTX 4070 Ti SUPER 16 GB",
            1099,
            94,
            285,
            vram_gb=16,
            length_mm=305,
            slot_width=3.0,
            board_power_w=285,
            power_connectors={"12vhpwr": 1},
        ),
        _product(
            "gpu-oversized-test-fixture",
            ComponentCategory.GPU,
            "Oversized Compatibility Test GPU 16 GB",
            "FixtureGPU",
            "Oversized Test Fixture",
            650,
            70,
            180,
            vram_gb=16,
            length_mm=500,
            slot_width=5.0,
            board_power_w=180,
            power_connectors={"pcie_8_pin": 1},
        ),
        _product(
            "gpu-unverified-test-fixture",
            ComponentCategory.GPU,
            "Unverified-Dimensions Test GPU 16 GB",
            "FixtureGPU",
            "Unverified Test Fixture",
            640,
            68,
            180,
            vram_gb=16,
            board_power_w=180,
        ),
        _product(
            "mb-b650m-wifi",
            ComponentCategory.MOTHERBOARD,
            "B650M Wi-Fi DDR5 Motherboard",
            "ExampleBoard",
            "B650M Wi-Fi",
            219,
            72,
            45,
            socket="AM5",
            chipset="B650",
            supported_cpu_generations=["Ryzen 7000"],
            memory_type="DDR5",
            form_factor="microatx",
            wifi=True,
            maximum_memory_gb=192,
            memory_slots=4,
            pcie_x16_slots=1,
            storage_interfaces=["m2_nvme", "sata"],
            m2_slots=2,
            sata_ports=4,
        ),
        _product(
            "mb-b650-atx-wifi",
            ComponentCategory.MOTHERBOARD,
            "B650 ATX Wi-Fi DDR5 Motherboard",
            "ExampleBoard",
            "B650 ATX Wi-Fi",
            279,
            82,
            50,
            socket="AM5",
            chipset="B650",
            supported_cpu_generations=["Ryzen 7000"],
            memory_type="DDR5",
            form_factor="atx",
            wifi=True,
            maximum_memory_gb=192,
            memory_slots=4,
            pcie_x16_slots=2,
            storage_interfaces=["m2_nvme", "sata"],
            m2_slots=3,
            sata_ports=4,
        ),
        _product(
            "mb-x670-atx-wifi",
            ComponentCategory.MOTHERBOARD,
            "X670 ATX Wi-Fi DDR5 Motherboard",
            "ExampleBoard",
            "X670 ATX Wi-Fi",
            389,
            92,
            55,
            socket="AM5",
            chipset="X670",
            supported_cpu_generations=["Ryzen 7000"],
            memory_type="DDR5",
            form_factor="atx",
            wifi=True,
            maximum_memory_gb=192,
            memory_slots=4,
            pcie_x16_slots=2,
            storage_interfaces=["m2_nvme", "sata"],
            m2_slots=4,
            sata_ports=6,
        ),
        _product(
            "mem-ddr5-32-5600",
            ComponentCategory.MEMORY,
            "32 GB DDR5-5600 Memory Kit",
            "ExampleMemory",
            "32 GB DDR5-5600",
            129,
            74,
            8,
            memory_type="DDR5",
            capacity_gb=32,
            module_count=2,
        ),
        _product(
            "mem-ddr5-32-6000",
            ComponentCategory.MEMORY,
            "32 GB DDR5-6000 Low-Latency Memory Kit",
            "ExampleMemory",
            "32 GB DDR5-6000",
            149,
            82,
            9,
            memory_type="DDR5",
            capacity_gb=32,
            module_count=2,
        ),
        _product(
            "mem-ddr5-64-6000",
            ComponentCategory.MEMORY,
            "64 GB DDR5-6000 Memory Kit",
            "ExampleMemory",
            "64 GB DDR5-6000",
            269,
            94,
            12,
            memory_type="DDR5",
            capacity_gb=64,
            module_count=2,
        ),
        _product(
            "ssd-nvme-2tb-value",
            ComponentCategory.STORAGE,
            "2 TB PCIe 4.0 NVMe SSD",
            "ExampleStorage",
            "2 TB NVMe Value",
            139,
            76,
            6,
            capacity_gb=2000,
            interface="m2_nvme",
        ),
        _product(
            "ssd-nvme-2tb-fast",
            ComponentCategory.STORAGE,
            "2 TB High-Performance PCIe 4.0 NVMe SSD",
            "ExampleStorage",
            "2 TB NVMe Performance",
            169,
            91,
            7,
            capacity_gb=2000,
            interface="m2_nvme",
        ),
        _product(
            "psu-750-gold",
            ComponentCategory.PSU,
            "750 W 80 Plus Gold Modular PSU",
            "ExamplePower",
            "750 W Gold",
            159,
            81,
            0,
            wattage=750,
            form_factor="atx",
            pcie_connectors={"pcie_8_pin": 3, "12vhpwr": 1},
        ),
        _product(
            "psu-850-gold",
            ComponentCategory.PSU,
            "850 W 80 Plus Gold ATX 3.0 Modular PSU",
            "ExamplePower",
            "850 W Gold ATX 3.0",
            189,
            90,
            0,
            wattage=850,
            form_factor="atx",
            pcie_connectors={"pcie_8_pin": 4, "12vhpwr": 1},
        ),
        _product(
            "cooler-single-tower",
            ComponentCategory.COOLER,
            "120 mm Single-Tower CPU Cooler",
            "ExampleCooling",
            "Single Tower 120",
            59,
            73,
            4,
            supported_sockets=["AM5"],
            cooler_type="air",
            height_mm=154,
        ),
        _product(
            "cooler-dual-tower",
            ComponentCategory.COOLER,
            "120 mm Dual-Tower CPU Cooler",
            "ExampleCooling",
            "Dual Tower 120",
            79,
            88,
            6,
            supported_sockets=["AM5"],
            cooler_type="air",
            height_mm=157,
        ),
        _product(
            "case-matx-air",
            ComponentCategory.CASE,
            "Airflow Micro-ATX Mini Tower Case",
            "ExampleCase",
            "mATX Air",
            99,
            75,
            0,
            case_size="mini_tower",
            supported_motherboard_sizes=["microatx", "miniitx"],
            maximum_gpu_length_mm=330,
            maximum_gpu_slot_width=4.0,
            maximum_cooler_height_mm=165,
            supported_psu_sizes=["atx"],
        ),
        _product(
            "case-atx-air",
            ComponentCategory.CASE,
            "Airflow ATX Mid-Tower Case",
            "ExampleCase",
            "ATX Air",
            139,
            84,
            0,
            case_size="mid_tower",
            supported_motherboard_sizes=["atx", "microatx", "miniitx"],
            maximum_gpu_length_mm=380,
            maximum_gpu_slot_width=4.0,
            maximum_cooler_height_mm=175,
            supported_psu_sizes=["atx"],
        ),
        _product(
            "case-atx-quiet",
            ComponentCategory.CASE,
            "Dampened ATX Quiet Mid-Tower Case",
            "ExampleCase",
            "ATX Quiet",
            159,
            82,
            0,
            case_size="mid_tower",
            supported_motherboard_sizes=["atx", "microatx", "miniitx"],
            maximum_gpu_length_mm=360,
            maximum_gpu_slot_width=4.0,
            maximum_cooler_height_mm=170,
            supported_psu_sizes=["atx"],
        ),
    ]
    return {product.product_id: product for product in products}


def demo_templates() -> tuple[Template, ...]:
    return (
        Template(
            BuildProfile.BEST_OVERALL,
            (
                "cpu-amd-7700",
                "gpu-rtx-5060ti-16",
                "mb-b650-atx-wifi",
                "mem-ddr5-32-6000",
                "ssd-nvme-2tb-fast",
                "psu-750-gold",
                "cooler-dual-tower",
                "case-atx-quiet",
            ),
            88,
            86,
            88,
            84,
        ),
        Template(
            BuildProfile.BEST_VALUE,
            (
                "cpu-amd-7600",
                "gpu-rtx-5060ti-16",
                "mb-b650m-wifi",
                "mem-ddr5-32-5600",
                "ssd-nvme-2tb-value",
                "psu-750-gold",
                "cooler-single-tower",
                "case-atx-air",
            ),
            81,
            94,
            79,
            91,
        ),
        Template(
            BuildProfile.HIGHEST_PERFORMANCE,
            (
                "cpu-amd-7700",
                "gpu-rtx-4070tis-16",
                "mb-b650m-wifi",
                "mem-ddr5-32-6000",
                "ssd-nvme-2tb-value",
                "psu-850-gold",
                "cooler-dual-tower",
                "case-atx-air",
            ),
            94,
            82,
            83,
            80,
        ),
        Template(
            BuildProfile.MOST_UPGRADEABLE,
            (
                "cpu-amd-7600",
                "gpu-rx-7800xt-16",
                "mb-x670-atx-wifi",
                "mem-ddr5-32-6000",
                "ssd-nvme-2tb-fast",
                "psu-850-gold",
                "cooler-dual-tower",
                "case-atx-air",
            ),
            86,
            83,
            96,
            78,
        ),
        Template(
            BuildProfile.LOWEST_POWER,
            (
                "cpu-amd-7600",
                "gpu-rtx-5060ti-16",
                "mb-b650m-wifi",
                "mem-ddr5-32-6000",
                "ssd-nvme-2tb-fast",
                "psu-750-gold",
                "cooler-dual-tower",
                "case-atx-air",
            ),
            83,
            89,
            82,
            96,
        ),
    )


class InMemoryRecommendationService:
    """Refresh-safe, deterministic contract implementation for the first vertical slice."""

    def __init__(
        self,
        settings: ApiSettings,
        *,
        products: Mapping[str, ProductRecord] | None = None,
        templates: Sequence[Template] | None = None,
    ) -> None:
        self.settings = settings
        self.products = dict(products or demo_catalog())
        self.templates = tuple(templates or demo_templates())
        self.compatibility_engine = CoreCompatibilityEngine(
            rule_version=settings.compatibility_rule_version
        )
        self.catalog_updated_at = datetime.now(UTC)
        self._requests: dict[str, GenerateBuildsRequest] = {}
        self._responses: dict[str, GenerateBuildsResponse] = {}
        self._builds: dict[str, BuildSummary] = {}
        self._build_request_ids: dict[str, str] = {}
        self._build_shares: dict[str, _InMemoryBuildShare] = {}
        self._interactions: list[tuple[str, CanonicalInteractionEvent, datetime]] = []
        self._lock = asyncio.Lock()

    async def ready(self) -> bool:
        return bool(self.products and self.templates)

    async def close(self) -> None:
        return None

    async def readiness_checks(self) -> dict[str, Literal["ready", "not_ready"]]:
        ready = await self.ready()
        return {
            "catalogue": "ready" if self.products else "not_ready",
            "build_profiles": "ready" if self.templates else "not_ready",
            "compatibility_engine": "ready" if ready else "not_ready",
        }

    async def freshness(self) -> FreshnessResponse:
        age_hours = (datetime.now(UTC) - self.catalog_updated_at).total_seconds() / 3600
        catalogue_status: Literal["fresh", "stale"] = (
            "fresh"
            if age_hours <= self.settings.catalogue_stale_after_hours
            else "stale"
        )
        price_status: Literal["fresh", "stale"] = (
            "fresh" if age_hours <= self.settings.price_stale_after_hours else "stale"
        )
        status: Literal["fresh", "stale"] = (
            "fresh" if catalogue_status == price_status == "fresh" else "stale"
        )
        return FreshnessResponse(
            data_version=self.settings.data_version,
            status=status,
            catalogue_status=catalogue_status,
            price_status=price_status,
            last_catalog_update=self.catalog_updated_at,
            prices_updated_at=self.catalog_updated_at,
            stale_after_hours=self.settings.price_stale_after_hours,
            catalogue_stale_after_hours=self.settings.catalogue_stale_after_hours,
            price_stale_after_hours=self.settings.price_stale_after_hours,
            source_count=1,
            product_count=len(self.products),
            listing_count=len(self.products),
            production_ready=False,
            release_artifact_verification="development_unverified",
            readiness_blockers=[
                "The controlled demo catalogue is not eligible for production traffic."
            ],
        )

    async def generate_builds(self, request: GenerateBuildsRequest) -> GenerateBuildsResponse:
        generated_at = datetime.now(UTC)
        request_id = f"req_{uuid4().hex}"
        existing_by_category: dict[ComponentCategory, ProductRecord] = {}
        include_owned_in_budget: dict[ComponentCategory, bool] = {}
        invalid_existing: list[str] = []
        for existing in request.existing_products:
            product = self.products.get(existing.product_id)
            if product is None or product.category is not existing.category:
                invalid_existing.append(existing.product_id)
                continue
            existing_by_category[existing.category] = product
            include_owned_in_budget[existing.category] = existing.include_in_budget

        if invalid_existing:
            response = self._infeasible(
                request_id,
                generated_at,
                reasons=[
                    InfeasibilityReason(
                        code="unknown_existing_product",
                        message=(
                            "Retained products need verified catalogue specifications before "
                            f"compatibility can be established: {', '.join(invalid_existing)}."
                        ),
                    )
                ],
            )
            await self._store_generation(request, response)
            return response.model_copy(deep=True)

        feasible_builds: list[BuildSummary] = []
        rejection_codes: set[str] = set()
        seen_configurations: set[tuple[str, ...]] = set()
        profiles = request.requested_profiles or [
            template.profile for template in self.templates[: request.max_builds]
        ]
        templates_by_profile = {template.profile: template for template in self.templates}
        requested_templates = [templates_by_profile[profile] for profile in profiles]
        for template in requested_templates:
            selected = {
                self.products[item].category: self.products[item] for item in template.product_ids
            }
            selected.update(existing_by_category)
            configuration = tuple(selected[category].product_id for category in ComponentCategory)
            if configuration in seen_configurations:
                continue
            seen_configurations.add(configuration)

            codes = self._requirement_rejections(selected, request)
            if codes:
                rejection_codes.update(codes)
                continue
            checks = self._evaluate_compatibility(selected)
            if any(
                check.status in {CompatibilityStatus.FAIL, CompatibilityStatus.UNKNOWN}
                for check in checks
            ):
                rejection_codes.add("compatibility")
                continue

            total_cents = sum(
                product.price_cents
                for category, product in selected.items()
                if category not in existing_by_category or include_owned_in_budget[category]
            )
            if total_cents > round(request.budget_sgd * 100):
                rejection_codes.add("budget")
                continue
            feasible_builds.append(
                self._build_summary(
                    template=template,
                    selected=selected,
                    request=request,
                    request_id=request_id,
                    generated_at=generated_at,
                    total_cents=total_cents,
                    already_owned=set(existing_by_category),
                    checks=checks,
                )
            )
            if len(feasible_builds) >= len(requested_templates):
                break

        if not feasible_builds:
            response = self._infeasible(
                request_id,
                generated_at,
                reasons=self._infeasibility_reasons(request, rejection_codes),
            )
        else:
            response = GenerateBuildsResponse(
                request_id=request_id,
                status=(
                    BuildGenerationStatus.COMPLETE
                    if len(feasible_builds) == len(requested_templates)
                    else BuildGenerationStatus.PARTIAL
                ),
                generated_at=generated_at,
                data_version=self.settings.data_version,
                ranking_model=self.settings.ranking_model_version,
                retrieval_model="deterministic-token-lexical-baseline-v1",
                performance_model="deterministic-relative-performance-baseline-v1",
                rule_version=self.settings.compatibility_rule_version,
                solver_version=self.settings.solver_version,
                solver_status=SolverStatus.FEASIBLE,
                solver_ran=False,
                builds=feasible_builds,
            )
        await self._store_generation(request, response)
        return response.model_copy(deep=True)

    async def _store_generation(
        self, request: GenerateBuildsRequest, response: GenerateBuildsResponse
    ) -> None:
        async with self._lock:
            self._requests[response.request_id] = request.model_copy(deep=True)
            self._responses[response.request_id] = response.model_copy(deep=True)
            for build in response.builds:
                self._builds[build.build_id] = build.model_copy(deep=True)
                self._build_request_ids[build.build_id] = response.request_id

    def _infeasible(
        self,
        request_id: str,
        generated_at: datetime,
        *,
        reasons: list[InfeasibilityReason],
    ) -> GenerateBuildsResponse:
        return GenerateBuildsResponse(
            request_id=request_id,
            status=BuildGenerationStatus.INFEASIBLE,
            generated_at=generated_at,
            data_version=self.settings.data_version,
            ranking_model=self.settings.ranking_model_version,
            retrieval_model="deterministic-token-lexical-baseline-v1",
            performance_model="deterministic-relative-performance-baseline-v1",
            rule_version=self.settings.compatibility_rule_version,
            solver_version=self.settings.solver_version,
            solver_status=SolverStatus.INFEASIBLE,
            solver_ran=False,
            builds=[],
            infeasibility=InfeasibilityExplanation(
                reasons=reasons,
                suggested_relaxations=[
                    SuggestedRelaxation(
                        field_path="budget_sgd",
                        current_value=None,
                        proposed_value="increase_by_10_percent",
                        expected_effect="Expands the feasible component combinations.",
                    )
                ],
            ),
        )

    def _infeasibility_reasons(
        self, request: GenerateBuildsRequest, codes: set[str]
    ) -> list[InfeasibilityReason]:
        reasons: list[InfeasibilityReason] = []
        if "budget" in codes:
            reasons.append(
                InfeasibilityReason(
                    code="budget_too_low",
                    message=(
                        f"No verified complete build fits the S${request.budget_sgd:,.2f} budget."
                    ),
                )
            )
        messages: dict[str, tuple[str, list[ComponentCategory]]] = {
            "minimum_gpu_vram": (
                "No candidate GPU satisfies the requested VRAM minimum.",
                [ComponentCategory.GPU],
            ),
            "minimum_memory": (
                "No candidate memory kit satisfies the requested capacity.",
                [ComponentCategory.MEMORY],
            ),
            "storage_capacity": (
                "No candidate storage product satisfies the requested capacity.",
                [ComponentCategory.STORAGE],
            ),
            "wifi": (
                "No feasible motherboard provides required Wi-Fi support.",
                [ComponentCategory.MOTHERBOARD],
            ),
            "case_size": (
                "No feasible case matches the requested size.",
                [ComponentCategory.CASE],
            ),
            "excluded_brand": (
                "Brand exclusions remove all otherwise feasible combinations.",
                [],
            ),
            "compatibility": (
                "All retrieved combinations had a hard incompatibility or unknown hard field.",
                [],
            ),
        }
        for code in sorted(codes - {"budget"}):
            message, categories = messages.get(
                code, ("No build satisfies all hard requirements.", [])
            )
            reasons.append(
                InfeasibilityReason(code=code, message=message, affected_categories=categories)
            )
        if not reasons:
            reasons.append(
                InfeasibilityReason(
                    code="no_feasible_build",
                    message="No build satisfies all hard requirements in the current catalogue.",
                )
            )
        return reasons

    def _requirement_rejections(
        self,
        selected: Mapping[ComponentCategory, ProductRecord],
        request: GenerateBuildsRequest,
    ) -> set[str]:
        codes: set[str] = set()
        requirements = request.requirements
        gpu = selected[ComponentCategory.GPU]
        memory = selected[ComponentCategory.MEMORY]
        storage = selected[ComponentCategory.STORAGE]
        motherboard = selected[ComponentCategory.MOTHERBOARD]
        case = selected[ComponentCategory.CASE]
        if (
            requirements.minimum_gpu_vram_gb is not None
            and int(gpu.attributes.get("vram_gb", -1)) < requirements.minimum_gpu_vram_gb
        ):
            codes.add("minimum_gpu_vram")
        if (
            requirements.minimum_memory_gb is not None
            and int(memory.attributes.get("capacity_gb", -1)) < requirements.minimum_memory_gb
        ):
            codes.add("minimum_memory")
        if (
            requirements.storage_gb is not None
            and int(storage.attributes.get("capacity_gb", -1)) < requirements.storage_gb
        ):
            codes.add("storage_capacity")
        if requirements.wifi_required and not motherboard.attributes.get("wifi", False):
            codes.add("wifi")
        if (
            requirements.case_size is not None
            and case.attributes.get("case_size") != requirements.case_size
        ):
            codes.add("case_size")
        excluded = {brand.casefold() for brand in request.preferences.excluded_brands}
        if any(product.brand.casefold() in excluded for product in selected.values()):
            codes.add("excluded_brand")
        return codes

    def _evaluate_compatibility(
        self, selected: Mapping[ComponentCategory, ProductRecord]
    ) -> list[CompatibilityCheck]:
        component_payload = {
            self._core_category(category): self._compatibility_payload(product)
            for category, product in selected.items()
        }
        report = self.compatibility_engine.check_build(component_payload)
        category_aliases = {
            "cpu": ComponentCategory.CPU,
            "gpu": ComponentCategory.GPU,
            "motherboard": ComponentCategory.MOTHERBOARD,
            "memory": ComponentCategory.MEMORY,
            "storage": ComponentCategory.STORAGE,
            "power_supply": ComponentCategory.PSU,
            "psu": ComponentCategory.PSU,
            "cooler": ComponentCategory.COOLER,
            "case": ComponentCategory.CASE,
        }
        checks: list[CompatibilityCheck] = []
        for result in report.results:
            raw_categories: list[str] = []
            component_evidence = result.evidence.get("components")
            if isinstance(component_evidence, Mapping):
                raw_categories.extend(str(key) for key in component_evidence)
            evidence_category = result.evidence.get("category")
            if evidence_category is not None:
                raw_categories.append(str(evidence_category))
            affected = list(
                dict.fromkeys(
                    category_aliases[raw] for raw in raw_categories if raw in category_aliases
                )
            )
            checks.append(
                CompatibilityCheck(
                    rule_id=result.rule_id,
                    status=CompatibilityStatus(result.status.value.casefold()),
                    message=result.message,
                    affected_categories=affected,
                )
            )
        return checks

    @staticmethod
    def _core_category(category: ComponentCategory) -> str:
        """Map the public API name to the core engine's canonical category key."""

        if category is ComponentCategory.PSU:
            return "power_supply"
        return category.value

    @staticmethod
    def _compatibility_payload(product: ProductRecord) -> dict[str, Any]:
        return {
            "product_id": product.product_id,
            "category": product.category.value,
            "canonical_name": product.canonical_name,
            "brand": product.brand,
            "model": product.model,
            "source_url": product.source_url,
            "category_attributes": dict(product.attributes),
        }

    def _build_summary(
        self,
        *,
        template: Template,
        selected: Mapping[ComponentCategory, ProductRecord],
        request: GenerateBuildsRequest,
        request_id: str,
        generated_at: datetime,
        total_cents: int,
        already_owned: set[ComponentCategory],
        checks: list[CompatibilityCheck],
    ) -> BuildSummary:
        components: list[BuildComponent] = []
        preferred = {brand.casefold() for brand in request.preferences.preferred_brands}
        for category in ComponentCategory:
            product = selected[category]
            reasons = [f"Meets the hard {category.value} requirements."]
            if product.brand.casefold() in preferred:
                reasons.append("Matches a preferred brand.")
            components.append(
                BuildComponent(
                    category=category,
                    product_id=product.product_id,
                    listing_id=f"listing-{product.product_id}",
                    canonical_name=product.canonical_name,
                    brand=product.brand,
                    retailer="Controlled demo listing",
                    listing_url=product.source_url,
                    price_sgd=round(product.price_cents / 100, 2),
                    already_owned=category in already_owned,
                    component_score=product.performance,
                    selection_reasons=reasons,
                    performance_signals=[
                        PerformanceSignal(
                            workload=workload.name.value,
                            metric="deterministic_relative_component_score",
                            value=round(product.performance, 2),
                            unit="relative index",
                            basis="relative",
                            confidence="low",
                            decision="deterministic_baseline",
                            model_version=self.settings.ranking_model_version,
                        )
                        for workload in request.workloads
                    ],
                    alternatives=self._alternatives(product, selected),
                )
            )
        cpu_score = selected[ComponentCategory.CPU].performance
        gpu_score = selected[ComponentCategory.GPU].performance
        storage_score = selected[ComponentCategory.STORAGE].performance
        workload_scores: dict[str, float] = {}
        for workload in request.workloads:
            if workload.name.value.startswith("gaming"):
                score = 0.8 * gpu_score + 0.2 * cpu_score
            elif workload.name.value == "local_ai":
                score = 0.9 * gpu_score + 0.1 * cpu_score
            elif workload.name.value == "software_development":
                score = 0.75 * cpu_score + 0.25 * storage_score
            else:
                score = 0.55 * gpu_score + 0.45 * cpu_score
            workload_scores[workload.name.value] = round(min(score, 100), 2)
        warnings = [check for check in checks if check.status is CompatibilityStatus.WARNING]
        return BuildSummary(
            request_id=request_id,
            build_id=f"build_{uuid4().hex}",
            profile=template.profile,
            total_price_sgd=round(total_cents / 100, 2),
            overall_score=template.overall_score,
            value_score=template.value_score,
            upgradeability_score=template.upgradeability_score,
            efficiency_score=template.efficiency_score,
            estimated_peak_power_w=(
                selected[ComponentCategory.CPU].power_w
                + selected[ComponentCategory.GPU].power_w
                + 100
            ),
            workload_scores=workload_scores,
            compatibility_status="warning" if warnings else "pass",
            components=components,
            compatibility_checks=checks,
            warnings=warnings,
            explanation=[
                ExplanationItem(
                    kind="compatibility",
                    text=(
                        "Every known hard compatibility rule passed an independent baseline check."
                    ),
                ),
                ExplanationItem(
                    kind="performance",
                    text=(
                        "Performance values are deterministic relative indices until a promoted "
                        "model and observed benchmark corpus are available."
                    ),
                ),
                ExplanationItem(
                    kind="price",
                    text=(
                        "Prices come from a controlled demo catalogue and are not "
                        "live-market quotes."
                    ),
                ),
            ],
            generated_at=generated_at,
            data_version=self.settings.data_version,
            ranking_model=self.settings.ranking_model_version,
            rule_version=self.settings.compatibility_rule_version,
            solver_version=self.settings.solver_version,
            solver_status=SolverStatus.FEASIBLE,
            solver_ran=False,
        )

    def _alternatives(
        self,
        product: ProductRecord,
        selected: Mapping[ComponentCategory, ProductRecord] | None = None,
    ) -> list[ReplacementCandidate]:
        alternatives = sorted(
            (
                candidate
                for candidate in self.products.values()
                if candidate.category is product.category
                and candidate.product_id != product.product_id
            ),
            key=lambda candidate: abs(candidate.price_cents - product.price_cents),
        )
        evaluated: list[ReplacementCandidate] = []
        for candidate in alternatives:
            status: str | None = None
            if selected is not None:
                substituted = dict(selected)
                substituted[product.category] = candidate
                checks = self._evaluate_compatibility(substituted)
                if any(
                    check.status in {CompatibilityStatus.FAIL, CompatibilityStatus.UNKNOWN}
                    for check in checks
                ):
                    continue
                status = (
                    "warning"
                    if any(check.status is CompatibilityStatus.WARNING for check in checks)
                    else "pass"
                )
            evaluated.append(
                ReplacementCandidate(
                    product_id=candidate.product_id,
                    canonical_name=candidate.canonical_name,
                    category=candidate.category,
                    price_sgd=candidate.price_cents / 100,
                    retailer="Controlled demo listing",
                    performance_delta=round(candidate.performance - product.performance, 2),
                    price_delta_sgd=round((candidate.price_cents - product.price_cents) / 100, 2),
                    power_delta_w=float(candidate.power_w - product.power_w),
                    compatibility_status=status,
                    reasons=(
                        ["Passed a full-build compatibility substitution check."]
                        if status is not None
                        else ["Requires a full compatibility recheck before replacement."]
                    ),
                )
            )
            if len(evaluated) >= 2:
                break
        return evaluated

    async def get_request_builds(self, request_id: str) -> GenerateBuildsResponse:
        response = self._responses.get(request_id)
        if response is None:
            raise ApiError(
                status_code=404,
                code="request_not_found",
                message=f"No generated-build request exists with ID '{request_id}'.",
            )
        return response.model_copy(deep=True)

    async def get_build(self, build_id: str) -> BuildSummary:
        build = self._builds.get(build_id)
        if build is None:
            raise ApiError(
                status_code=404,
                code="build_not_found",
                message=f"No generated build exists with ID '{build_id}'.",
            )
        return build.model_copy(deep=True)

    async def admin_operations(self) -> AdminOperationsResponse:
        return AdminOperationsResponse(
            data_version=self.settings.data_version,
            generated_at=datetime.now(UTC),
            mode="demo",
            pipeline_operations=None,
            pipeline_failure_events_available=False,
            notes=[
                "Controlled demo mode has no ingestion pipeline or retailer-price operations data.",
                "Configure processed_catalog mode and an administrator token for live operations "
                "counters.",
            ],
        )

    async def create_build_share(self, build_id: str) -> BuildShareCreated:
        build = await self.get_build(build_id)
        created_at = datetime.now(UTC)
        expires_at = created_at + timedelta(hours=self.settings.build_share_ttl_hours)
        share_id = f"share_{uuid4().hex}"
        revocation_token = secrets.token_urlsafe(32)
        public_share = PublicBuildShare(
            share_id=share_id,
            created_at=created_at,
            expires_at=expires_at,
            snapshot=public_build_snapshot(build),
        )
        async with self._lock:
            self._build_shares[share_id] = _InMemoryBuildShare(
                snapshot=public_share,
                revocation_token_digest=hashlib.sha256(revocation_token.encode("utf-8")).hexdigest(),
            )
        return BuildShareCreated(
            share_id=share_id,
            revocation_token=revocation_token,
            created_at=created_at,
            expires_at=expires_at,
        )

    async def get_build_share(self, share_id: str) -> PublicBuildShare:
        async with self._lock:
            share = self._build_shares.get(share_id)
            if share is not None and share.snapshot.expires_at > datetime.now(UTC):
                return share.snapshot.model_copy(deep=True)
        raise ApiError(
            status_code=404,
            code="build_share_not_found",
            message="No active public build share exists with this ID.",
        )

    async def revoke_build_share(
        self, share_id: str, request: RevokeBuildShareRequest
    ) -> BuildShareRevoked:
        digest = hashlib.sha256(request.revocation_token.encode("utf-8")).hexdigest()
        async with self._lock:
            share = self._build_shares.get(share_id)
            if (
                share is None
                or share.snapshot.expires_at <= datetime.now(UTC)
                or not hmac.compare_digest(share.revocation_token_digest, digest)
            ):
                raise ApiError(
                    status_code=404,
                    code="build_share_not_found",
                    message="No active public build share exists with this ID.",
                )
            del self._build_shares[share_id]
        return BuildShareRevoked(share_id=share_id, revoked_at=datetime.now(UTC))

    async def replace_component(
        self, build_id: str, request: ReplacementRequest
    ) -> ReplacementResponse:
        if request.mode == "reoptimize_unlocked":
            raise ApiError(
                status_code=422,
                code="replacement_mode_not_available",
                message=(
                    "Unlocked re-optimisation is not available until the core optimiser "
                    "adapter is enabled. Use 'lock_other_components' for this runtime."
                ),
            )
        original = await self.get_build(build_id)
        replacement = self.products.get(request.replacement_product_id)
        if replacement is None:
            raise ApiError(
                status_code=404,
                code="product_not_found",
                message=f"No product exists with ID '{request.replacement_product_id}'.",
            )
        if replacement.category is not request.category:
            raise ApiError(
                status_code=422,
                code="category_mismatch",
                message="The replacement product does not belong to the requested category.",
            )
        selected = {
            component.category: self.products[component.product_id]
            for component in original.components
        }
        prior = selected[request.category]
        prior_component = next(
            component for component in original.components if component.category is request.category
        )
        if prior.product_id == replacement.product_id:
            raise ApiError(
                status_code=422,
                code="replacement_is_current_product",
                message="Choose a different product for the replacement.",
            )
        selected[request.category] = replacement
        checks = self._evaluate_compatibility(selected)
        blocking = [
            check
            for check in checks
            if check.status in {CompatibilityStatus.FAIL, CompatibilityStatus.UNKNOWN}
        ]
        if blocking:
            raise ApiError(
                status_code=409,
                code="incompatible_replacement",
                message="The replacement was rejected by one or more hard compatibility rules.",
                details={"checks": [check.model_dump(mode="json") for check in blocking]},
            )
        request_id = self._build_request_ids.get(build_id)
        generation_request = self._requests.get(request_id or "")
        prior_counted_in_budget = True
        if prior_component.already_owned and generation_request is not None:
            retained = next(
                (
                    product
                    for product in generation_request.existing_products
                    if product.category is request.category
                ),
                None,
            )
            prior_counted_in_budget = retained.include_in_budget if retained is not None else False
        price_delta_cents = replacement.price_cents - (
            prior.price_cents if prior_counted_in_budget else 0
        )
        if generation_request and round(original.total_price_sgd * 100) + price_delta_cents > round(
            generation_request.budget_sgd * 100
        ):
            raise ApiError(
                status_code=409,
                code="replacement_exceeds_budget",
                message="The compatible replacement would exceed the original request budget.",
            )
        components: list[BuildComponent] = []
        for component in original.components:
            if component.category is not request.category:
                components.append(component.model_copy(deep=True))
                continue
            components.append(
                component.model_copy(
                    update={
                        "product_id": replacement.product_id,
                        "listing_id": f"listing-{replacement.product_id}",
                        "canonical_name": replacement.canonical_name,
                        "brand": replacement.brand,
                        "price_sgd": replacement.price_cents / 100,
                        "already_owned": False,
                        "component_score": replacement.performance,
                        "alternatives": self._alternatives(replacement, selected),
                        "selection_reasons": [
                            "Selected by the user and independently rechecked for compatibility."
                        ],
                    },
                    deep=True,
                )
            )
        workload_delta = round((replacement.performance - prior.performance) * 0.5, 2)
        new_workload_scores = {
            name: (None if value is None else round(max(0, min(100, value + workload_delta)), 2))
            for name, value in original.workload_scores.items()
        }
        new_build = original.model_copy(
            update={
                "build_id": f"build_{uuid4().hex}",
                "total_price_sgd": round(original.total_price_sgd + price_delta_cents / 100, 2),
                "overall_score": round(
                    max(0, min(100, original.overall_score + workload_delta * 0.4)), 2
                ),
                "workload_scores": new_workload_scores,
                "components": components,
                "compatibility_checks": checks,
                "warnings": [
                    check for check in checks if check.status is CompatibilityStatus.WARNING
                ],
                "generated_at": datetime.now(UTC),
                "solver_status": SolverStatus.FEASIBLE,
                "solver_ran": False,
            },
            deep=True,
        )
        # Revalidate after model_copy(update=...) because Pydantic deliberately skips it.
        new_build = BuildSummary.model_validate(new_build.model_dump())
        async with self._lock:
            self._builds[new_build.build_id] = new_build.model_copy(deep=True)
            if request_id:
                self._build_request_ids[new_build.build_id] = request_id
        return ReplacementResponse(
            build=new_build,
            changed_categories=[request.category],
            price_delta_sgd=round(price_delta_cents / 100, 2),
            workload_score_deltas={name: workload_delta for name in original.workload_scores},
            new_warnings=new_build.warnings,
            data_version=self.settings.data_version,
            ranking_model=self.settings.ranking_model_version,
            rule_version=self.settings.compatibility_rule_version,
            solver_version=self.settings.solver_version,
            solver_status=SolverStatus.FEASIBLE,
            solver_ran=False,
        )

    async def search_products(self, request: ProductSearchRequest) -> ProductSearchResponse:
        try:
            requested_page = resolve_product_search_page(request)
        except InvalidProductSearchCursor as error:
            raise ApiError(
                status_code=422,
                code="invalid_pagination_cursor",
                message=str(error),
            ) from error
        compatible_build: BuildSummary | None = None
        if (
            request.compatible_with_build_id
            and request.compatible_with_build_id not in self._builds
        ):
            raise ApiError(
                status_code=404,
                code="build_not_found",
                message=(
                    f"No generated build exists with ID '{request.compatible_with_build_id}'."
                ),
            )
        if request.compatible_with_build_id:
            compatible_build = self._builds[request.compatible_with_build_id]
        selected = (
            {
                component.category: self.products[component.product_id]
                for component in compatible_build.components
            }
            if compatible_build is not None
            else None
        )
        query_tokens = {token for token in request.query.casefold().split() if token}
        query_matches: list[tuple[int, ProductRecord]] = []
        for product in self.products.values():
            searchable = f"{product.canonical_name} {product.brand} {product.model}".casefold()
            matched = sum(token in searchable for token in query_tokens)
            if not query_tokens or matched > 0:
                query_matches.append((matched, product))

        category_counts = {
            category: sum(product.category is category for _, product in query_matches)
            for category in ComponentCategory
        }
        category_scoped = [
            (matched, product)
            for matched, product in query_matches
            if request.category is None or product.category is request.category
        ]
        filtered_category = len(query_matches) - len(category_scoped)
        brand_counts: dict[str, int] = {}
        for _, product in category_scoped:
            brand_counts[product.brand] = brand_counts.get(product.brand, 0) + 1

        brand_scoped = [
            (matched, product)
            for matched, product in category_scoped
            if request.brand is None or product.brand.casefold() == request.brand.casefold()
        ]
        filtered_brand = len(category_scoped) - len(brand_scoped)

        matching: list[tuple[int, ProductRecord, CompatibilityStatus | None]] = []
        filtered_incompatible = 0
        filtered_unknown = 0
        for matched, product in brand_scoped:
            compatibility_status: CompatibilityStatus | None = None
            if selected is not None:
                current = selected.get(product.category)
                if current is not None and current.product_id == product.product_id:
                    continue
                substituted = dict(selected)
                substituted[product.category] = product
                checks = self._evaluate_compatibility(substituted)
                if any(check.status is CompatibilityStatus.FAIL for check in checks):
                    filtered_incompatible += 1
                    continue
                if any(check.status is CompatibilityStatus.UNKNOWN for check in checks):
                    filtered_unknown += 1
                    continue
                compatibility_status = (
                    CompatibilityStatus.WARNING
                    if any(check.status is CompatibilityStatus.WARNING for check in checks)
                    else CompatibilityStatus.PASS
                )
            matching.append((matched, product, compatibility_status))
        matching.sort(
            key=lambda item: (
                -item[0],
                item[1].price_cents,
                item[1].canonical_name.casefold(),
                item[1].product_id,
            )
        )
        total = len(matching)
        page_size = request.effective_page_size
        total_pages = max(1, (total + page_size - 1) // page_size)
        page = min(requested_page, total_pages)
        offset = (page - 1) * page_size
        products = [
            ProductSearchItem(
                product_id=product.product_id,
                category=product.category,
                canonical_name=product.canonical_name,
                brand=product.brand,
                model=product.model,
                lowest_price_sgd=product.price_cents / 100,
                stock_status="in_stock",
                compatibility_status=compatibility_status,
            )
            for _, product, compatibility_status in matching[offset : offset + page_size]
        ]
        query_id, _ = product_search_identity(
            request,
            data_version=self.settings.data_version,
            retrieval_model="deterministic-token-baseline-v1",
            rule_version=self.settings.compatibility_rule_version,
        )
        return ProductSearchResponse(
            query_id=query_id,
            products=products,
            total=total,
            retrieved_candidates=len(query_matches),
            filtered_category=filtered_category,
            filtered_brand=filtered_brand,
            filtered_incompatible=filtered_incompatible,
            filtered_unknown=filtered_unknown,
            data_version=self.settings.data_version,
            retrieval_model="deterministic-token-baseline-v1",
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
                canonical_products=len(self.products),
                retailer_listings=len(self.products),
                source_count=1,
                category_count=len({product.category for product in self.products.values()}),
                as_of=self.catalog_updated_at,
                scope_label="Controlled illustrative API demo",
            ),
        )

    def _require_product(self, product_id: str) -> ProductRecord:
        product = self.products.get(product_id)
        if product is None:
            raise ApiError(
                status_code=404,
                code="product_not_found",
                message=f"No product exists with ID '{product_id}'.",
            )
        return product

    async def get_product(self, product_id: str) -> ProductDetail:
        product = self._require_product(product_id)
        return ProductDetail(
            product_id=product.product_id,
            category=product.category,
            canonical_name=product.canonical_name,
            brand=product.brand,
            model=product.model,
            lowest_price_sgd=product.price_cents / 100,
            stock_status="in_stock",
            manufacturer_part_number=None,
            attributes=dict(product.attributes),
            source_confidence=1.0,
            source_url=product.source_url,
            updated_at=self.catalog_updated_at,
            data_version=self.settings.data_version,
        )

    async def get_prices(self, product_id: str) -> ProductPricesResponse:
        product = self._require_product(product_id)
        price = product.price_cents / 100
        listing_id = f"listing-{product_id}"
        history = HistoricalPriceObservation(
            listing_id=listing_id,
            observed_at=self.catalog_updated_at,
            base_price=price,
            shipping_price=0,
            stock_status="in_stock",
            retailer="Controlled demo listing",
            currency="SGD",
            source_url=product.source_url,
        )
        return ProductPricesResponse(
            product_id=product_id,
            current_lowest_price_sgd=price,
            observations=[
                PriceObservation(
                    listing_id=listing_id,
                    retailer="Controlled demo listing",
                    observed_at=self.catalog_updated_at,
                    base_price_sgd=price,
                    shipping_price_sgd=0,
                    stock_status="in_stock",
                    condition="new",
                    current_offer_eligible=True,
                    listing_url=product.source_url,
                )
            ],
            price_intelligence=summarize_price_history([history]),
            data_version=self.settings.data_version,
        )

    async def get_benchmarks(self, product_id: str) -> ProductBenchmarksResponse:
        product = self._require_product(product_id)
        return ProductBenchmarksResponse(
            product_id=product_id,
            benchmarks=[
                BenchmarkObservation(
                    benchmark_name="deterministic_relative_component_score",
                    workload="general",
                    score=product.performance,
                    unit="relative index",
                    higher_is_better=True,
                    basis="predicted",
                    model_version=self.settings.ranking_model_version,
                )
            ],
            data_version=self.settings.data_version,
            performance_model_version=self.settings.ranking_model_version,
        )

    async def get_reviews(self, product_id: str) -> ProductReviewsResponse:
        self._require_product(product_id)
        return ProductReviewsResponse(
            product_id=product_id,
            evidence=[],
            data_version=self.settings.data_version,
        )

    async def check_compatibility(
        self, request: CompatibilityCheckRequest
    ) -> CompatibilityCheckResponse:
        selected: dict[ComponentCategory, ProductRecord] = {}
        custom_unknowns: list[CompatibilityCheck] = []
        for item in request.components:
            if item.category in selected:
                custom_unknowns.append(
                    CompatibilityCheck(
                        rule_id="exactly-one-per-category",
                        status=CompatibilityStatus.FAIL,
                        message=f"Multiple {item.category.value} components were supplied.",
                        affected_categories=[item.category],
                    )
                )
                continue
            product = self.products.get(item.product_id or "")
            if product is None:
                custom_unknowns.append(
                    CompatibilityCheck(
                        rule_id="catalogue-specifications-known",
                        status=CompatibilityStatus.UNKNOWN,
                        message=(
                            f"No verified catalogue specifications are available for "
                            f"'{item.product_id or item.canonical_name or item.category.value}'."
                        ),
                        affected_categories=[item.category],
                    )
                )
                continue
            if product.category is not item.category:
                custom_unknowns.append(
                    CompatibilityCheck(
                        rule_id="catalogue-category",
                        status=CompatibilityStatus.FAIL,
                        message="The supplied category conflicts with the canonical product.",
                        affected_categories=[item.category, product.category],
                    )
                )
                continue
            selected[item.category] = product
        if len(selected) == len(ComponentCategory) and not custom_unknowns:
            checks = self._evaluate_compatibility(selected)
        else:
            checks = custom_unknowns
            missing = [category for category in ComponentCategory if category not in selected]
            if missing:
                checks.append(
                    CompatibilityCheck(
                        rule_id="complete-build",
                        status=CompatibilityStatus.UNKNOWN,
                        message=(
                            "A complete-build decision requires all eight component categories."
                        ),
                        affected_categories=missing,
                    )
                )
        if any(check.status is CompatibilityStatus.FAIL for check in checks):
            status = CompatibilityStatus.FAIL
        elif any(check.status is CompatibilityStatus.UNKNOWN for check in checks):
            status = CompatibilityStatus.UNKNOWN
        elif any(check.status is CompatibilityStatus.WARNING for check in checks):
            status = CompatibilityStatus.WARNING
        else:
            status = CompatibilityStatus.PASS
        return CompatibilityCheckResponse(
            status=status,
            is_feasible=status in {CompatibilityStatus.PASS, CompatibilityStatus.WARNING},
            checks=checks,
            rule_version=self.settings.compatibility_rule_version,
            data_version=self.settings.data_version,
        )

    async def record_interaction(
        self, event: CanonicalInteractionEvent
    ) -> InteractionAccepted:
        event_id = interaction_event_id(event)
        accepted_at = datetime.now(UTC)
        async with self._lock:
            existing = next(
                (
                    item
                    for item in self._interactions
                    if item[0] == event_id
                    or (
                        event.idempotency_key_sha256 is not None
                        and item[1].session_id == event.session_id
                        and item[1].idempotency_key_sha256
                        == event.idempotency_key_sha256
                    )
                    or (
                        event.impression_id is not None
                        and item[1].impression_id == event.impression_id
                        and item[1].event_type == event.event_type
                    )
                ),
                None,
            )
            if existing is not None:
                _, existing_event, existing_at = existing
                if not (
                    same_canonical_interaction(existing_event, event)
                    or same_impression_interaction(existing_event, event)
                ):
                    raise ApiError(
                        status_code=409,
                        code="interaction_idempotency_conflict",
                        message="Idempotency-Key was already used for another interaction.",
                    )
                return InteractionAccepted(
                    event_id=event_id,
                    accepted_at=existing_at,
                    data_version=existing_event.data_version or self.settings.data_version,
                    rule_version=(
                        existing_event.rule_version
                        or self.settings.compatibility_rule_version
                    ),
                    trust_level=existing_event.trust_level,
                    replayed=True,
                )
            self._interactions.append((event_id, event, accepted_at))
        return InteractionAccepted(
            event_id=event_id,
            accepted_at=accepted_at,
            data_version=event.data_version or self.settings.data_version,
            rule_version=event.rule_version or self.settings.compatibility_rule_version,
            trust_level=event.trust_level,
        )
