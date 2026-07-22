"""Versioned HTTP contracts for the recommendation service."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ApiModel(BaseModel):
    # The response/request contract intentionally exposes ``model_version``.
    # Avoid Pydantic 2.9's broad ``model_`` warning without allowing fields to
    # collide with the actual validation and serialization APIs.
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        protected_namespaces=("model_validate", "model_dump"),
    )


class ComponentCategory(StrEnum):
    CPU = "cpu"
    GPU = "gpu"
    MOTHERBOARD = "motherboard"
    MEMORY = "memory"
    STORAGE = "storage"
    PSU = "psu"
    COOLER = "cooler"
    CASE = "case"


REQUIRED_CATEGORIES = tuple(ComponentCategory)


class WorkloadName(StrEnum):
    GAMING_1080P = "gaming_1080p"
    GAMING_1440P = "gaming_1440p"
    GAMING_4K = "gaming_4k"
    LOCAL_AI = "local_ai"
    SOFTWARE_DEVELOPMENT = "software_development"
    CONTENT_CREATION = "content_creation"


class BuildProfile(StrEnum):
    BEST_OVERALL = "best_overall"
    BEST_VALUE = "best_value"
    HIGHEST_PERFORMANCE = "highest_performance"
    MOST_UPGRADEABLE = "most_upgradeable"
    LOWEST_POWER = "lowest_power"


class BuildGenerationStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    INFEASIBLE = "infeasible"


class SolverStatus(StrEnum):
    OPTIMAL = "OPTIMAL"
    FEASIBLE = "FEASIBLE"
    INFEASIBLE = "INFEASIBLE"
    MODEL_INVALID = "MODEL_INVALID"
    UNKNOWN = "UNKNOWN"


class CompatVerdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    UNKNOWN = "unknown"


class WorkloadInput(ApiModel):
    name: WorkloadName
    weight: float = Field(gt=0, le=1)


class ExistingProductInput(ApiModel):
    product_id: str = Field(min_length=1, max_length=200)
    category: ComponentCategory
    canonical_name: str | None = Field(default=None, max_length=300)
    include_in_budget: bool = False


class BuildRequirements(ApiModel):
    minimum_gpu_vram_gb: int | None = Field(default=None, ge=0, le=256)
    minimum_memory_gb: int | None = Field(default=None, ge=1, le=2048)
    storage_gb: int | None = Field(default=None, ge=1, le=1_000_000)
    wifi_required: bool = False
    case_size: Literal["small_form_factor", "mini_tower", "mid_tower", "full_tower"] | None = None
    in_stock_only: bool = True

    @field_validator("minimum_gpu_vram_gb")
    @classmethod
    def zero_vram_means_no_minimum(cls, value: int | None) -> int | None:
        return None if value == 0 else value

    @field_validator("wifi_required", mode="before")
    @classmethod
    def null_wifi_means_not_required(cls, value: Any) -> Any:
        return False if value is None else value


class BuildPreferences(ApiModel):
    noise: Literal["low", "medium", "any"] | None = None
    upgradeability: Literal["low", "medium", "high"] | None = None
    power_efficiency: Literal["low", "medium", "high"] | None = None
    preferred_brands: list[str] = Field(default_factory=list, max_length=50)
    excluded_brands: list[str] = Field(default_factory=list, max_length=50)

    @model_validator(mode="after")
    def brands_must_not_overlap(self) -> BuildPreferences:
        preferred = {brand.casefold() for brand in self.preferred_brands}
        excluded = {brand.casefold() for brand in self.excluded_brands}
        overlap = sorted(preferred & excluded)
        if overlap:
            raise ValueError(f"brands cannot be both preferred and excluded: {', '.join(overlap)}")
        return self


class GenerateBuildsRequest(ApiModel):
    budget_sgd: float = Field(gt=0, le=1_000_000)
    workloads: list[WorkloadInput] = Field(min_length=1, max_length=6)
    existing_products: list[ExistingProductInput] = Field(default_factory=list, max_length=8)
    requirements: BuildRequirements = Field(default_factory=BuildRequirements)
    preferences: BuildPreferences = Field(default_factory=BuildPreferences)
    performance_target: str | None = Field(default=None, min_length=1, max_length=200)
    max_builds: int = Field(default=5, ge=1, le=5)
    requested_profiles: list[BuildProfile] | None = Field(default=None, min_length=1, max_length=5)

    @model_validator(mode="after")
    def validate_request(self) -> GenerateBuildsRequest:
        workload_names = [workload.name for workload in self.workloads]
        if len(set(workload_names)) != len(workload_names):
            raise ValueError("workload names must be unique")
        if abs(sum(workload.weight for workload in self.workloads) - 1.0) > 1e-6:
            raise ValueError("workload weights must sum to 1")
        categories = [product.category for product in self.existing_products]
        if len(set(categories)) != len(categories):
            raise ValueError("existing products must contain at most one product per category")
        if self.requested_profiles is not None:
            if len(set(self.requested_profiles)) != len(self.requested_profiles):
                raise ValueError("requested_profiles must be unique")
            if len(self.requested_profiles) > self.max_builds:
                raise ValueError("requested_profiles cannot contain more entries than max_builds")
        return self


class SourceReference(ApiModel):
    label: str
    url: str


class PerformanceBenchmarkEvidence(ApiModel):
    benchmark_id: str
    benchmark_name: str
    benchmark_version: str
    resolution: str | None = None
    preset: str | None = None
    operating_system: str | None = None
    driver_version: str | None = None
    source_url: str
    observed_at: datetime


class PerformanceSignal(ApiModel):
    workload: str
    metric: str
    value: float | None
    unit: str | None = None
    basis: Literal["observed", "predicted", "relative", "insufficient_data"]
    confidence: Literal["high", "medium", "low"] | None = None
    decision: Literal[
        "observed_benchmark",
        "precise_model_prediction",
        "model_not_promotion_eligible",
        "input_outside_training_contract",
        "model_not_promotion_eligible_and_input_outside_training_contract",
        "precise_predictions_disabled",
        "precise_predictions_disabled_and_input_outside_training_contract",
        "deterministic_baseline",
    ] | None = None
    model_version: str | None = None
    observed_at: datetime | None = None
    sources: list[SourceReference] = Field(default_factory=list)
    supporting_benchmark_ids: list[str] = Field(default_factory=list)
    benchmark_evidence: list[PerformanceBenchmarkEvidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def decision_matches_basis(self) -> PerformanceSignal:
        """Keep optional, legacy-safe decision data semantically truthful when present."""

        if self.decision is None:
            return self
        if self.basis == "observed" and self.decision != "observed_benchmark":
            raise ValueError("observed performance requires the observed_benchmark decision")
        if self.basis == "predicted" and self.decision != "precise_model_prediction":
            raise ValueError("predicted performance requires the precise_model_prediction decision")
        if self.basis == "relative" and self.decision not in {
            "model_not_promotion_eligible",
            "input_outside_training_contract",
            "model_not_promotion_eligible_and_input_outside_training_contract",
            "precise_predictions_disabled",
            "precise_predictions_disabled_and_input_outside_training_contract",
            "deterministic_baseline",
        }:
            raise ValueError("relative performance requires a bounded fallback decision")
        if self.basis == "insufficient_data":
            raise ValueError("insufficient-data performance must not claim a model decision")
        return self


class CompatibilityCheck(ApiModel):
    rule_id: str
    status: CompatVerdict
    message: str
    affected_categories: list[ComponentCategory] = Field(default_factory=list)
    evidence_source: str | None = None


class ExplanationItem(ApiModel):
    kind: Literal["performance", "value", "compatibility", "preference", "price"]
    text: str
    supporting_ids: list[str] = Field(default_factory=list)


class ReplacementCandidate(ApiModel):
    product_id: str
    canonical_name: str
    category: ComponentCategory
    price_sgd: float = Field(ge=0)
    retailer: str | None = None
    performance_delta: float | None = None
    price_delta_sgd: float | None = None
    power_delta_w: float | None = None
    compatibility_status: Literal["pass", "warning"] | None = None
    reasons: list[str] = Field(default_factory=list)


class BuildComponent(ApiModel):
    category: ComponentCategory
    product_id: str
    listing_id: str | None = None
    canonical_name: str
    brand: str | None = None
    retailer: str | None = None
    listing_url: str | None = None
    price_sgd: float = Field(ge=0)
    already_owned: bool = False
    component_score: float | None = Field(default=None, ge=0, le=100)
    selection_reasons: list[str] = Field(default_factory=list)
    performance_signals: list[PerformanceSignal] = Field(default_factory=list)
    alternatives: list[ReplacementCandidate] = Field(default_factory=list)


class BuildResult(ApiModel):
    build_id: str
    profile: BuildProfile
    total_price_sgd: float = Field(ge=0)
    overall_score: float = Field(ge=0, le=100)
    value_score: float | None = Field(default=None, ge=0, le=100)
    upgradeability_score: float | None = Field(default=None, ge=0, le=100)
    efficiency_score: float | None = Field(default=None, ge=0, le=100)
    estimated_peak_power_w: float | None = Field(default=None, ge=0)
    workload_scores: dict[str, float | None]
    compatibility_status: Literal["pass", "warning"]
    components: list[BuildComponent]
    compatibility_checks: list[CompatibilityCheck] = Field(default_factory=list)
    warnings: list[CompatibilityCheck] = Field(default_factory=list)
    explanation: list[ExplanationItem] = Field(default_factory=list)
    alternatives: list[ReplacementCandidate] = Field(default_factory=list)
    generated_at: datetime
    data_version: str
    ranking_model: str
    rule_version: str
    solver_version: str
    solver_status: SolverStatus
    solver_ran: bool

    @model_validator(mode="after")
    def ensure_safe_complete_build(self) -> BuildResult:
        if any(
            check.status in {CompatVerdict.FAIL, CompatVerdict.UNKNOWN}
            for check in self.compatibility_checks
        ):
            raise ValueError("returned builds may not contain hard FAIL or UNKNOWN checks")
        present = [component.category for component in self.components]
        if len(set(present)) != len(present):
            raise ValueError("returned builds may not contain duplicate component categories")
        if set(present) != set(REQUIRED_CATEGORIES):
            raise ValueError("returned builds must contain every required component category")
        if self.solver_status in {SolverStatus.MODEL_INVALID, SolverStatus.UNKNOWN}:
            raise ValueError("returned builds require a conclusive solver status")
        return self


class PublicBuildComponent(ApiModel):
    """A public component projection with no listing, ownership, or internal identifiers."""

    category: ComponentCategory
    canonical_name: str = Field(min_length=1, max_length=400)
    brand: str | None = Field(default=None, max_length=120)
    price_sgd: float | None = Field(default=None, ge=0)
    component_score: float | None = Field(default=None, ge=0, le=100)
    selection_reason: str | None = Field(default=None, max_length=500)


class PublicBuildSnapshot(ApiModel):
    """Immutable allow-listed build data that is safe to render from a public link."""

    profile: BuildProfile
    total_price_sgd: float = Field(ge=0)
    overall_score: float = Field(ge=0, le=100)
    value_score: float | None = Field(default=None, ge=0, le=100)
    upgradeability_score: float | None = Field(default=None, ge=0, le=100)
    efficiency_score: float | None = Field(default=None, ge=0, le=100)
    estimated_peak_power_w: float | None = Field(default=None, ge=0)
    workload_scores: dict[str, float | None]
    compatibility_status: Literal["pass", "warning"]
    components: list[PublicBuildComponent] = Field(min_length=len(REQUIRED_CATEGORIES))
    explanations: list[str] = Field(default_factory=list, max_length=4)
    warnings: list[str] = Field(default_factory=list, max_length=4)
    generated_at: datetime
    data_version: str = Field(min_length=1, max_length=120)
    ranking_model: str = Field(min_length=1, max_length=120)
    rule_version: str = Field(min_length=1, max_length=120)
    solver_version: str = Field(min_length=1, max_length=120)

    @model_validator(mode="after")
    def has_one_component_per_required_category(self) -> PublicBuildSnapshot:
        categories = [item.category for item in self.components]
        if len(categories) != len(set(categories)) or set(categories) != set(REQUIRED_CATEGORIES):
            raise ValueError("public build snapshots require exactly one component per category")
        return self


class BuildShareCreated(ApiModel):
    """Creation response; the revocation capability is returned exactly once."""

    share_id: str = Field(min_length=1, max_length=80)
    revocation_token: str = Field(min_length=32, max_length=200)
    created_at: datetime
    expires_at: datetime


class PublicBuildShare(ApiModel):
    share_id: str = Field(min_length=1, max_length=80)
    created_at: datetime
    expires_at: datetime
    snapshot: PublicBuildSnapshot


class RevokeBuildShareRequest(ApiModel):
    revocation_token: str = Field(min_length=32, max_length=200)


class BuildShareRevoked(ApiModel):
    share_id: str = Field(min_length=1, max_length=80)
    revoked_at: datetime


class AdminMappingQueue(ApiModel):
    offer_count: int = Field(ge=0)
    matched_count: int = Field(ge=0)
    unmatched_count: int = Field(ge=0)
    manual_review_count: int = Field(ge=0)
    rejected_conflict_count: int = Field(ge=0)
    model_rejected_count: int = Field(ge=0)


class AdminPriceFreshness(ApiModel):
    snapshot_count: int = Field(ge=0)
    newest_observed_at: datetime | None = None
    stale_snapshot_count: int | None = Field(default=None, ge=0)
    stale_after_hours: int = Field(ge=1)


class AdminMissingField(ApiModel):
    category: ComponentCategory
    field_group: str = Field(min_length=1, max_length=120)
    missing_product_count: int = Field(ge=0)
    product_count: int = Field(ge=0)


class AdminPipelineOperations(ApiModel):
    """Aggregate-only status from instrumented pipeline user-code receipts."""

    event_window_hours: int = Field(ge=1, le=24 * 31)
    event_count: int = Field(ge=0)
    succeeded_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    latest_event_at: datetime | None = None
    latest_failure_at: datetime | None = None
    invalid_receipt_count: int = Field(ge=0)
    truncated: bool = False


class AdminOperationsResponse(ApiModel):
    """Read-only operational counters; absence is explicit instead of inferred as healthy."""

    data_version: str = Field(min_length=1, max_length=120)
    generated_at: datetime
    mode: Literal["demo", "processed_catalog"]
    mapping_queue: AdminMappingQueue | None = None
    price_freshness: AdminPriceFreshness | None = None
    missing_critical_fields: list[AdminMissingField] = Field(default_factory=list)
    release_blockers: list[str] = Field(default_factory=list)
    pipeline_operations: AdminPipelineOperations | None = None
    pipeline_failure_events_available: bool
    notes: list[str] = Field(default_factory=list)


class InfeasibilityReason(ApiModel):
    code: str
    message: str
    affected_categories: list[ComponentCategory] = Field(default_factory=list)


class SuggestedRelaxation(ApiModel):
    field_path: str
    current_value: Any
    proposed_value: Any
    expected_effect: str


class InfeasibilityExplanation(ApiModel):
    reasons: list[InfeasibilityReason]
    suggested_relaxations: list[SuggestedRelaxation] = Field(default_factory=list)


class SolverProfileOutcome(ApiModel):
    profile: BuildProfile
    status: SolverStatus
    wall_time_seconds: float = Field(ge=0)
    objective_value: int | None = None


class GenerateBuildsResponse(ApiModel):
    request_id: str
    status: BuildGenerationStatus
    generated_at: datetime
    data_version: str
    ranking_model: str
    retrieval_model: str
    performance_model: str
    rule_version: str
    solver_version: str
    solver_status: SolverStatus
    solver_ran: bool
    solver_profile_statuses: list[SolverProfileOutcome] = Field(default_factory=list)
    solver_validator_rejections: int = Field(default=0, ge=0)
    builds: list[BuildResult]
    infeasibility: InfeasibilityExplanation | None = None

    @model_validator(mode="after")
    def status_matches_builds(self) -> GenerateBuildsResponse:
        if self.status is BuildGenerationStatus.INFEASIBLE and self.builds:
            raise ValueError("infeasible responses cannot contain builds")
        if self.status is not BuildGenerationStatus.INFEASIBLE and not self.builds:
            raise ValueError("complete or partial responses must contain at least one build")
        if self.solver_status in {SolverStatus.MODEL_INVALID, SolverStatus.UNKNOWN}:
            raise ValueError("recommendation responses require a conclusive solver status")
        if self.solver_status is SolverStatus.INFEASIBLE and self.builds:
            raise ValueError("an infeasible solver outcome cannot publish builds")
        if not self.solver_ran and self.solver_profile_statuses:
            raise ValueError("profile solver outcomes require solver_ran=true")
        return self


class ProductSearchRequest(ApiModel):
    query: str = Field(default="", max_length=500)
    category: ComponentCategory | None = None
    compatible_with_build_id: str | None = None
    brand: str | None = None
    in_stock_only: bool = True
    limit: int = Field(default=20, ge=1, le=100)
    page: int | None = Field(default=None, ge=1, le=1_000_000)
    page_size: int | None = Field(default=None, ge=1, le=100)
    cursor: str | None = Field(default=None, min_length=1, max_length=512)

    @property
    def effective_page_size(self) -> int:
        """Use the paged contract when present while preserving legacy ``limit`` callers."""

        return self.page_size if self.page_size is not None else self.limit


_PRODUCT_SEARCH_QUERY_VERSION = 1
_PRODUCT_SEARCH_QUERY_DOMAIN = "pcbr-product-search-query-v1"


def product_search_identity(
    request: ProductSearchRequest,
    *,
    data_version: str,
    retrieval_model: str,
) -> tuple[str, dict[str, Any]]:
    """Return a stable query ID and its versioned durable analytics payload.

    Pagination is deliberately excluded: all pages of one ranked candidate set must share the
    same query identity. Data and retrieval versions are included so feedback cannot silently
    cross serving releases.
    """

    constraints: dict[str, Any] = {
        "schema_version": _PRODUCT_SEARCH_QUERY_VERSION,
        "kind": "product_search",
        "query": " ".join(request.query.casefold().split()),
        "category": request.category.value if request.category else None,
        "compatible_with_build_id": request.compatible_with_build_id,
        "brand": request.brand.casefold() if request.brand else None,
        "in_stock_only": request.in_stock_only,
        "data_version": data_version,
        "retrieval_model": retrieval_model,
    }
    canonical = json.dumps(
        {"domain": _PRODUCT_SEARCH_QUERY_DOMAIN, **constraints},
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"search_{digest}", constraints


class InvalidProductSearchCursor(ValueError):
    """Raised when an opaque catalogue cursor is malformed or belongs to another search."""


_PRODUCT_SEARCH_CURSOR_VERSION = 1
_PRODUCT_SEARCH_CURSOR_DOMAIN = "pcbr-product-search-cursor-v1"


def _product_search_cursor_scope(request: ProductSearchRequest) -> str:
    scope = {
        "brand": request.brand.casefold() if request.brand else None,
        "category": request.category.value if request.category else None,
        "compatible_with_build_id": request.compatible_with_build_id,
        "in_stock_only": request.in_stock_only,
        "page_size": request.effective_page_size,
        "query": " ".join(request.query.casefold().split()),
    }
    canonical = json.dumps(scope, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


def _product_search_cursor_signature(*, page: int, page_size: int, scope: str) -> str:
    payload = (
        f"{_PRODUCT_SEARCH_CURSOR_DOMAIN}|{_PRODUCT_SEARCH_CURSOR_VERSION}|"
        f"{page}|{page_size}|{scope}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def encode_product_search_cursor(request: ProductSearchRequest, *, page: int) -> str:
    """Return a deterministic, opaque cursor scoped to one immutable search shape."""

    if not 1 <= page <= 1_000_000:
        raise ValueError("cursor page must be between 1 and 1,000,000")
    page_size = request.effective_page_size
    scope = _product_search_cursor_scope(request)
    payload = {
        "p": page,
        "s": page_size,
        "scope": scope,
        "sig": _product_search_cursor_signature(
            page=page,
            page_size=page_size,
            scope=scope,
        ),
        "v": _PRODUCT_SEARCH_CURSOR_VERSION,
    }
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return encoded.rstrip(b"=").decode("ascii")


def resolve_product_search_page(request: ProductSearchRequest) -> int:
    """Resolve and validate page/cursor semantics without trusting client offsets."""

    if request.cursor is None:
        return request.page or 1
    try:
        encoded = request.cursor.encode("ascii")
        padding = b"=" * (-len(encoded) % 4)
        decoded = base64.b64decode(encoded + padding, altchars=b"-_", validate=True)
        payload = json.loads(decoded.decode("utf-8"))
    except (UnicodeEncodeError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        raise InvalidProductSearchCursor("pagination cursor is malformed") from error
    if not isinstance(payload, dict) or set(payload) != {"p", "s", "scope", "sig", "v"}:
        raise InvalidProductSearchCursor("pagination cursor has an unsupported shape")
    page = payload["p"]
    page_size = payload["s"]
    scope = payload["scope"]
    signature = payload["sig"]
    version = payload["v"]
    if (
        type(page) is not int
        or type(page_size) is not int
        or not isinstance(scope, str)
        or not isinstance(signature, str)
        or version != _PRODUCT_SEARCH_CURSOR_VERSION
        or not 1 <= page <= 1_000_000
        or not 1 <= page_size <= 100
    ):
        raise InvalidProductSearchCursor("pagination cursor contains invalid values")
    expected_scope = _product_search_cursor_scope(request)
    expected_signature = _product_search_cursor_signature(
        page=page,
        page_size=page_size,
        scope=scope,
    )
    if (
        page_size != request.effective_page_size
        or not hmac.compare_digest(scope, expected_scope)
        or not hmac.compare_digest(signature, expected_signature)
    ):
        raise InvalidProductSearchCursor("pagination cursor does not match this search")
    if request.page is not None and request.page != page:
        raise InvalidProductSearchCursor("page does not match the pagination cursor")
    return page


class ProductSearchItem(ApiModel):
    product_id: str
    category: ComponentCategory
    canonical_name: str
    brand: str | None = None
    model: str | None = None
    lowest_price_sgd: float | None = Field(default=None, ge=0)
    stock_status: str | None = None
    compatibility_status: CompatVerdict | None = None


class ProductFacetCount(ApiModel):
    value: str = Field(min_length=1)
    count: int = Field(ge=0)


class ProductSearchFacets(ApiModel):
    categories: list[ProductFacetCount] = Field(default_factory=list)
    brands: list[ProductFacetCount] = Field(default_factory=list)


class ProductSearchPagination(ApiModel):
    page: int = Field(ge=1)
    page_size: int = Field(ge=1, le=100)
    total_pages: int = Field(ge=1)
    has_previous: bool
    has_next: bool
    previous_cursor: str | None = None
    next_cursor: str | None = None

    @model_validator(mode="after")
    def navigation_matches_page(self) -> ProductSearchPagination:
        if self.page > self.total_pages:
            raise ValueError("pagination page cannot exceed total_pages")
        if self.has_previous != (self.page > 1):
            raise ValueError("has_previous does not match page")
        if self.has_next != (self.page < self.total_pages):
            raise ValueError("has_next does not match total_pages")
        if self.has_previous != (self.previous_cursor is not None):
            raise ValueError("previous_cursor presence must match has_previous")
        if self.has_next != (self.next_cursor is not None):
            raise ValueError("next_cursor presence must match has_next")
        return self


class SourceAttribution(ApiModel):
    """Visible provenance and licence notice required for public catalogue output."""

    source_name: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    licence_or_access_note: str = Field(min_length=1)
    attribution_notice: str | None = None
    licence_url: str | None = None
    retrieved_at: datetime


class CatalogueCoverage(ApiModel):
    canonical_products: int = Field(ge=0)
    retailer_listings: int | None = Field(default=None, ge=0)
    source_count: int | None = Field(default=None, ge=0)
    category_count: int | None = Field(default=None, ge=0)
    as_of: datetime | None = None
    scope_label: str = Field(min_length=1)
    source_attributions: list[SourceAttribution] = Field(default_factory=list)


class ProductSearchResponse(ApiModel):
    query_id: str = Field(min_length=1, max_length=80)
    products: list[ProductSearchItem]
    total: int = Field(ge=0)
    # Candidate counts cover the bounded search universe before pagination.  They
    # intentionally contain no product identifiers or raw query text.
    retrieved_candidates: int = Field(default=0, ge=0)
    filtered_category: int = Field(default=0, ge=0)
    filtered_brand: int = Field(default=0, ge=0)
    filtered_incompatible: int = Field(default=0, ge=0)
    filtered_unknown: int = Field(default=0, ge=0)
    data_version: str
    retrieval_model: str
    facets: ProductSearchFacets | None = None
    pagination: ProductSearchPagination | None = None
    coverage: CatalogueCoverage | None = None

    @model_validator(mode="after")
    def candidate_funnel_is_consistent(self) -> ProductSearchResponse:
        if self.total > self.retrieved_candidates:
            raise ValueError("search total cannot exceed retrieved candidates")
        if self.filtered_category + self.filtered_brand > self.retrieved_candidates:
            raise ValueError("category and brand filters cannot exceed retrieved candidates")
        return self


class ProductDetail(ProductSearchItem):
    manufacturer_part_number: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    source_confidence: float | None = Field(default=None, ge=0, le=1)
    source_url: str | None = None
    source_attributions: list[SourceAttribution] = Field(default_factory=list)
    updated_at: datetime
    data_version: str


class PriceObservation(ApiModel):
    listing_id: str
    retailer: str
    observed_at: datetime
    base_price_sgd: float = Field(ge=0)
    shipping_price_sgd: float = Field(ge=0)
    stock_status: str
    condition: str
    current_offer_eligible: bool
    listing_url: str | None = None


class PriceHistoryAnomaly(ApiModel):
    observed_at: datetime
    listing_id: str
    delivered_price_sgd: float = Field(ge=0)
    direction: Literal["high", "low"]
    modified_z_score: float | None = None
    source_url: str | None = None


class PriceIntelligenceSummary(ApiModel):
    """Descriptive statistics from stored observations, never a price forecast."""

    basis: Literal["descriptive_observed_history"] = "descriptive_observed_history"
    currency: str = Field(min_length=3, max_length=3)
    as_of: datetime
    current_delivered_price_sgd: float | None = Field(default=None, ge=0)
    median_30d_sgd: float | None = Field(default=None, ge=0)
    median_90d_sgd: float | None = Field(default=None, ge=0)
    percentile_90d: float | None = Field(default=None, ge=0, le=100)
    recent_low_90d_sgd: float | None = Field(default=None, ge=0)
    volatility_90d_pct: float | None = Field(default=None, ge=0)
    current_seller_count: int = Field(ge=0)
    seller_trend: Literal["increasing", "stable", "decreasing", "insufficient_history"]
    stock_trend: Literal["increasing", "stable", "decreasing", "insufficient_history"]
    history_days_30d: int = Field(ge=0, le=31)
    history_days_90d: int = Field(ge=0, le=91)
    history_sufficient: bool
    labels: list[str] = Field(default_factory=list)
    anomalies: list[PriceHistoryAnomaly] = Field(default_factory=list)
    observations_analyzed: int = Field(ge=1)
    analysis_truncated: bool = False

    @model_validator(mode="after")
    def sparse_history_has_no_precision_statistics(self) -> PriceIntelligenceSummary:
        if not self.history_sufficient and (
            self.percentile_90d is not None or self.volatility_90d_pct is not None
        ):
            raise ValueError(
                "insufficient price history cannot expose percentile or volatility statistics"
            )
        return self


class ProductPricesResponse(ApiModel):
    product_id: str
    current_lowest_price_sgd: float | None
    observations: list[PriceObservation]
    price_intelligence: PriceIntelligenceSummary | None = None
    data_version: str


class BenchmarkObservation(ApiModel):
    benchmark_name: str
    workload: str
    score: float
    unit: str
    higher_is_better: bool
    basis: Literal["observed", "predicted"]
    model_version: str | None = None
    source_url: str | None = None
    observed_at: datetime | None = None


class ProductBenchmarksResponse(ApiModel):
    product_id: str
    benchmarks: list[BenchmarkObservation]
    data_version: str
    performance_model_version: str


class ReviewNote(ApiModel):
    aspect: str
    sentiment: Literal["positive", "neutral", "negative", "mixed"]
    evidence_text: str
    source_url: str | None = None
    published_at: datetime | None = None
    confidence: float = Field(ge=0, le=1)


class ProductReviewsResponse(ApiModel):
    product_id: str
    evidence: list[ReviewNote]
    data_version: str


class ReplacementRequest(ApiModel):
    category: ComponentCategory
    replacement_product_id: str
    mode: Literal["lock_other_components", "reoptimize_unlocked"] = "lock_other_components"


class ReplacementResponse(ApiModel):
    build: BuildResult
    changed_categories: list[ComponentCategory]
    price_delta_sgd: float
    workload_score_deltas: dict[str, float]
    new_warnings: list[CompatibilityCheck]
    data_version: str
    ranking_model: str
    rule_version: str
    solver_version: str
    solver_status: SolverStatus
    solver_ran: bool
    solver_profile_statuses: list[SolverProfileOutcome] = Field(default_factory=list)
    solver_validator_rejections: int = Field(default=0, ge=0)


class CompatibilityComponent(ApiModel):
    product_id: str | None = None
    category: ComponentCategory
    canonical_name: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class CompatibilityCheckRequest(ApiModel):
    components: list[CompatibilityComponent] = Field(min_length=1, max_length=16)


class CompatibilityCheckResponse(ApiModel):
    status: CompatVerdict
    is_feasible: bool
    checks: list[CompatibilityCheck]
    rule_version: str
    data_version: str


class InteractionRecord(ApiModel):
    event_type: Literal[
        "search_submitted",
        "build_generated",
        "build_viewed",
        "build_saved",
        "build_shared",
        "component_viewed",
        "component_replaced",
        "comparison_opened",
        "retailer_clicked",
        "recommendation_dismissed",
        "feedback_submitted",
    ]
    session_id: str = Field(min_length=1, max_length=160)
    user_id: str | None = Field(default=None, max_length=160)
    query_id: str | None = None
    build_id: str | None = None
    product_id: str | None = None
    rank_position: int | None = Field(default=None, ge=1)
    model_version: str | None = Field(
        default=None,
        description="Deprecated client hint; the API stores its active ranking-model version.",
        json_schema_extra={"deprecated": True},
    )
    data_version: str | None = Field(
        default=None,
        description="Deprecated client hint; the API stores its active data version.",
        json_schema_extra={"deprecated": True},
    )
    rule_version: str | None = Field(
        default=None,
        min_length=1,
        description="Deprecated client hint; the API stores its active compatibility-rule version.",
        json_schema_extra={"deprecated": True},
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def ranked_event_identifies_a_result(self) -> InteractionRecord:
        if self.rank_position is not None and self.product_id is None and self.build_id is None:
            raise ValueError("ranked events must reference a product or build")
        return self


class InteractionAccepted(ApiModel):
    event_id: str
    accepted_at: datetime
    status: Literal["accepted"] = "accepted"
    data_version: str
    rule_version: str


class CatalogueReadinessSummary(ApiModel):
    """Public, aggregate-only view of a measured catalogue release gate.

    The detailed report remains a pipeline artifact.  This response deliberately exposes
    counts, coverage, and release blockers only, never source documents, retailer URLs, or
    unreviewed entity-resolution evidence.
    """

    products_by_category: dict[str, int] = Field(default_factory=dict)
    compatibility_ready_products_by_category: dict[str, int] = Field(default_factory=dict)
    matched_listings_by_category: dict[str, int] = Field(default_factory=dict)
    in_stock_listings_by_category: dict[str, int] = Field(default_factory=dict)
    offer_count: int = Field(ge=0)
    mapping_rate: float = Field(ge=0, le=1)
    has_complete_priced_coverage: bool
    has_complete_in_stock_coverage: bool
    product_provenance_complete_count: int = Field(ge=0)
    offer_provenance_complete_count: int = Field(ge=0)
    offer_rights_production_valid_count: int = Field(ge=0)
    rights_territory: str = Field(min_length=2, max_length=8)
    entity_resolution_model_version: str | None = Field(default=None, max_length=200)
    entity_resolution_model_production_authorized: bool
    production_ready: bool
    production_blockers: list[str] = Field(default_factory=list)


class FreshnessResponse(ApiModel):
    data_version: str
    status: Literal["fresh", "stale", "degraded"]
    last_catalog_update: datetime
    prices_updated_at: datetime
    stale_after_hours: int
    source_count: int
    product_count: int
    listing_count: int
    production_ready: bool
    release_artifact_verification: Literal[
        "verified", "development_unverified", "not_verified"
    ]
    readiness_blockers: list[str] = Field(default_factory=list)
    catalogue_readiness: CatalogueReadinessSummary | None = None


class HealthResponse(ApiModel):
    status: Literal["ok"]
    service: str
    environment: str


class ReadinessResponse(ApiModel):
    status: Literal["ready", "not_ready"]
    checks: dict[str, Literal["ready", "not_ready"]]
    data_version: str
    ranking_model: str
    rule_version: str
    solver_version: str


class ErrorDetail(ApiModel):
    code: str
    message: str
    request_id: str | None = None
    details: Any | None = None


class ErrorResponse(ApiModel):
    message: str
    error: ErrorDetail
