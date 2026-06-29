"""Public contracts for constraint-based PC build optimisation.

The optimiser deliberately consumes a small, immutable candidate projection rather than
depending on persistence or ranking internals.  Domain objects can be converted with the
``from_domain`` helpers in :mod:`pc_build_recommender.optimizer.adapters`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar

from pc_build_recommender.domain import (
    BuildRequestSpec,
    BuildPreset,
    MasterProduct,
    CompatVerdict,
    ComponentKind,
    RetailerListing,
)

REQUIRED_CATEGORIES: tuple[ComponentKind, ...] = (
    ComponentKind.CPU,
    ComponentKind.GPU,
    ComponentKind.MOTHERBOARD,
    ComponentKind.MEMORY,
    ComponentKind.STORAGE,
    ComponentKind.POWER_SUPPLY,
    ComponentKind.COOLER,
    ComponentKind.CASE,
)


def _domain_compatibility_status(value: object) -> CompatVerdict:
    """Normalise the compatibility package's uppercase and domain lowercase statuses."""

    raw_value = getattr(value, "value", value)
    return CompatVerdict(str(raw_value).casefold())


class FeatureOperator(StrEnum):
    """Supported hard-filter comparisons for structured requirements."""

    EQUALS = "equals"
    AT_LEAST = "at_least"
    CONTAINS = "contains"
    TRUTHY = "truthy"


class OptimizationStatus(StrEnum):
    """Stable solver statuses exposed without leaking OR-Tools integer constants."""

    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    MODEL_INVALID = "model_invalid"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class CandidateScores:
    """Normalised component-level objective inputs, each conventionally in ``[0, 100]``."""

    performance: float = 0.0
    value: float = 0.0
    reliability: float = 0.0
    upgradeability: float = 0.0
    efficiency: float = 0.0
    preference: float = 0.0
    warning_penalty: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "performance",
            "value",
            "reliability",
            "upgradeability",
            "efficiency",
            "preference",
            "warning_penalty",
        ):
            value = float(getattr(self, name))
            if not 0 <= value <= 100:
                raise ValueError(f"{name} must be in [0, 100], got {value}")


@dataclass(frozen=True, slots=True)
class OptimizationCandidate:
    """One canonical product paired with its selected retailer offer."""

    product_id: str
    category: ComponentKind
    price_cents: int
    brand: str = ""
    canonical_name: str = ""
    listing_id: str | None = None
    in_stock: bool = True
    scores: CandidateScores = field(default_factory=CandidateScores)
    attributes: Mapping[str, Any] = field(default_factory=dict)
    power_draw_watts: int | None = None
    psu_wattage: int | None = None
    required_power_connectors: Mapping[str, int] = field(default_factory=dict)
    provided_power_connectors: Mapping[str, int] = field(default_factory=dict)
    eps_connectors: int | None = None
    recommended_psu_watts: int | None = None
    source_product: object | None = field(default=None, repr=False, compare=False)
    source_listing: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.product_id:
            raise ValueError("product_id must not be empty")
        object.__setattr__(self, "category", ComponentKind(self.category))
        if self.price_cents < 0:
            raise ValueError("price_cents must be non-negative")
        for name in ("power_draw_watts", "psu_wattage", "eps_connectors"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")
        for connector_map_name in (
            "required_power_connectors",
            "provided_power_connectors",
        ):
            connector_map = getattr(self, connector_map_name)
            if any(int(count) < 0 for count in connector_map.values()):
                raise ValueError(f"{connector_map_name} counts must be non-negative")

    @classmethod
    def from_domain(
        cls,
        product: MasterProduct,
        listing: RetailerListing | None,
        *,
        scores: CandidateScores | Mapping[str, float] | object | None = None,
    ) -> OptimizationCandidate:
        """Create a candidate from public domain objects through the narrow adapter."""

        from .adapters import candidate_from_domain

        return candidate_from_domain(product, listing, scores=scores)

    def attribute(self, name: str, default: Any = None) -> Any:
        """Read a category attribute by name."""

        return self.attributes.get(name, default)


@dataclass(frozen=True, slots=True)
class FeatureRequirement:
    """A category-scoped hard feature requirement."""

    category: ComponentKind
    attribute: str
    expected: Any = True
    operator: FeatureOperator = FeatureOperator.EQUALS
    description: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "category", ComponentKind(self.category))
        object.__setattr__(self, "operator", FeatureOperator(self.operator))
        if not self.attribute:
            raise ValueError("feature attribute must not be empty")


@dataclass(frozen=True, slots=True)
class PairwiseCompatibility:
    """A precomputed compatibility outcome for two concrete products.

    A hard ``UNKNOWN`` is treated exactly like a hard ``FAIL``.  This prevents missing
    compatibility evidence from silently becoming a passing build.
    """

    left_product_id: str
    right_product_id: str
    status: CompatVerdict
    message: str = ""
    hard: bool = True
    penalty_points: int = 5

    def __post_init__(self) -> None:
        if not self.left_product_id or not self.right_product_id:
            raise ValueError("pairwise constraints require two product IDs")
        if self.left_product_id == self.right_product_id:
            raise ValueError("a pairwise constraint must reference distinct products")
        object.__setattr__(self, "status", _domain_compatibility_status(self.status))
        if self.penalty_points < 0:
            raise ValueError("penalty_points must be non-negative")

    @property
    def key(self) -> frozenset[str]:
        return frozenset((self.left_product_id, self.right_product_id))

    @property
    def is_forbidden(self) -> bool:
        return self.status == CompatVerdict.FAIL or (
            self.hard and self.status == CompatVerdict.UNKNOWN
        )


IndependentValidator = Callable[[Mapping[ComponentKind, OptimizationCandidate]], object]


@dataclass(frozen=True, slots=True)
class ProfileWeights:
    """Integer-friendly profile weights applied to normalised candidate scores."""

    performance: int
    value: int
    reliability: int
    upgradeability: int
    efficiency: int
    preference: int
    price_penalty_divisor: int | None = None
    power_penalty_per_watt: int = 0

    def __post_init__(self) -> None:
        score_weights = (
            self.performance,
            self.value,
            self.reliability,
            self.upgradeability,
            self.efficiency,
            self.preference,
        )
        if any(weight < 0 for weight in score_weights):
            raise ValueError("profile weights must be non-negative")
        if sum(score_weights) != 100:
            raise ValueError("profile score weights must sum to 100")
        if self.price_penalty_divisor is not None and self.price_penalty_divisor <= 0:
            raise ValueError("price_penalty_divisor must be positive")


PROFILE_WEIGHTS: Mapping[BuildPreset, ProfileWeights] = {
    BuildPreset.BEST_OVERALL: ProfileWeights(45, 25, 10, 10, 5, 5),
    BuildPreset.BEST_VALUE: ProfileWeights(
        20,
        55,
        10,
        5,
        5,
        5,
        price_penalty_divisor=100,
    ),
    BuildPreset.HIGHEST_PERFORMANCE: ProfileWeights(70, 10, 5, 5, 5, 5),
    BuildPreset.MOST_UPGRADEABLE: ProfileWeights(20, 10, 10, 50, 5, 5),
    BuildPreset.LOWEST_POWER: ProfileWeights(
        10,
        10,
        10,
        5,
        60,
        5,
        power_penalty_per_watt=1_000,
    ),
}


DEFAULT_POWER_ALLOWANCES_WATTS: Mapping[ComponentKind, int] = {
    ComponentKind.MOTHERBOARD: 50,
    ComponentKind.MEMORY: 10,
    ComponentKind.STORAGE: 10,
    ComponentKind.COOLER: 10,
    ComponentKind.CASE: 10,
}


@dataclass(frozen=True, slots=True)
class OptimizationProblem:
    """All candidates and hard constraints needed for one build-generation request."""

    candidates: tuple[OptimizationCandidate, ...]
    budget_cents: int
    profiles: tuple[BuildPreset, ...] = tuple(BuildPreset)
    locked_product_ids: frozenset[str] = field(default_factory=frozenset)
    minimum_gpu_vram_gb: int | None = None
    minimum_memory_gb: int | None = None
    minimum_storage_gb: int | None = None
    required_memory_type: object | None = None
    required_motherboard_form_factor: object | None = None
    wifi_required: bool = False
    required_case_size: object | None = None
    in_stock_only: bool = True
    excluded_brands: frozenset[str] = field(default_factory=frozenset)
    required_features: tuple[FeatureRequirement, ...] = ()
    pairwise_compatibility: tuple[PairwiseCompatibility, ...] = ()
    power_headroom_percent: int = 25
    base_power_watts: int = 0
    category_power_allowances_watts: Mapping[ComponentKind, int] = field(
        default_factory=lambda: dict(DEFAULT_POWER_ALLOWANCES_WATTS)
    )
    required_eps_connectors: int = 1
    diversity_distance: int = 2
    meaningful_categories: tuple[ComponentKind, ...] = REQUIRED_CATEGORIES
    exclude_locked_from_budget: bool = True
    independent_validator: IndependentValidator | None = field(
        default=None, repr=False, compare=False
    )
    time_limit_seconds: float = 5.0
    random_seed: int = 42

    _MAX_HEADROOM: ClassVar[int] = 100

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "profiles", tuple(BuildPreset(p) for p in self.profiles))
        object.__setattr__(self, "locked_product_ids", frozenset(self.locked_product_ids))
        object.__setattr__(
            self,
            "excluded_brands",
            frozenset(brand.casefold() for brand in self.excluded_brands),
        )
        object.__setattr__(
            self,
            "meaningful_categories",
            tuple(ComponentKind(category) for category in self.meaningful_categories),
        )
        if self.budget_cents < 0:
            raise ValueError("budget_cents must be non-negative")
        if not self.profiles:
            raise ValueError("at least one objective profile is required")
        if len(set(self.profiles)) != len(self.profiles):
            raise ValueError("profiles must be unique")
        if not 0 <= self.power_headroom_percent <= self._MAX_HEADROOM:
            raise ValueError("power_headroom_percent must be between 0 and 100")
        if self.base_power_watts < 0:
            raise ValueError("base_power_watts must be non-negative")
        if self.required_eps_connectors < 0:
            raise ValueError("required_eps_connectors must be non-negative")
        if self.diversity_distance < 1:
            raise ValueError("diversity_distance must be at least 1")
        if self.time_limit_seconds <= 0:
            raise ValueError("time_limit_seconds must be positive")
        product_ids = [candidate.product_id for candidate in self.candidates]
        if len(set(product_ids)) != len(product_ids):
            raise ValueError("candidates must contain at most one offer per canonical product")
        pair_keys = [pair.key for pair in self.pairwise_compatibility]
        if len(set(pair_keys)) != len(pair_keys):
            raise ValueError("pairwise compatibility entries must be unique per product pair")

    @classmethod
    def from_domain(
        cls,
        request: BuildRequestSpec,
        products: Iterable[MasterProduct],
        listings: Iterable[RetailerListing],
        *,
        scores_by_product: Mapping[str, CandidateScores | Mapping[str, float] | object]
        | None = None,
        pairwise_compatibility: Iterable[PairwiseCompatibility]
        | Mapping[tuple[str, str], object] = (),
        independent_validator: IndependentValidator | None = None,
        compatibility_engine: object | None = None,
        **overrides: Any,
    ) -> OptimizationProblem:
        """Build a problem from the stable public domain request and catalogue types."""

        from .adapters import problem_from_domain

        return problem_from_domain(
            request,
            products,
            listings,
            scores_by_product=scores_by_product,
            pairwise_compatibility=pairwise_compatibility,
            independent_validator=independent_validator,
            compatibility_engine=compatibility_engine,
            **overrides,
        )

    def acquisition_price_cents(self, candidate: OptimizationCandidate) -> int:
        if self.exclude_locked_from_budget and candidate.product_id in self.locked_product_ids:
            return 0
        return candidate.price_cents


@dataclass(frozen=True, slots=True)
class ProfileSolveRecord:
    profile: BuildPreset
    status: OptimizationStatus
    wall_time_seconds: float
    objective_value: int | None = None


@dataclass(frozen=True, slots=True)
class OptimizationSolution:
    """One independently validated complete build."""

    profile: BuildPreset
    selected: Mapping[ComponentKind, OptimizationCandidate]
    total_price_cents: int
    catalog_total_price_cents: int
    objective_value: int
    estimated_load_watts: int
    required_psu_watts: int
    solver_status: OptimizationStatus
    warnings: tuple[str, ...] = ()
    compatibility_report: object | None = field(default=None, repr=False, compare=False)

    @property
    def product_ids(self) -> frozenset[str]:
        return frozenset(candidate.product_id for candidate in self.selected.values())

    @property
    def selected_candidates(self) -> tuple[OptimizationCandidate, ...]:
        return tuple(self.selected[category] for category in REQUIRED_CATEGORIES)


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    """Build-generation result with explicit partial and infeasible diagnostics."""

    status: OptimizationStatus
    solutions: tuple[OptimizationSolution, ...]
    infeasibility_reasons: tuple[str, ...] = ()
    profile_statuses: tuple[ProfileSolveRecord, ...] = ()
    rejected_by_validator: int = 0

    @property
    def is_feasible(self) -> bool:
        return bool(self.solutions)
