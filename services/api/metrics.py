"""Bounded in-process Prometheus metrics for the single-worker API deployment."""

from __future__ import annotations

from collections import defaultdict
from threading import Lock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from services.api.models import AdminOperationsResponse

_DURATION_BUCKETS_SECONDS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
_BUILD_OUTCOMES = frozenset({"complete", "partial", "infeasible"})
_SOLVER_STATUSES = frozenset({"OPTIMAL", "FEASIBLE", "INFEASIBLE", "MODEL_INVALID", "UNKNOWN"})
_BUILD_PROFILES = frozenset(
    {"best_overall", "best_value", "highest_performance", "most_upgradeable", "lowest_power"}
)
_COMPATIBILITY_STATUSES = frozenset({"pass", "fail", "warning", "unknown"})
_INTERACTION_EVENT_TYPES = frozenset(
    {
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
    }
)
_FRESHNESS_STATUSES = frozenset({"fresh", "stale", "degraded"})
_RELEASE_ARTIFACT_VERIFICATIONS = frozenset({"verified", "development_unverified", "not_verified"})
_PERFORMANCE_SIGNAL_BASES = frozenset({"observed", "predicted", "relative", "insufficient_data"})
_PERFORMANCE_SIGNAL_CONFIDENCES = frozenset({"observed", "high", "medium", "low", "not_applicable"})
_PERFORMANCE_SIGNAL_DECISIONS = frozenset(
    {
        "observed_benchmark",
        "precise_model_prediction",
        "model_not_promotion_eligible",
        "input_outside_training_contract",
        "model_not_promotion_eligible_and_input_outside_training_contract",
        "precise_predictions_disabled",
        "precise_predictions_disabled_and_input_outside_training_contract",
        "deterministic_baseline",
        "not_applicable",
    }
)


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _require_label(value: str, allowed: frozenset[str], field_name: str) -> str:
    """Reject arbitrary labels so a caller cannot create unbounded metric series."""

    if value not in allowed:
        raise ValueError(f"unsupported {field_name} metric label: {value!r}")
    return value


def _boolean_label(value: bool) -> str:
    return "true" if value else "false"


class RequestMetrics:
    """Thread-safe counters with route-template labels to prevent unbounded cardinality."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._active = 0
        self._requests: defaultdict[tuple[str, str, int], int] = defaultdict(int)
        self._duration_count: defaultdict[tuple[str, str], int] = defaultdict(int)
        self._duration_sum: defaultdict[tuple[str, str], float] = defaultdict(float)
        self._duration_buckets: defaultdict[tuple[str, str, float], int] = defaultdict(int)

    def request_started(self) -> None:
        with self._lock:
            self._active += 1

    def request_finished(
        self,
        *,
        method: str,
        route: str,
        status_code: int,
        duration_seconds: float,
    ) -> None:
        key = (method, route)
        with self._lock:
            self._active = max(0, self._active - 1)
            self._requests[(method, route, status_code)] += 1
            self._duration_count[key] += 1
            self._duration_sum[key] += duration_seconds
            for bucket in _DURATION_BUCKETS_SECONDS:
                if duration_seconds <= bucket:
                    self._duration_buckets[(method, route, bucket)] += 1

    def render(self) -> str:
        """Render the Prometheus text exposition format from one consistent snapshot."""

        with self._lock:
            active = self._active
            requests = dict(self._requests)
            duration_count = dict(self._duration_count)
            duration_sum = dict(self._duration_sum)
            duration_buckets = dict(self._duration_buckets)

        lines = [
            "# HELP pcbr_http_requests_in_progress Current API requests in progress.",
            "# TYPE pcbr_http_requests_in_progress gauge",
            f"pcbr_http_requests_in_progress {active}",
            "# HELP pcbr_http_requests_total Completed API requests.",
            "# TYPE pcbr_http_requests_total counter",
        ]
        for (method, route, status_code), count in sorted(requests.items()):
            labels = (
                f'method="{_escape_label(method)}",'
                f'route="{_escape_label(route)}",status="{status_code}"'
            )
            lines.append(f"pcbr_http_requests_total{{{labels}}} {count}")

        lines.extend(
            [
                "# HELP pcbr_http_request_duration_seconds API request duration.",
                "# TYPE pcbr_http_request_duration_seconds histogram",
            ]
        )
        for method, route in sorted(duration_count):
            labels = f'method="{_escape_label(method)}",route="{_escape_label(route)}"'
            for bucket in _DURATION_BUCKETS_SECONDS:
                count = duration_buckets.get((method, route, bucket), 0)
                lines.append(
                    f'pcbr_http_request_duration_seconds_bucket{{{labels},le="{bucket:g}"}} {count}'
                )
            count = duration_count[(method, route)]
            lines.append(f'pcbr_http_request_duration_seconds_bucket{{{labels},le="+Inf"}} {count}')
            lines.append(
                f"pcbr_http_request_duration_seconds_sum{{{labels}}} "
                f"{duration_sum[(method, route)]:.9f}"
            )
            lines.append(f"pcbr_http_request_duration_seconds_count{{{labels}}} {count}")
        return "\n".join(lines) + "\n"


class DomainMetrics:
    """Low-cardinality outcome metrics for the recommendation domain.

    Request IDs, product IDs, query terms, model versions, and human-readable failure
    messages are deliberately not labels.  All labels are small closed enums so a public
    endpoint cannot make the process retain an unbounded number of time series.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._build_generations: defaultdict[tuple[str, str, str], int] = defaultdict(int)
        self._builds_returned: defaultdict[str, int] = defaultdict(int)
        self._optimizer_profile_outcomes: defaultdict[tuple[str, str], int] = defaultdict(int)
        self._optimizer_validator_rejections = 0
        self._component_replacements: defaultdict[tuple[str, str], int] = defaultdict(int)
        self._product_search_requests = 0
        self._product_search_results = 0
        self._product_search_candidates: defaultdict[str, int] = defaultdict(int)
        self._product_search_filtered: defaultdict[str, int] = defaultdict(int)
        self._product_search_empty = 0
        self._performance_signals: defaultdict[tuple[str, str, str], int] = defaultdict(int)
        self._performance_fallbacks: defaultdict[str, int] = defaultdict(int)
        self._compatibility_requests: defaultdict[tuple[str, str], int] = defaultdict(int)
        self._compatibility_checks: defaultdict[str, int] = defaultdict(int)
        self._interaction_events: defaultdict[str, int] = defaultdict(int)
        self._freshness_observations: defaultdict[tuple[str, str], int] = defaultdict(int)
        self._catalogue_freshness_status = "degraded"
        self._catalogue_production_ready = False
        self._freshness_probe_success = False
        self._catalogue_products = 0
        self._catalogue_listings = 0
        self._catalogue_release_blockers = 0
        self._release_artifact_verification = "not_verified"
        self._entity_resolution_mapping_state = "not_observed"
        self._entity_resolution_manual_review_queue_items: int | None = None
        self._entity_resolution_unmatched_offer_items: int | None = None
        self._entity_resolution_rejected_conflict_items: int | None = None
        self._entity_resolution_model_rejected_items: int | None = None
        self._catalogue_missing_critical_field_values: int | None = None
        self._pipeline_operations_state = "not_observed"
        self._pipeline_operation_failures_in_window: int | None = None
        self._pipeline_operation_invalid_receipts_in_window: int | None = None
        self._pipeline_operation_window_hours: int | None = None
        self._pipeline_operation_receipts_truncated: bool | None = None

    def record_build_generation(
        self,
        *,
        outcome: str,
        solver_status: str,
        solver_ran: bool,
        build_count: int,
        validator_rejections: int,
        profile_statuses: tuple[tuple[str, str], ...],
    ) -> None:
        outcome = _require_label(outcome, _BUILD_OUTCOMES, "build outcome")
        solver_status = _require_label(solver_status, _SOLVER_STATUSES, "solver status")
        if build_count < 0 or validator_rejections < 0:
            raise ValueError("build metric counts must be non-negative")
        validated_profiles = tuple(
            (
                _require_label(profile, _BUILD_PROFILES, "build profile"),
                _require_label(status, _SOLVER_STATUSES, "solver status"),
            )
            for profile, status in profile_statuses
        )
        with self._lock:
            self._build_generations[(outcome, solver_status, _boolean_label(solver_ran))] += 1
            self._builds_returned[outcome] += build_count
            self._optimizer_validator_rejections += validator_rejections
            for profile, status in validated_profiles:
                self._optimizer_profile_outcomes[(profile, status)] += 1

    def record_component_replacement(self, *, solver_status: str, solver_ran: bool) -> None:
        solver_status = _require_label(solver_status, _SOLVER_STATUSES, "solver status")
        with self._lock:
            self._component_replacements[(solver_status, _boolean_label(solver_ran))] += 1

    def record_product_search(
        self,
        *,
        result_count: int,
        ranked_candidates: int,
        retrieved_candidates: int,
        filtered_category: int,
        filtered_brand: int,
        filtered_incompatible: int,
        filtered_unknown: int,
    ) -> None:
        counts = (
            result_count,
            ranked_candidates,
            retrieved_candidates,
            filtered_category,
            filtered_brand,
            filtered_incompatible,
            filtered_unknown,
        )
        if min(counts) < 0:
            raise ValueError("product-search metric counts must be non-negative")
        after_category_filter = retrieved_candidates - filtered_category
        after_brand_filter = after_category_filter - filtered_brand
        if after_category_filter < 0 or after_brand_filter < 0:
            raise ValueError("product-search filters cannot exceed retrieved candidates")
        if ranked_candidates > after_brand_filter:
            raise ValueError("ranked candidates cannot exceed the candidate funnel")
        if result_count > ranked_candidates:
            raise ValueError("returned products cannot exceed ranked candidates")
        with self._lock:
            self._product_search_requests += 1
            self._product_search_results += result_count
            self._product_search_candidates["retrieved"] += retrieved_candidates
            self._product_search_candidates["after_category_filter"] += after_category_filter
            self._product_search_candidates["after_brand_filter"] += after_brand_filter
            self._product_search_candidates["ranked"] += ranked_candidates
            self._product_search_filtered["category"] += filtered_category
            self._product_search_filtered["brand"] += filtered_brand
            self._product_search_filtered["incompatible"] += filtered_incompatible
            self._product_search_filtered["unknown"] += filtered_unknown
            if result_count == 0:
                self._product_search_empty += 1

    def record_performance_signals(
        self,
        *,
        signals: tuple[tuple[str, str | None, str | None], ...],
    ) -> None:
        """Record only the bounded provenance visible in validated build responses.

        This is response evidence rather than an inference-call counter: a diversified
        response can legitimately contain several signals for one product or workload.
        ``relative`` and ``insufficient_data`` are explicit API fallback states.
        Their closed decision code distinguishes model-promotion policy from an
        input outside the training contract without exposing raw feature values.
        """

        validated = tuple(
            (
                _require_label(basis, _PERFORMANCE_SIGNAL_BASES, "performance basis"),
                _require_label(
                    "not_applicable" if confidence is None else confidence,
                    _PERFORMANCE_SIGNAL_CONFIDENCES,
                    "performance confidence",
                ),
                _require_label(
                    "not_applicable" if decision is None else decision,
                    _PERFORMANCE_SIGNAL_DECISIONS,
                    "performance decision",
                ),
            )
            for basis, confidence, decision in signals
        )
        with self._lock:
            for basis, confidence, decision in validated:
                self._performance_signals[(basis, confidence, decision)] += 1
                if basis in {"relative", "insufficient_data"}:
                    self._performance_fallbacks[decision] += 1

    def record_compatibility(
        self,
        *,
        status: str,
        is_feasible: bool,
        check_statuses: tuple[str, ...],
    ) -> None:
        status = _require_label(status, _COMPATIBILITY_STATUSES, "compatibility status")
        validated_check_statuses = tuple(
            _require_label(item, _COMPATIBILITY_STATUSES, "compatibility status")
            for item in check_statuses
        )
        with self._lock:
            self._compatibility_requests[(status, _boolean_label(is_feasible))] += 1
            for item in validated_check_statuses:
                self._compatibility_checks[item] += 1

    def record_interaction(self, *, event_type: str) -> None:
        event_type = _require_label(event_type, _INTERACTION_EVENT_TYPES, "interaction event")
        with self._lock:
            self._interaction_events[event_type] += 1

    def record_freshness(
        self,
        *,
        status: str,
        production_ready: bool,
        product_count: int,
        listing_count: int,
        release_blocker_count: int,
        release_artifact_verification: str,
    ) -> None:
        status = _require_label(status, _FRESHNESS_STATUSES, "freshness status")
        release_artifact_verification = _require_label(
            release_artifact_verification,
            _RELEASE_ARTIFACT_VERIFICATIONS,
            "release artifact verification",
        )
        if min(product_count, listing_count, release_blocker_count) < 0:
            raise ValueError("freshness metric counts must be non-negative")
        with self._lock:
            self._freshness_observations[(status, _boolean_label(production_ready))] += 1
            self._catalogue_freshness_status = status
            self._catalogue_production_ready = production_ready
            self._freshness_probe_success = True
            self._catalogue_products = product_count
            self._catalogue_listings = listing_count
            self._catalogue_release_blockers = release_blocker_count
            self._release_artifact_verification = release_artifact_verification

    def record_freshness_probe_failure(self) -> None:
        """Fail closed while preserving the last known catalogue cardinalities."""

        with self._lock:
            self._freshness_observations[("degraded", "false")] += 1
            self._catalogue_freshness_status = "degraded"
            self._catalogue_production_ready = False
            self._freshness_probe_success = False
            self._catalogue_release_blockers = max(self._catalogue_release_blockers, 1)

    def record_admin_operations(
        self,
        *,
        entity_resolution_mapping_available: bool,
        manual_review_count: int | None,
        unmatched_offer_count: int | None,
        rejected_conflict_count: int | None,
        model_rejected_count: int | None,
        missing_critical_field_value_count: int | None,
        pipeline_receipts_available: bool,
        pipeline_failed_count: int | None,
        pipeline_invalid_receipt_count: int | None,
        pipeline_window_hours: int | None,
        pipeline_receipts_truncated: bool | None,
    ) -> None:
        """Record only the bounded aggregate evidence from an admin operations response."""

        mapping_counts = (
            manual_review_count,
            unmatched_offer_count,
            rejected_conflict_count,
            model_rejected_count,
            missing_critical_field_value_count,
        )
        if entity_resolution_mapping_available:
            if (
                manual_review_count is None
                or unmatched_offer_count is None
                or rejected_conflict_count is None
                or model_rejected_count is None
                or missing_critical_field_value_count is None
            ):
                raise ValueError(
                    "available entity-resolution mapping metrics require complete aggregate counts"
                )
            if (
                min(
                    manual_review_count,
                    unmatched_offer_count,
                    rejected_conflict_count,
                    model_rejected_count,
                    missing_critical_field_value_count,
                )
                < 0
            ):
                raise ValueError("entity-resolution mapping metric counts must be non-negative")
        elif any(value is not None for value in mapping_counts):
            raise ValueError(
                "unavailable entity-resolution mapping metrics cannot include aggregate counts"
            )

        pipeline_counts = (
            pipeline_failed_count,
            pipeline_invalid_receipt_count,
            pipeline_window_hours,
            pipeline_receipts_truncated,
        )
        if pipeline_receipts_available:
            if (
                pipeline_failed_count is None
                or pipeline_invalid_receipt_count is None
                or pipeline_window_hours is None
                or pipeline_receipts_truncated is None
            ):
                raise ValueError(
                    "available pipeline metrics require complete aggregate receipt evidence"
                )
            if min(pipeline_failed_count, pipeline_invalid_receipt_count) < 0:
                raise ValueError("pipeline metric counts must be non-negative")
            if pipeline_window_hours < 1:
                raise ValueError("pipeline metric window must be positive")
        elif any(value is not None for value in pipeline_counts):
            raise ValueError(
                "unavailable pipeline metrics cannot include aggregate receipt evidence"
            )

        with self._lock:
            self._entity_resolution_mapping_state = (
                "available" if entity_resolution_mapping_available else "unavailable"
            )
            self._entity_resolution_manual_review_queue_items = manual_review_count
            self._entity_resolution_unmatched_offer_items = unmatched_offer_count
            self._entity_resolution_rejected_conflict_items = rejected_conflict_count
            self._entity_resolution_model_rejected_items = model_rejected_count
            self._catalogue_missing_critical_field_values = missing_critical_field_value_count
            self._pipeline_operations_state = (
                "available" if pipeline_receipts_available else "unavailable"
            )
            self._pipeline_operation_failures_in_window = pipeline_failed_count
            self._pipeline_operation_invalid_receipts_in_window = pipeline_invalid_receipt_count
            self._pipeline_operation_window_hours = pipeline_window_hours
            self._pipeline_operation_receipts_truncated = pipeline_receipts_truncated

    def render(self) -> str:
        """Render domain metrics from a single internally consistent snapshot."""

        with self._lock:
            build_generations = dict(self._build_generations)
            builds_returned = dict(self._builds_returned)
            optimizer_profile_outcomes = dict(self._optimizer_profile_outcomes)
            optimizer_validator_rejections = self._optimizer_validator_rejections
            component_replacements = dict(self._component_replacements)
            product_search_requests = self._product_search_requests
            product_search_results = self._product_search_results
            product_search_candidates = dict(self._product_search_candidates)
            product_search_filtered = dict(self._product_search_filtered)
            product_search_empty = self._product_search_empty
            performance_signals = dict(self._performance_signals)
            performance_fallbacks = dict(self._performance_fallbacks)
            compatibility_requests = dict(self._compatibility_requests)
            compatibility_checks = dict(self._compatibility_checks)
            interaction_events = dict(self._interaction_events)
            freshness_observations = dict(self._freshness_observations)
            catalogue_freshness_status = self._catalogue_freshness_status
            catalogue_production_ready = self._catalogue_production_ready
            freshness_probe_success = self._freshness_probe_success
            catalogue_products = self._catalogue_products
            catalogue_listings = self._catalogue_listings
            catalogue_release_blockers = self._catalogue_release_blockers
            release_artifact_verification = self._release_artifact_verification
            entity_resolution_mapping_state = self._entity_resolution_mapping_state
            entity_resolution_manual_review_queue_items = (
                self._entity_resolution_manual_review_queue_items
            )
            entity_resolution_unmatched_offer_items = self._entity_resolution_unmatched_offer_items
            entity_resolution_rejected_conflict_items = (
                self._entity_resolution_rejected_conflict_items
            )
            entity_resolution_model_rejected_items = self._entity_resolution_model_rejected_items
            catalogue_missing_critical_field_values = self._catalogue_missing_critical_field_values
            pipeline_operations_state = self._pipeline_operations_state
            pipeline_operation_failures_in_window = self._pipeline_operation_failures_in_window
            pipeline_operation_invalid_receipts_in_window = (
                self._pipeline_operation_invalid_receipts_in_window
            )
            pipeline_operation_window_hours = self._pipeline_operation_window_hours
            pipeline_operation_receipts_truncated = self._pipeline_operation_receipts_truncated

        lines = [
            "# HELP pcbr_build_generation_total Complete build-generation attempts by outcome.",
            "# TYPE pcbr_build_generation_total counter",
        ]
        for (outcome, solver_status, solver_ran), count in sorted(build_generations.items()):
            lines.append(
                "pcbr_build_generation_total{"
                f'outcome="{outcome}",solver_status="{solver_status}",solver_ran="{solver_ran}"'
                f"}} {count}"
            )
        lines.extend(
            [
                "# HELP pcbr_builds_returned_total Complete builds returned by outcome.",
                "# TYPE pcbr_builds_returned_total counter",
            ]
        )
        for outcome, count in sorted(builds_returned.items()):
            lines.append(f'pcbr_builds_returned_total{{outcome="{outcome}"}} {count}')
        lines.extend(
            [
                "# HELP pcbr_optimizer_profile_outcomes_total Optimizer profile outcomes.",
                "# TYPE pcbr_optimizer_profile_outcomes_total counter",
            ]
        )
        for (profile, status), count in sorted(optimizer_profile_outcomes.items()):
            lines.append(
                "pcbr_optimizer_profile_outcomes_total{"
                f'profile="{profile}",status="{status}"'
                f"}} {count}"
            )
        lines.extend(
            [
                "# HELP pcbr_optimizer_validator_rejections_total Candidates rejected by the "
                "independent optimizer validator.",
                "# TYPE pcbr_optimizer_validator_rejections_total counter",
                f"pcbr_optimizer_validator_rejections_total {optimizer_validator_rejections}",
                "# HELP pcbr_component_replacements_total Successful component-replacement "
                "responses.",
                "# TYPE pcbr_component_replacements_total counter",
            ]
        )
        for (solver_status, solver_ran), count in sorted(component_replacements.items()):
            lines.append(
                "pcbr_component_replacements_total{"
                f'solver_status="{solver_status}",solver_ran="{solver_ran}"'
                f"}} {count}"
            )
        lines.extend(
            [
                "# HELP pcbr_product_search_requests_total Successful product-search requests.",
                "# TYPE pcbr_product_search_requests_total counter",
                f"pcbr_product_search_requests_total {product_search_requests}",
                "# HELP pcbr_product_search_results_total Products returned by successful "
                "searches.",
                "# TYPE pcbr_product_search_results_total counter",
                f"pcbr_product_search_results_total {product_search_results}",
                "# HELP pcbr_product_search_candidates_total Candidate observations at "
                "bounded search stages before pagination.",
                "# TYPE pcbr_product_search_candidates_total counter",
            ]
        )
        for stage, count in sorted(product_search_candidates.items()):
            lines.append(f'pcbr_product_search_candidates_total{{stage="{stage}"}} {count}')
        lines.extend(
            [
                "# HELP pcbr_product_search_filtered_total Candidates removed by authoritative "
                "request or compatibility filters.",
                "# TYPE pcbr_product_search_filtered_total counter",
            ]
        )
        for reason, count in sorted(product_search_filtered.items()):
            lines.append(f'pcbr_product_search_filtered_total{{reason="{reason}"}} {count}')
        lines.extend(
            [
                "# HELP pcbr_product_search_empty_total Successful searches with no returned "
                "products.",
                "# TYPE pcbr_product_search_empty_total counter",
                f"pcbr_product_search_empty_total {product_search_empty}",
                "# HELP pcbr_performance_signals_total Performance provenance shown in "
                "validated build responses.",
                "# TYPE pcbr_performance_signals_total counter",
            ]
        )
        for (basis, confidence, decision), count in sorted(performance_signals.items()):
            lines.append(
                "pcbr_performance_signals_total{"
                f'basis="{basis}",confidence="{confidence}",decision="{decision}"'
                f"}} {count}"
            )
        lines.extend(
            [
                "# HELP pcbr_performance_fallbacks_total Explicit serving-visible performance "
                "fallbacks.",
                "# TYPE pcbr_performance_fallbacks_total counter",
            ]
        )
        for decision, count in sorted(performance_fallbacks.items()):
            lines.append(f'pcbr_performance_fallbacks_total{{decision="{decision}"}} {count}')
        lines.extend(
            [
                "# HELP pcbr_compatibility_requests_total Successful compatibility checks by "
                "final status.",
                "# TYPE pcbr_compatibility_requests_total counter",
            ]
        )
        for (status, feasible), count in sorted(compatibility_requests.items()):
            lines.append(
                "pcbr_compatibility_requests_total{"
                f'status="{status}",feasible="{feasible}"'
                f"}} {count}"
            )
        lines.extend(
            [
                "# HELP pcbr_compatibility_checks_total Individual compatibility-rule results.",
                "# TYPE pcbr_compatibility_checks_total counter",
            ]
        )
        for status, count in sorted(compatibility_checks.items()):
            lines.append(f'pcbr_compatibility_checks_total{{status="{status}"}} {count}')
        lines.extend(
            [
                "# HELP pcbr_interaction_events_total Accepted product and build interaction "
                "events.",
                "# TYPE pcbr_interaction_events_total counter",
            ]
        )
        for event_type, count in sorted(interaction_events.items()):
            lines.append(f'pcbr_interaction_events_total{{event_type="{event_type}"}} {count}')
        lines.extend(
            [
                "# HELP pcbr_catalogue_freshness_observations_total Freshness reads by status "
                "and production authorization.",
                "# TYPE pcbr_catalogue_freshness_observations_total counter",
            ]
        )
        for (status, production_ready), count in sorted(freshness_observations.items()):
            lines.append(
                "pcbr_catalogue_freshness_observations_total{"
                f'status="{status}",production_ready="{production_ready}"'
                f"}} {count}"
            )
        lines.extend(
            [
                "# HELP pcbr_catalogue_freshness_status Latest measured catalogue freshness state.",
                "# TYPE pcbr_catalogue_freshness_status gauge",
            ]
        )
        for freshness_status in sorted(_FRESHNESS_STATUSES):
            lines.append(
                "pcbr_catalogue_freshness_status{"
                f'status="{freshness_status}"'
                f"}} {int(freshness_status == catalogue_freshness_status)}"
            )
        lines.extend(
            [
                "# HELP pcbr_catalogue_production_ready Whether the latest freshness and "
                "release evidence authorizes production use.",
                "# TYPE pcbr_catalogue_production_ready gauge",
                f"pcbr_catalogue_production_ready {int(catalogue_production_ready)}",
                "# HELP pcbr_catalogue_freshness_probe_success Whether the latest internal "
                "freshness probe completed.",
                "# TYPE pcbr_catalogue_freshness_probe_success gauge",
                f"pcbr_catalogue_freshness_probe_success {int(freshness_probe_success)}",
                "# HELP pcbr_catalogue_products Latest catalogue product count reported by "
                "freshness.",
                "# TYPE pcbr_catalogue_products gauge",
                f"pcbr_catalogue_products {catalogue_products}",
                "# HELP pcbr_catalogue_listings Latest catalogue listing count reported by "
                "freshness.",
                "# TYPE pcbr_catalogue_listings gauge",
                f"pcbr_catalogue_listings {catalogue_listings}",
                "# HELP pcbr_catalogue_release_blockers Latest production-release blocker count.",
                "# TYPE pcbr_catalogue_release_blockers gauge",
                f"pcbr_catalogue_release_blockers {catalogue_release_blockers}",
                "# HELP pcbr_serving_release_artifact_verification Latest immutable serving "
                "release startup-verification state.",
                "# TYPE pcbr_serving_release_artifact_verification gauge",
                "pcbr_serving_release_artifact_verification{"
                f'status="{release_artifact_verification}"}} 1',
                "# HELP pcbr_entity_resolution_mapping_observation Latest validated mapping "
                "queue observation state.",
                "# TYPE pcbr_entity_resolution_mapping_observation gauge",
                "pcbr_entity_resolution_mapping_observation{"
                f'state="{entity_resolution_mapping_state}"}} 1',
                "# HELP pcbr_pipeline_operations_observation Latest validated instrumented "
                "pipeline receipt observation state.",
                "# TYPE pcbr_pipeline_operations_observation gauge",
                f'pcbr_pipeline_operations_observation{{state="{pipeline_operations_state}"}} 1',
            ]
        )
        if entity_resolution_mapping_state == "available":
            assert entity_resolution_manual_review_queue_items is not None
            assert entity_resolution_unmatched_offer_items is not None
            assert entity_resolution_rejected_conflict_items is not None
            assert entity_resolution_model_rejected_items is not None
            assert catalogue_missing_critical_field_values is not None
            lines.extend(
                [
                    "# HELP pcbr_entity_resolution_manual_review_queue_items Latest unresolved "
                    "manual-review mapping items.",
                    "# TYPE pcbr_entity_resolution_manual_review_queue_items gauge",
                    "pcbr_entity_resolution_manual_review_queue_items "
                    f"{entity_resolution_manual_review_queue_items}",
                    "# HELP pcbr_entity_resolution_unmatched_offer_items Latest unmatched "
                    "retailer offers.",
                    "# TYPE pcbr_entity_resolution_unmatched_offer_items gauge",
                    "pcbr_entity_resolution_unmatched_offer_items "
                    f"{entity_resolution_unmatched_offer_items}",
                    "# HELP pcbr_entity_resolution_rejected_conflict_items Latest mapping "
                    "conflicts rejected by hard rules.",
                    "# TYPE pcbr_entity_resolution_rejected_conflict_items gauge",
                    "pcbr_entity_resolution_rejected_conflict_items "
                    f"{entity_resolution_rejected_conflict_items}",
                    "# HELP pcbr_entity_resolution_model_rejected_items Latest candidate "
                    "mappings rejected by the model policy.",
                    "# TYPE pcbr_entity_resolution_model_rejected_items gauge",
                    "pcbr_entity_resolution_model_rejected_items "
                    f"{entity_resolution_model_rejected_items}",
                    "# HELP pcbr_catalogue_missing_critical_field_values Latest aggregate "
                    "critical-field values missing from the catalogue.",
                    "# TYPE pcbr_catalogue_missing_critical_field_values gauge",
                    "pcbr_catalogue_missing_critical_field_values "
                    f"{catalogue_missing_critical_field_values}",
                ]
            )
        if pipeline_operations_state == "available":
            assert pipeline_operation_failures_in_window is not None
            assert pipeline_operation_invalid_receipts_in_window is not None
            assert pipeline_operation_window_hours is not None
            assert pipeline_operation_receipts_truncated is not None
            lines.extend(
                [
                    "# HELP pcbr_pipeline_operation_failures_in_window Latest failed "
                    "instrumented pipeline operations in the configured receipt window.",
                    "# TYPE pcbr_pipeline_operation_failures_in_window gauge",
                    "pcbr_pipeline_operation_failures_in_window "
                    f"{pipeline_operation_failures_in_window}",
                    "# HELP pcbr_pipeline_operation_invalid_receipts_in_window Latest invalid "
                    "pipeline receipts excluded from the aggregate window.",
                    "# TYPE pcbr_pipeline_operation_invalid_receipts_in_window gauge",
                    "pcbr_pipeline_operation_invalid_receipts_in_window "
                    f"{pipeline_operation_invalid_receipts_in_window}",
                    "# HELP pcbr_pipeline_operation_receipt_window_hours Configured operation "
                    "receipt observation window.",
                    "# TYPE pcbr_pipeline_operation_receipt_window_hours gauge",
                    "pcbr_pipeline_operation_receipt_window_hours "
                    f"{pipeline_operation_window_hours}",
                    "# HELP pcbr_pipeline_operation_receipts_truncated Whether the bounded "
                    "receipt scan omitted older events.",
                    "# TYPE pcbr_pipeline_operation_receipts_truncated gauge",
                    "pcbr_pipeline_operation_receipts_truncated "
                    f"{1 if pipeline_operation_receipts_truncated else 0}",
                ]
            )
        return "\n".join(lines) + "\n"


REQUEST_METRICS = RequestMetrics()
DOMAIN_METRICS = DomainMetrics()


def record_admin_operations_response(response: AdminOperationsResponse) -> None:
    """Copy a validated admin aggregate response into bounded process-local gauges."""

    mapping_queue = response.mapping_queue
    pipeline_operations = response.pipeline_operations
    DOMAIN_METRICS.record_admin_operations(
        entity_resolution_mapping_available=mapping_queue is not None,
        manual_review_count=(mapping_queue.manual_review_count if mapping_queue else None),
        unmatched_offer_count=(mapping_queue.unmatched_count if mapping_queue else None),
        rejected_conflict_count=(mapping_queue.rejected_conflict_count if mapping_queue else None),
        model_rejected_count=(mapping_queue.model_rejected_count if mapping_queue else None),
        missing_critical_field_value_count=(
            sum(item.missing_product_count for item in response.missing_critical_fields)
            if mapping_queue
            else None
        ),
        pipeline_receipts_available=response.pipeline_failure_events_available,
        pipeline_failed_count=(pipeline_operations.failed_count if pipeline_operations else None),
        pipeline_invalid_receipt_count=(
            pipeline_operations.invalid_receipt_count if pipeline_operations else None
        ),
        pipeline_window_hours=(
            pipeline_operations.event_window_hours if pipeline_operations else None
        ),
        pipeline_receipts_truncated=(
            pipeline_operations.truncated if pipeline_operations else None
        ),
    )
