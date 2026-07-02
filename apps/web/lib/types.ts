export const componentCategories = [
  "cpu",
  "gpu",
  "motherboard",
  "memory",
  "storage",
  "psu",
  "cooler",
  "case",
] as const;

export type ComponentCategory = (typeof componentCategories)[number];

export type WorkloadName =
  | "gaming_1080p"
  | "gaming_1440p"
  | "gaming_4k"
  | "local_ai"
  | "software_development"
  | "content_creation";

export type CompatibilityStatus = "pass" | "warning" | "unknown" | "fail";

export type BuildProfile =
  | "best_overall"
  | "best_value"
  | "highest_performance"
  | "most_upgradeable"
  | "lowest_power";

export interface ExistingProductInput {
  product_id: string;
  category: ComponentCategory;
  canonical_name?: string;
  include_in_budget: boolean;
}

export interface BuildRequest {
  budget_sgd: number;
  performance_target?: string;
  workloads: Array<{
    name: WorkloadName;
    weight: number;
  }>;
  existing_products: ExistingProductInput[];
  requirements: {
    minimum_gpu_vram_gb?: number;
    minimum_memory_gb?: number;
    storage_gb?: number;
    wifi_required?: boolean;
    case_size?: "small_form_factor" | "mini_tower" | "mid_tower" | "full_tower";
    in_stock_only?: boolean;
  };
  preferences: {
    noise?: "low" | "medium" | "any";
    upgradeability?: "low" | "medium" | "high";
    power_efficiency?: "low" | "medium" | "high";
    preferred_brands: string[];
    excluded_brands: string[];
  };
  max_builds?: number;
  requested_profiles?: BuildProfile[];
}

export interface SourceReference {
  label: string;
  url: string;
}

export type PerformanceDecision =
  | "observed_benchmark"
  | "precise_model_prediction"
  | "model_not_promotion_eligible"
  | "input_outside_training_contract"
  | "model_not_promotion_eligible_and_input_outside_training_contract"
  | "precise_predictions_disabled"
  | "precise_predictions_disabled_and_input_outside_training_contract"
  | "deterministic_baseline";

export interface PerformanceSignal {
  workload: string;
  metric: string;
  value: number | null;
  unit?: string;
  basis: "observed" | "predicted" | "relative" | "insufficient_data";
  confidence?: "high" | "medium" | "low";
  decision?: PerformanceDecision;
  model_version?: string;
  observed_at?: string;
  sources?: SourceReference[];
}

export interface CompatibilityCheck {
  rule_id: string;
  status: CompatibilityStatus;
  message: string;
  affected_categories?: ComponentCategory[];
  evidence_source?: string;
}

export interface ExplanationItem {
  kind: "performance" | "value" | "compatibility" | "preference" | "price";
  text: string;
  supporting_ids?: string[];
}

export interface ReplacementCandidate {
  product_id: string;
  canonical_name: string;
  category: ComponentCategory;
  price_sgd: number;
  retailer?: string;
  performance_delta?: number;
  price_delta_sgd?: number;
  power_delta_w?: number;
  compatibility_status?: CompatibilityStatus;
  reasons?: string[];
}

export interface ReplacementOption extends Omit<ReplacementCandidate, "price_sgd"> {
  price_sgd: number | null;
}

export interface BuildComponent {
  category: ComponentCategory;
  product_id: string;
  listing_id?: string;
  canonical_name: string;
  brand?: string;
  retailer?: string;
  listing_url?: string;
  price_sgd: number;
  already_owned?: boolean;
  component_score?: number;
  selection_reasons?: string[];
  performance_signals?: PerformanceSignal[];
  alternatives?: ReplacementCandidate[];
  impression_token?: string | null;
}

export interface BuildSummary {
  build_id: string;
  request_id?: string;
  profile: BuildProfile;
  total_price_sgd: number;
  overall_score: number;
  value_score?: number;
  upgradeability_score?: number;
  efficiency_score?: number;
  estimated_peak_power_w?: number;
  workload_scores: Record<string, number | null>;
  compatibility_status: "pass" | "warning";
  components: BuildComponent[];
  compatibility_checks?: CompatibilityCheck[];
  warnings?: CompatibilityCheck[];
  explanation?: ExplanationItem[] | string[];
  alternatives?: ReplacementCandidate[];
  generated_at: string;
  data_version: string;
  ranking_model: string;
  rule_version: string;
  solver_version: string;
  solver_status: string;
  impression_token?: string | null;
}

/** Allow-listed snapshot returned from a durable public build-share endpoint. */
export interface PublicBuildComponent {
  category: ComponentCategory;
  canonical_name: string;
  brand?: string | null;
  price_sgd?: number | null;
  component_score?: number | null;
  selection_reason?: string | null;
}

export interface PublicBuildSnapshot {
  profile: BuildProfile;
  total_price_sgd: number;
  overall_score: number;
  value_score?: number | null;
  upgradeability_score?: number | null;
  efficiency_score?: number | null;
  estimated_peak_power_w?: number | null;
  workload_scores: Record<string, number | null>;
  compatibility_status: "pass" | "warning";
  components: PublicBuildComponent[];
  explanations: string[];
  warnings: string[];
  generated_at: string;
  data_version: string;
  ranking_model: string;
  rule_version: string;
  solver_version: string;
}

export interface BuildShareCreated {
  share_id: string;
  /** Store only in the originating browser; it is never part of the public URL. */
  revocation_token: string;
  created_at: string;
  expires_at: string;
}

export interface BuildShareRevoked {
  share_id: string;
  revoked_at: string;
}

export interface PublicBuildShare {
  share_id: string;
  created_at: string;
  expires_at: string;
  snapshot: PublicBuildSnapshot;
}

export interface SuggestedRelaxation {
  field_path: string;
  current_value: unknown;
  proposed_value: unknown;
  expected_effect: string;
}

export interface InfeasibilityExplanation {
  reasons: Array<{
    code: string;
    message: string;
    affected_categories?: ComponentCategory[];
  }>;
  suggested_relaxations?: SuggestedRelaxation[];
}

export interface GenerateBuildsResponse {
  request_id: string;
  status: "complete" | "partial" | "infeasible";
  generated_at: string;
  data_version: string;
  ranking_model: string;
  rule_version: string;
  solver_version: string;
  solver_status: string;
  builds: BuildSummary[];
  infeasibility?: InfeasibilityExplanation | null;
  request?: BuildRequest;
}

export interface ProductSearchItem {
  product_id: string;
  category: ComponentCategory;
  canonical_name: string;
  brand?: string;
  model?: string;
  lowest_price_sgd?: number | null;
  stock_status?: string | null;
  compatibility_status?: CompatibilityStatus | null;
  impression_token?: string | null;
}

export interface ProductSearchRequest {
  query: string;
  category?: ComponentCategory;
  compatible_with_build_id?: string;
  brand?: string;
  in_stock_only?: boolean;
  limit?: number;
  /** Optional fields used by paged catalogue providers. Legacy APIs ignore these by omission. */
  page?: number;
  page_size?: number;
  cursor?: string;
}

export interface ProductFacetCount<TValue extends string = string> {
  value: TValue;
  count: number;
}

export interface ProductSearchFacets {
  categories?: ProductFacetCount<ComponentCategory>[];
  brands?: ProductFacetCount[];
}

export interface ProductSearchPagination {
  page: number;
  page_size: number;
  total_pages: number;
  has_previous: boolean;
  has_next: boolean;
  previous_cursor?: string | null;
  next_cursor?: string | null;
}

export interface CatalogueCoverage {
  canonical_products: number;
  retailer_listings?: number | null;
  source_count?: number | null;
  category_count?: number | null;
  as_of?: string | null;
  scope_label: string;
  source_attributions?: SourceAttribution[];
}

export interface SourceAttribution {
  source_name: string;
  source_url: string;
  licence_or_access_note: string;
  attribution_notice?: string | null;
  licence_url?: string | null;
  retrieved_at: string;
}

export interface ProductSearchResponse {
  /** Stable across pagination for one normalized query and serving release. */
  query_id: string;
  products: ProductSearchItem[];
  total: number;
  filtered_incompatible: number;
  filtered_unknown: number;
  data_version: string;
  retrieval_model: string;
  /** Optional provider metadata. The UI does not infer coverage or facets when absent. */
  facets?: ProductSearchFacets;
  pagination?: ProductSearchPagination;
  coverage?: CatalogueCoverage;
}

export interface ProductDetail extends ProductSearchItem {
  manufacturer_part_number?: string | null;
  attributes: Record<string, unknown>;
  source_confidence?: number | null;
  source_url?: string | null;
  source_attributions?: SourceAttribution[];
  updated_at: string;
  data_version: string;
}

export interface PriceObservation {
  listing_id: string;
  retailer: string;
  observed_at: string;
  base_price_sgd: number;
  shipping_price_sgd: number;
  stock_status: string;
  condition: string;
  current_offer_eligible: boolean;
  listing_url?: string | null;
}

export interface PriceHistoryAnomaly {
  observed_at: string;
  listing_id: string;
  delivered_price_sgd: number;
  direction: "high" | "low";
  modified_z_score?: number | null;
  source_url?: string | null;
}

export interface PriceIntelligenceSummary {
  basis: "descriptive_observed_history";
  currency: string;
  as_of: string;
  current_delivered_price_sgd: number | null;
  median_30d_sgd: number | null;
  median_90d_sgd: number | null;
  percentile_90d: number | null;
  recent_low_90d_sgd: number | null;
  volatility_90d_pct: number | null;
  current_seller_count: number;
  seller_trend: "increasing" | "stable" | "decreasing" | "insufficient_history";
  stock_trend: "increasing" | "stable" | "decreasing" | "insufficient_history";
  history_days_30d: number;
  history_days_90d: number;
  history_sufficient: boolean;
  labels: string[];
  anomalies: PriceHistoryAnomaly[];
  observations_analyzed: number;
  analysis_truncated: boolean;
}

export interface ProductPricesResponse {
  product_id: string;
  current_lowest_price_sgd: number | null;
  observations: PriceObservation[];
  price_intelligence?: PriceIntelligenceSummary | null;
  data_version: string;
}

export interface BenchmarkObservation {
  benchmark_name: string;
  workload: string;
  score: number;
  unit: string;
  higher_is_better: boolean;
  basis: "observed" | "predicted";
  model_version?: string | null;
  source_url?: string | null;
  observed_at?: string | null;
}

export interface ProductBenchmarksResponse {
  product_id: string;
  benchmarks: BenchmarkObservation[];
  data_version: string;
  performance_model_version: string;
}

export interface ReviewEvidence {
  aspect: string;
  sentiment: "positive" | "neutral" | "negative" | "mixed";
  evidence_text: string;
  source_url?: string | null;
  published_at?: string | null;
  confidence: number;
}

export interface ProductReviewsResponse {
  product_id: string;
  evidence: ReviewEvidence[];
  data_version: string;
}

export interface ReplacementRequest {
  category: ComponentCategory;
  replacement_product_id: string;
  mode: "lock_other_components" | "reoptimize_unlocked";
}

export interface ReplacementResponse {
  build: BuildSummary;
  changed_categories: ComponentCategory[];
  price_delta_sgd: number;
  workload_score_deltas: Record<string, number>;
  new_warnings: CompatibilityCheck[];
  data_version: string;
  ranking_model: string;
  rule_version: string;
  solver_version: string;
}

export interface FreshnessSummary {
  data_version: string;
  status: "fresh" | "stale" | "degraded";
  catalogue_status: "fresh" | "stale" | "degraded";
  price_status: "fresh" | "stale" | "degraded";
  last_catalog_update: string | null;
  prices_updated_at: string | null;
  stale_after_hours: number;
  catalogue_stale_after_hours: number;
  price_stale_after_hours: number;
  source_count: number;
  product_count: number;
  listing_count: number;
  production_ready: boolean;
  readiness_blockers: string[];
  catalogue_readiness?: CatalogueReadinessSummary | null;
}

export interface AdminMappingQueue {
  offer_count: number;
  matched_count: number;
  unmatched_count: number;
  manual_review_count: number;
  rejected_conflict_count: number;
  model_rejected_count: number;
}

export interface AdminPriceFreshness {
  snapshot_count: number;
  newest_observed_at?: string | null;
  stale_snapshot_count?: number | null;
  stale_after_hours: number;
}

export interface AdminMissingField {
  category: ComponentCategory;
  field_group: string;
  missing_product_count: number;
  product_count: number;
}

export interface AdminPipelineOperations {
  event_window_hours: number;
  event_count: number;
  succeeded_count: number;
  failed_count: number;
  latest_event_at?: string | null;
  latest_failure_at?: string | null;
  invalid_receipt_count: number;
  truncated: boolean;
}

export interface AdminOperationsResponse {
  data_version: string;
  generated_at: string;
  mode: "demo" | "processed_catalog";
  mapping_queue?: AdminMappingQueue | null;
  price_freshness?: AdminPriceFreshness | null;
  missing_critical_fields: AdminMissingField[];
  release_blockers: string[];
  pipeline_operations?: AdminPipelineOperations | null;
  pipeline_failure_events_available: boolean;
  notes: string[];
}

export interface CatalogueReadinessSummary {
  products_by_category: Record<string, number>;
  compatibility_ready_products_by_category: Record<string, number>;
  matched_listings_by_category: Record<string, number>;
  in_stock_listings_by_category: Record<string, number>;
  offer_count: number;
  mapping_rate: number;
  has_complete_priced_coverage: boolean;
  has_complete_in_stock_coverage: boolean;
  product_provenance_complete_count: number;
  offer_provenance_complete_count: number;
  offer_rights_production_valid_count: number;
  rights_territory: string;
  entity_resolution_model_version?: string | null;
  entity_resolution_model_production_authorized: boolean;
  production_ready: boolean;
  production_blockers: string[];
}

export interface CompatibilityComponent {
  product_id?: string | null;
  category: ComponentCategory;
  canonical_name?: string | null;
  attributes?: Record<string, unknown>;
}

export interface CompatibilityCheckRequest {
  components: CompatibilityComponent[];
}

export interface CompatibilityCheckResponse {
  status: CompatibilityStatus;
  is_feasible: boolean;
  checks: CompatibilityCheck[];
  rule_version: string;
  data_version: string;
}

export interface InteractionEvent {
  event_type:
    | "search_submitted"
    | "build_generated"
    | "build_viewed"
    | "build_saved"
    | "build_shared"
    | "component_viewed"
    | "component_replaced"
    | "comparison_opened"
    | "retailer_clicked"
    | "recommendation_dismissed"
    | "feedback_submitted";
  session_id: string;
  user_id?: string;
  query_id?: string;
  build_id?: string;
  product_id?: string;
  /** One-based display position when present. */
  rank_position?: number;
  model_version?: string;
  data_version?: string;
  rule_version?: string;
  metadata?: Record<string, unknown>;
  impression_token?: string;
}

export interface InteractionAccepted {
  event_id: string;
  accepted_at: string;
  status: "accepted";
  data_version: string;
  rule_version: string;
  trust_level: "verified_impression" | "legacy_untrusted";
  replayed: boolean;
}

export interface ApiErrorDetails {
  checks?: CompatibilityCheck[];
  compatibility_checks?: CompatibilityCheck[];
  reasons?: Array<{ message?: string }>;
  evidence?: string[];
  [key: string]: unknown;
}

export interface ApiErrorEnvelope {
  code: string;
  message: string;
  request_id?: string | null;
  details?: ApiErrorDetails | Array<Record<string, unknown>>;
}

export interface ApiErrorPayload {
  detail?:
    | string
    | Array<{ msg?: string; loc?: Array<string | number> }>
    | ApiErrorDetails;
  message?: string;
  error?: ApiErrorEnvelope;
  reasons?: Array<{ message?: string }>;
  compatibility_checks?: CompatibilityCheck[];
  infeasibility?: InfeasibilityExplanation;
}
