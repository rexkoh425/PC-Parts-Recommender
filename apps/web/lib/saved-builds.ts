import { componentCategories } from "./types";
import type { BuildComponent, BuildProfile, BuildSummary } from "./types";

export const SAVED_BUILDS_KEY = "pc-build-recommender:saved-builds:v1";

export interface SavedBuild {
  build: BuildSummary;
  saved_at: string;
}

const buildProfiles = new Set<BuildProfile>([
  "best_overall",
  "best_value",
  "highest_performance",
  "most_upgradeable",
  "lowest_power",
]);
const componentCategorySet = new Set<string>(componentCategories);

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

function isValidTimestamp(value: unknown): value is string {
  return isNonEmptyString(value) && Number.isFinite(Date.parse(value));
}

function isBuildComponent(value: unknown): value is BuildComponent {
  if (!value || typeof value !== "object") return false;
  const component = value as Record<string, unknown>;
  return (
    isNonEmptyString(component.category) &&
    componentCategorySet.has(component.category) &&
    isNonEmptyString(component.product_id) &&
    isNonEmptyString(component.canonical_name) &&
    isFiniteNumber(component.price_sgd) &&
    component.price_sgd >= 0
  );
}

function isWorkloadScores(value: unknown): value is Record<string, number | null> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  return Object.entries(value).every(
    ([name, score]) => isNonEmptyString(name) && (score === null || isFiniteNumber(score)),
  );
}

function isOptionalFiniteNumber(value: unknown): boolean {
  return value === undefined || isFiniteNumber(value);
}

function isBuildSummary(value: unknown): value is BuildSummary {
  if (!value || typeof value !== "object") return false;
  const build = value as Record<string, unknown>;
  return (
    isNonEmptyString(build.build_id) &&
    isNonEmptyString(build.profile) &&
    buildProfiles.has(build.profile as BuildProfile) &&
    isFiniteNumber(build.total_price_sgd) &&
    build.total_price_sgd >= 0 &&
    isFiniteNumber(build.overall_score) &&
    isOptionalFiniteNumber(build.value_score) &&
    isOptionalFiniteNumber(build.upgradeability_score) &&
    isOptionalFiniteNumber(build.efficiency_score) &&
    isOptionalFiniteNumber(build.estimated_peak_power_w) &&
    isWorkloadScores(build.workload_scores) &&
    (build.compatibility_status === "pass" || build.compatibility_status === "warning") &&
    Array.isArray(build.components) &&
    build.components.every(isBuildComponent) &&
    isValidTimestamp(build.generated_at) &&
    isNonEmptyString(build.data_version) &&
    isNonEmptyString(build.ranking_model) &&
    isNonEmptyString(build.rule_version) &&
    isNonEmptyString(build.solver_version) &&
    isNonEmptyString(build.solver_status)
  );
}

export function parseSavedBuilds(value: string | null): SavedBuild[] {
  if (!value) return [];
  try {
    const parsed: unknown = JSON.parse(value);
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter((entry): entry is SavedBuild => {
        if (!entry || typeof entry !== "object") return false;
        const candidate = entry as Record<string, unknown>;
        return isBuildSummary(candidate.build) && isValidTimestamp(candidate.saved_at);
      })
      .map((entry) => ({ ...entry, build: withoutImpressionTokens(entry.build) }));
  } catch {
    return [];
  }
}

export function readSavedBuilds(): SavedBuild[] {
  if (typeof window === "undefined") return [];
  return parseSavedBuilds(window.localStorage.getItem(SAVED_BUILDS_KEY));
}

export function writeSavedBuilds(builds: SavedBuild[]): void {
  if (typeof window === "undefined") return;
  const sanitized = builds.map((entry) => ({
    ...entry,
    build: withoutImpressionTokens(entry.build),
  }));
  window.localStorage.setItem(SAVED_BUILDS_KEY, JSON.stringify(sanitized));
  window.dispatchEvent(new Event("pcbr:saved-builds-changed"));
}

export function saveBuild(build: BuildSummary): SavedBuild[] {
  const current = readSavedBuilds().filter((entry) => entry.build.build_id !== build.build_id);
  const next = [
    { build: withoutImpressionTokens(build), saved_at: new Date().toISOString() },
    ...current,
  ];
  writeSavedBuilds(next);
  return next;
}

export function withoutImpressionTokens(build: BuildSummary): BuildSummary {
  return {
    ...build,
    impression_token: undefined,
    components: Array.isArray(build.components)
      ? build.components.map((component) => ({
          ...component,
          impression_token: undefined,
        }))
      : [],
  };
}

export function removeSavedBuild(buildId: string): SavedBuild[] {
  const next = readSavedBuilds().filter((entry) => entry.build.build_id !== buildId);
  writeSavedBuilds(next);
  return next;
}
