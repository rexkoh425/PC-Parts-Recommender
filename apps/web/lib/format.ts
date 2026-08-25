import type {
  BuildProfile,
  ComponentCategory,
  FreshnessSummary,
  WorkloadName,
} from "./types";

/*
 * en-SG renders SGD as a bare "$", which reads as US dollars to most visitors
 * and left the page showing "$2,672" beside copy that said "S$0". The symbol
 * is applied here rather than by Intl so every figure carries the same one.
 */
const sgdNumberFormatter = new Intl.NumberFormat("en-SG", {
  minimumFractionDigits: 0,
  maximumFractionDigits: 2,
});

export function formatSgd(value: number): string {
  return `S$${sgdNumberFormatter.format(Number.isFinite(value) ? value : 0)}`;
}

export function formatScore(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(1) : "—";
}

export function clampScore(value: number | null | undefined): number {
  if (typeof value !== "number" || !Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(100, value));
}

export const profileLabels: Record<BuildProfile, string> = {
  best_overall: "Best overall",
  best_value: "Best value",
  highest_performance: "Highest performance",
  most_upgradeable: "Most upgradeable",
  lowest_power: "Lowest power",
};

export const categoryLabels: Record<ComponentCategory, string> = {
  cpu: "Processor",
  gpu: "Graphics",
  motherboard: "Motherboard",
  memory: "Memory",
  storage: "Storage",
  psu: "Power supply",
  cooler: "CPU cooler",
  case: "Case",
};

export const workloadLabels: Record<WorkloadName, string> = {
  gaming_1080p: "1080p gaming",
  gaming_1440p: "1440p gaming",
  gaming_4k: "4K gaming",
  local_ai: "Local AI",
  software_development: "Software development",
  content_creation: "Content creation",
};

export function humanizeToken(value: string): string {
  return value
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

export function formatFreshness(value?: string | null): string {
  if (!value) return "Freshness unavailable";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Freshness unavailable";

  const elapsed = Date.now() - date.getTime();
  if (elapsed < 60_000) return "Updated just now";
  if (elapsed < 3_600_000) return `Updated ${Math.max(1, Math.floor(elapsed / 60_000))}m ago`;
  if (elapsed < 86_400_000) return `Updated ${Math.floor(elapsed / 3_600_000)}h ago`;
  return `Updated ${Math.floor(elapsed / 86_400_000)}d ago`;
}

export function formatFreshnessSummary(freshness: FreshnessSummary | null): string {
  if (!freshness) return "Checking market data";
  if (freshness.price_status === "stale") {
    return `Prices stale · ${formatFreshness(freshness.prices_updated_at)}`;
  }
  if (freshness.price_status === "degraded") return "Price freshness unavailable";
  if (freshness.catalogue_status === "stale") {
    return `Catalogue stale · ${formatFreshness(freshness.last_catalog_update)}`;
  }
  if (freshness.catalogue_status === "degraded") {
    return "Catalogue freshness unavailable";
  }
  return formatFreshness(freshness.prices_updated_at ?? freshness.last_catalog_update);
}
