import {
  componentCategories,
  type BuildProfile,
  type BuildResult,
  type ComponentCategory,
  type PublicBuildSnapshot,
} from "./types";

const sharedBuildVersion = 1;
const maximumTokenLength = 12_000;
const maximumTextLength = 360;
const maximumSharedExplanations = 4;
const maximumSharedWarnings = 4;
const profileValues = new Set<BuildProfile>([
  "best_overall",
  "best_value",
  "highest_performance",
  "most_upgradeable",
  "lowest_power",
]);
const categoryValues = new Set<ComponentCategory>(componentCategories);

export interface SharedBuildComponent {
  category: ComponentCategory;
  canonical_name: string;
  brand?: string;
  price_sgd: number | null;
  component_score?: number;
  selection_reason?: string;
}

/**
 * A deliberately bounded, public projection of a generated build.
 *
 * It excludes request text, internal identifiers, ownership flags, listing URLs,
 * retailer names, and benchmark/source URLs. The public website can therefore
 * render a portable build snapshot without exposing the originating browser's
 * saved data or treating a link as a fresh market quote.
 */
export interface SharedBuildSnapshot {
  version: typeof sharedBuildVersion;
  generated_at: string;
  profile: BuildProfile;
  total_price_sgd: number;
  overall_score: number;
  value_score?: number;
  upgradeability_score?: number;
  efficiency_score?: number;
  estimated_peak_power_w?: number;
  workload_scores: Record<string, number | null>;
  compatibility_status: "pass" | "warning";
  components: SharedBuildComponent[];
  explanations: string[];
  warnings: string[];
  data_version: string;
  ranking_model: string;
  rule_version: string;
  solver_version: string;
}

function boundedText(value: unknown, maximum = maximumTextLength): string | undefined {
  if (typeof value !== "string") return undefined;
  const normalized = value.trim();
  if (!normalized || normalized.length > maximum) return undefined;
  return normalized;
}

function boundedNumber(value: unknown, minimum = 0, maximum = 1_000_000_000): number | undefined {
  return typeof value === "number" && Number.isFinite(value) && value >= minimum && value <= maximum
    ? value
    : undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function encodeBase64Url(value: string): string {
  const bytes = new TextEncoder().encode(value);
  let binary = "";
  for (let start = 0; start < bytes.length; start += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(start, start + 0x8000));
  }
  return btoa(binary).replaceAll("+", "-").replaceAll("/", "_").replaceAll("=", "");
}

function decodeBase64Url(token: string): string | undefined {
  if (!token || token.length > maximumTokenLength || !/^[A-Za-z0-9_-]+$/.test(token)) return undefined;
  try {
    const padded = token.replaceAll("-", "+").replaceAll("_", "/").padEnd(Math.ceil(token.length / 4) * 4, "=");
    const binary = atob(padded);
    const bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0));
    return new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch {
    return undefined;
  }
}

function shareableExplanations(build: BuildResult): string[] {
  return (build.explanation ?? [])
    .map((item) => (typeof item === "string" ? item : item.text))
    .map((item) => boundedText(item))
    .filter((item): item is string => Boolean(item))
    .slice(0, maximumSharedExplanations);
}

function shareableWarnings(build: BuildResult): string[] {
  return (build.warnings ?? [])
    .filter((item) => item.status === "warning")
    .map((item) => boundedText(item.message))
    .filter((item): item is string => Boolean(item))
    .slice(0, maximumSharedWarnings);
}

export function publicBuildSnapshot(build: BuildResult): SharedBuildSnapshot {
  return {
    version: sharedBuildVersion,
    generated_at: build.generated_at,
    profile: build.profile,
    total_price_sgd: build.total_price_sgd,
    overall_score: build.overall_score,
    ...(typeof build.value_score === "number" ? { value_score: build.value_score } : {}),
    ...(typeof build.upgradeability_score === "number"
      ? { upgradeability_score: build.upgradeability_score }
      : {}),
    ...(typeof build.efficiency_score === "number" ? { efficiency_score: build.efficiency_score } : {}),
    ...(typeof build.estimated_peak_power_w === "number"
      ? { estimated_peak_power_w: build.estimated_peak_power_w }
      : {}),
    workload_scores: Object.fromEntries(
      Object.entries(build.workload_scores ?? {}).filter(
        ([key, value]) => /^[a-z0-9_]{1,64}$/.test(key) && (value === null || boundedNumber(value, 0, 100) !== undefined),
      ),
    ),
    compatibility_status: build.compatibility_status,
    components: build.components.map((component) => ({
      category: component.category,
      canonical_name: component.canonical_name,
      ...(boundedText(component.brand, 120) ? { brand: boundedText(component.brand, 120) } : {}),
      price_sgd: component.already_owned ? null : component.price_sgd,
      ...(typeof component.component_score === "number" ? { component_score: component.component_score } : {}),
      ...(boundedText(component.selection_reasons?.[0])
        ? { selection_reason: boundedText(component.selection_reasons?.[0]) }
        : {}),
    })),
    explanations: shareableExplanations(build),
    warnings: shareableWarnings(build),
    data_version: boundedText(build.data_version, 120) ?? "Not reported",
    ranking_model: boundedText(build.ranking_model, 120) ?? "Not reported",
    rule_version: boundedText(build.rule_version, 120) ?? "Not reported",
    solver_version: boundedText(build.solver_version, 120) ?? "Not reported",
  };
}

export function encodeSharedBuild(build: BuildResult): string {
  return encodeBase64Url(JSON.stringify(publicBuildSnapshot(build)));
}

export function sharedBuildHref(build: BuildResult): string {
  return `/share?build=${encodeURIComponent(encodeSharedBuild(build))}`;
}

/** Convert the server's already allow-listed public projection to the page model. */
export function sharedSnapshotFromApi(snapshot: PublicBuildSnapshot): SharedBuildSnapshot {
  return {
    version: sharedBuildVersion,
    generated_at: snapshot.generated_at,
    profile: snapshot.profile,
    total_price_sgd: snapshot.total_price_sgd,
    overall_score: snapshot.overall_score,
    ...(typeof snapshot.value_score === "number" ? { value_score: snapshot.value_score } : {}),
    ...(typeof snapshot.upgradeability_score === "number"
      ? { upgradeability_score: snapshot.upgradeability_score }
      : {}),
    ...(typeof snapshot.efficiency_score === "number" ? { efficiency_score: snapshot.efficiency_score } : {}),
    ...(typeof snapshot.estimated_peak_power_w === "number"
      ? { estimated_peak_power_w: snapshot.estimated_peak_power_w }
      : {}),
    workload_scores: snapshot.workload_scores,
    compatibility_status: snapshot.compatibility_status,
    components: snapshot.components.map((component) => ({
      category: component.category,
      canonical_name: component.canonical_name,
      ...(component.brand ? { brand: component.brand } : {}),
      price_sgd: component.price_sgd ?? null,
      ...(typeof component.component_score === "number"
        ? { component_score: component.component_score }
        : {}),
      ...(component.selection_reason ? { selection_reason: component.selection_reason } : {}),
    })),
    explanations: snapshot.explanations,
    warnings: snapshot.warnings,
    data_version: snapshot.data_version,
    ranking_model: snapshot.ranking_model,
    rule_version: snapshot.rule_version,
    solver_version: snapshot.solver_version,
  };
}

function parseComponent(value: unknown): SharedBuildComponent | undefined {
  if (!isRecord(value) || typeof value.category !== "string" || !categoryValues.has(value.category as ComponentCategory)) {
    return undefined;
  }
  const canonicalName = boundedText(value.canonical_name);
  const price = value.price_sgd === null ? null : boundedNumber(value.price_sgd);
  if (!canonicalName || price === undefined) return undefined;
  const brand = boundedText(value.brand, 120);
  const componentScore = boundedNumber(value.component_score, 0, 100);
  const selectionReason = boundedText(value.selection_reason);
  return {
    category: value.category as ComponentCategory,
    canonical_name: canonicalName,
    ...(brand ? { brand } : {}),
    price_sgd: price,
    ...(componentScore !== undefined ? { component_score: componentScore } : {}),
    ...(selectionReason ? { selection_reason: selectionReason } : {}),
  };
}

function parseSharedTextList(value: unknown, maximumItems: number): string[] | undefined {
  if (!Array.isArray(value) || value.length > maximumItems) return undefined;
  const items = value.map((item) => boundedText(item));
  return items.every((item) => item !== undefined) ? (items as string[]) : undefined;
}

export function decodeSharedBuild(token: string | null | undefined): SharedBuildSnapshot | undefined {
  if (!token) return undefined;
  const decoded = decodeBase64Url(token);
  if (!decoded || decoded.length > 8_000) return undefined;
  try {
    const value: unknown = JSON.parse(decoded);
    if (!isRecord(value) || value.version !== sharedBuildVersion || typeof value.profile !== "string") {
      return undefined;
    }
    if (!profileValues.has(value.profile as BuildProfile) || !["pass", "warning"].includes(String(value.compatibility_status))) {
      return undefined;
    }
    const generatedAt = boundedText(value.generated_at, 64);
    const dataVersion = boundedText(value.data_version, 120);
    const rankingModel = boundedText(value.ranking_model, 120);
    const ruleVersion = boundedText(value.rule_version, 120);
    const solverVersion = boundedText(value.solver_version, 120);
    const totalPrice = boundedNumber(value.total_price_sgd);
    const overallScore = boundedNumber(value.overall_score, 0, 100);
    if (
      !generatedAt || Number.isNaN(Date.parse(generatedAt)) || !dataVersion || !rankingModel || !ruleVersion || !solverVersion ||
      totalPrice === undefined || overallScore === undefined || !Array.isArray(value.components) || value.components.length !== componentCategories.length
    ) {
      return undefined;
    }
    const components = value.components.map(parseComponent);
    if (components.some((item) => item === undefined) || new Set(components.map((item) => item?.category)).size !== componentCategories.length) {
      return undefined;
    }
    if (!isRecord(value.workload_scores) || Object.keys(value.workload_scores).length > 12) return undefined;
    const workloadScores: Record<string, number | null> = {};
    for (const [key, score] of Object.entries(value.workload_scores)) {
      const parsedScore = score === null ? null : boundedNumber(score, 0, 100);
      if (!/^[a-z0-9_]{1,64}$/.test(key) || parsedScore === undefined) return undefined;
      workloadScores[key] = parsedScore;
    }
    const explanations = parseSharedTextList(value.explanations, maximumSharedExplanations);
    const warnings = parseSharedTextList(value.warnings, maximumSharedWarnings);
    if (!explanations || !warnings) return undefined;
    const optionalScore = (field: string) => boundedNumber(value[field], 0, 100);
    const estimatedPower = boundedNumber(value.estimated_peak_power_w, 0, 10_000);
    return {
      version: sharedBuildVersion,
      generated_at: generatedAt,
      profile: value.profile as BuildProfile,
      total_price_sgd: totalPrice,
      overall_score: overallScore,
      ...(optionalScore("value_score") !== undefined ? { value_score: optionalScore("value_score") } : {}),
      ...(optionalScore("upgradeability_score") !== undefined ? { upgradeability_score: optionalScore("upgradeability_score") } : {}),
      ...(optionalScore("efficiency_score") !== undefined ? { efficiency_score: optionalScore("efficiency_score") } : {}),
      ...(estimatedPower !== undefined ? { estimated_peak_power_w: estimatedPower } : {}),
      workload_scores: workloadScores,
      compatibility_status: value.compatibility_status as "pass" | "warning",
      components: components as SharedBuildComponent[],
      explanations,
      warnings,
      data_version: dataVersion,
      ranking_model: rankingModel,
      rule_version: ruleVersion,
      solver_version: solverVersion,
    };
  } catch {
    return undefined;
  }
}
