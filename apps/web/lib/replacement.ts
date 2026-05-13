import type {
  BuildResult,
  CompatVerdict,
  ComponentKind,
  ProductSearchItem,
  ReplacementCandidate,
  ReplacementOption,
  ReplacementResponse,
} from "./types";

export interface ReplacementChangeSummary {
  changedCategories: ComponentKind[];
  priceDeltaSgd: number;
  powerDeltaW: number | null;
  workloadScoreDeltas: Array<[string, number]>;
}

export function summarizeReplacementChange(
  previousBuild: Pick<BuildResult, "estimated_peak_power_w">,
  response: Pick<
    ReplacementResponse,
    "build" | "changed_categories" | "price_delta_sgd" | "workload_score_deltas"
  >,
): ReplacementChangeSummary {
  const previousPower = previousBuild.estimated_peak_power_w;
  const nextPower = response.build.estimated_peak_power_w;
  return {
    changedCategories: [...new Set(response.changed_categories)],
    priceDeltaSgd: response.price_delta_sgd,
    powerDeltaW:
      typeof previousPower === "number" && typeof nextPower === "number"
        ? nextPower - previousPower
        : null,
    workloadScoreDeltas: Object.entries(response.workload_score_deltas),
  };
}

export function replacementCandidateStatus(
  candidate: Pick<ReplacementCandidate, "compatibility_status">,
): CompatVerdict {
  return candidate.compatibility_status ?? "unknown";
}

export function canApplyReplacementCandidate(
  candidate: Pick<ReplacementCandidate, "compatibility_status">,
): boolean {
  const status = replacementCandidateStatus(candidate);
  return status === "pass" || status === "warning";
}

export function firstApplicableCandidateId(
  candidates: Array<Pick<ReplacementCandidate, "product_id" | "compatibility_status">>,
): string {
  return candidates.find(canApplyReplacementCandidate)?.product_id ?? "";
}

export function replacementStatusLabel(status: CompatVerdict): string {
  const labels: Record<CompatVerdict, string> = {
    pass: "Checked",
    warning: "Warning",
    unknown: "Not verified",
    fail: "Incompatible",
  };
  return labels[status];
}

export function productSearchItemToReplacementCandidate(
  product: ProductSearchItem,
): ReplacementOption {
  return {
    product_id: product.product_id,
    canonical_name: product.canonical_name,
    category: product.category,
    price_sgd: product.lowest_price_sgd ?? null,
    compatibility_status: product.compatibility_status ?? "unknown",
  };
}
