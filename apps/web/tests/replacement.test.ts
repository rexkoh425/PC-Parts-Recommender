import { describe, expect, it } from "vitest";
import {
  canApplyReplacementCandidate,
  firstApplicableCandidateId,
  productSearchItemToReplacementCandidate,
  replacementCandidateStatus,
  replacementStatusLabel,
  summarizeReplacementChange,
} from "../lib/replacement";
import type { BuildSummary, CompatibilityStatus, ReplacementCandidate } from "../lib/types";

function candidate(
  id: string,
  status?: CompatibilityStatus,
): ReplacementCandidate {
  return {
    product_id: id,
    canonical_name: `Candidate ${id}`,
    category: "gpu",
    price_sgd: 1000,
    compatibility_status: status,
  };
}

describe("replacement compatibility boundary", () => {
  it("treats a missing status as unknown rather than pass", () => {
    expect(replacementCandidateStatus(candidate("missing"))).toBe("unknown");
    expect(replacementStatusLabel(replacementCandidateStatus(candidate("missing")))).toBe(
      "Not verified",
    );
    expect(canApplyReplacementCandidate(candidate("missing"))).toBe(false);
  });

  it.each([
    ["pass", true],
    ["warning", true],
    ["unknown", false],
    ["fail", false],
  ] as const)("allows status %s: %s", (status, allowed) => {
    expect(canApplyReplacementCandidate(candidate(status, status))).toBe(allowed);
  });

  it("skips failed and unknown candidates when choosing a default", () => {
    expect(
      firstApplicableCandidateId([
        candidate("failed", "fail"),
        candidate("unknown", "unknown"),
        candidate("warning", "warning"),
      ]),
    ).toBe("warning");
  });

  it.each(["pass", "warning", "unknown", "fail"] as const)(
    "preserves search API status %s",
    (status) => {
      const mapped = productSearchItemToReplacementCandidate({
        product_id: `gpu-${status}`,
        category: "gpu",
        canonical_name: "GPU candidate",
        compatibility_status: status,
      });
      expect(mapped.compatibility_status).toBe(status);
    },
  );

  it("preserves an unavailable search price instead of inventing S$0", () => {
    const mapped = productSearchItemToReplacementCandidate({
      product_id: "gpu-no-price",
      category: "gpu",
      canonical_name: "GPU without an in-stock listing",
      lowest_price_sgd: null,
      compatibility_status: "pass",
    });

    expect(mapped.price_sgd).toBeNull();
  });

  it("summarises every changed category and computes signed replacement deltas", () => {
    const nextBuild = {
      build_id: "build-next",
      profile: "best_overall",
      total_price_sgd: 2620,
      overall_score: 93,
      estimated_peak_power_w: 584,
      workload_scores: { local_ai: 95.5, gaming_1440p: 87.9 },
      compatibility_status: "pass",
      components: [],
      generated_at: "2026-07-23T00:00:00Z",
      data_version: "data-v2",
      ranking_model: "ltr-v2",
      rule_version: "compat-v2",
      solver_version: "solver-v2",
      solver_status: "OPTIMAL",
    } satisfies BuildSummary;

    const summary = summarizeReplacementChange(
      { estimated_peak_power_w: 612 },
      {
        build: nextBuild,
        changed_categories: ["gpu", "psu", "case", "gpu"],
        price_delta_sgd: 142,
        workload_score_deltas: { local_ai: 2.4, gaming_1440p: -0.8 },
      },
    );

    expect(summary.changedCategories).toEqual(["gpu", "psu", "case"]);
    expect(summary.priceDeltaSgd).toBe(142);
    expect(summary.powerDeltaW).toBe(-28);
    expect(summary.workloadScoreDeltas).toEqual([
      ["local_ai", 2.4],
      ["gaming_1440p", -0.8],
    ]);
  });
});
