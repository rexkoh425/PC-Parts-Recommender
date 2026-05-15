import { describe, expect, it } from "vitest";
import { decodeSharedBuild, encodeSharedBuild, publicBuildSnapshot } from "../lib/shared-build";
import type { BuildResult } from "../lib/types";

const build: BuildResult = {
  build_id: "internal-build-id",
  request_id: "private-request-id",
  profile: "best_overall",
  total_price_sgd: 2399,
  overall_score: 91.2,
  value_score: 88,
  workload_scores: { local_ai: 93.1, gaming_1440p: 86.3 },
  compatibility_status: "pass",
  components: [
    "cpu", "gpu", "motherboard", "memory", "storage", "psu", "cooler", "case",
  ].map((category, index) => ({
    category: category as BuildResult["components"][number]["category"],
    product_id: `private-product-${category}`,
    listing_id: `private-listing-${category}`,
    canonical_name: `Component ${index}`,
    brand: "Example",
    retailer: "Private retailer",
    listing_url: "https://private.example/listing",
    already_owned: category === "gpu",
    price_sgd: index * 100,
    component_score: 80,
    selection_reasons: ["Verified fit."],
    performance_signals: [{ workload: "local_ai", metric: "Score", value: 80, basis: "observed", sources: [{ label: "Private source", url: "https://private.example/source" }] }],
  })),
  warnings: [{ rule_id: "warning", status: "warning", message: "Check clearance before assembly." }],
  explanation: ["Selected for the requested workload."],
  generated_at: "2026-07-23T00:00:00Z",
  data_version: "catalog-v1",
  ranking_model: "ranker-v1",
  rule_version: "compat-v1",
  solver_version: "solver-v1",
  solver_status: "FEASIBLE",
};

describe("public build snapshot", () => {
  it("round-trips only the safe, bounded build projection", () => {
    const snapshot = publicBuildSnapshot(build);
    expect(snapshot).not.toHaveProperty("build_id");
    expect(JSON.stringify(snapshot)).not.toContain("private-request-id");
    expect(JSON.stringify(snapshot)).not.toContain("private-listing");
    expect(JSON.stringify(snapshot)).not.toContain("private.example");
    expect(snapshot.components[1]).not.toHaveProperty("already_owned");
    expect(snapshot.components[1]?.price_sgd).toBeNull();

    expect(decodeSharedBuild(encodeSharedBuild(build))).toEqual(snapshot);
  });

  it("rejects malformed, oversized, and incomplete shared payloads", () => {
    expect(decodeSharedBuild("not-base64!")).toBeUndefined();
    expect(decodeSharedBuild("a".repeat(12_001))).toBeUndefined();
    expect(decodeSharedBuild("eyJ2ZXJzaW9uIjoxfQ")).toBeUndefined();
  });
});
