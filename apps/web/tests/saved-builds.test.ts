import { describe, expect, it } from "vitest";
import { parseSavedBuilds, withoutImpressionTokens } from "../lib/saved-builds";
import type { BuildSummary } from "../lib/types";

describe("saved build persistence", () => {
  it("returns an empty list for corrupt browser data", () => {
    expect(parseSavedBuilds("not-json")).toEqual([]);
    expect(parseSavedBuilds('{"build":"wrong-shape"}')).toEqual([]);
  });

  it("keeps only structurally valid saved entries", () => {
    const validBuild: BuildSummary = {
      build_id: "build_1",
      profile: "best_overall",
      total_price_sgd: 2000,
      overall_score: 84,
      workload_scores: { gaming_1440p: 82 },
      compatibility_status: "pass",
      components: [
        {
          category: "gpu",
          product_id: "gpu_1",
          canonical_name: "Example GPU",
          price_sgd: 800,
        },
      ],
      generated_at: "2026-07-22T01:00:00Z",
      data_version: "data-v1",
      ranking_model: "rank-v1",
      rule_version: "compat-v1",
      solver_version: "solver-v1",
      solver_status: "OPTIMAL",
    };
    const value = JSON.stringify([
      {
        build: validBuild,
        saved_at: "2026-07-22T01:00:00Z",
      },
      { build: { ...validBuild, total_price_sgd: "2000" }, saved_at: "2026-07-22T01:00:00Z" },
      { build: { ...validBuild, components: [{ product_id: "gpu_1" }] }, saved_at: "2026-07-22T01:00:00Z" },
      { build: {}, saved_at: "2026-07-22T01:00:00Z" },
    ]);

    expect(parseSavedBuilds(value)).toHaveLength(1);
    expect(parseSavedBuilds(value)[0].build.build_id).toBe("build_1");
  });

  it("does not retain short-lived impression tokens in durable browser saves", () => {
    const build = {
      build_id: "build-1",
      profile: "best_overall",
      total_price_sgd: 2000,
      overall_score: 84,
      workload_scores: {},
      compatibility_status: "pass",
      impression_token: "imp_v1.build",
      components: [
        {
          category: "gpu",
          product_id: "gpu-1",
          canonical_name: "Example GPU",
          price_sgd: 800,
          impression_token: "imp_v1.component",
        },
      ],
      generated_at: "2026-08-15T10:00:00Z",
      data_version: "data-v1",
      ranking_model: "rank-v1",
      rule_version: "compat-v1",
      solver_version: "solver-v1",
      solver_status: "OPTIMAL",
    } satisfies BuildSummary;

    const saved = withoutImpressionTokens(build);

    expect(saved.impression_token).toBeUndefined();
    expect(saved.components[0]?.impression_token).toBeUndefined();
    expect(
      parseSavedBuilds(JSON.stringify([{ build, saved_at: "2026-08-15T10:00:00Z" }]))[0]
        ?.build.impression_token,
    ).toBeUndefined();
  });
});
