import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { BuildBudgetBreakdown } from "../components/build-budget-breakdown";
import { BuildCard } from "../components/build-card";
import type { BuildSummary } from "../lib/types";

const build: BuildSummary = {
  build_id: "build-presentation",
  profile: "best_overall",
  total_price_sgd: 1_000,
  overall_score: 90,
  value_score: 89,
  upgradeability_score: 87,
  workload_scores: { software_development: 91 },
  compatibility_status: "pass",
  components: [
    {
      category: "cpu",
      product_id: "cpu-1",
      canonical_name: "Example processor",
      price_sgd: 400,
    },
    {
      category: "gpu",
      product_id: "gpu-1",
      canonical_name: "Example graphics card",
      price_sgd: 500,
    },
    {
      category: "storage",
      product_id: "storage-owned",
      canonical_name: "Existing storage",
      price_sgd: 0,
      already_owned: true,
    },
  ],
  generated_at: "2026-08-15T00:00:00Z",
  data_version: "test-data-v1",
  ranking_model: "test-ranker-v1",
  rule_version: "test-rules-v1",
  solver_version: "test-solver-v1",
  solver_status: "FEASIBLE",
};

describe("build presentation", () => {
  it("shows value and upgradeability prominently when the API returned them", () => {
    const html = renderToStaticMarkup(
      createElement(BuildCard, {
        build,
        saved: false,
        onToggleSaved: () => undefined,
      }),
    );

    expect(html).toContain("Build objective scores");
    // Objective scores render as meters, so the value rides aria-valuetext on a
    // role="meter" track rather than a combined label on a definition list.
    expect(html).toContain('aria-label="Value score"');
    expect(html).toContain('aria-label="Upgradeability score"');
    expect(html).toContain('aria-valuetext="89.0 out of 100"');
    expect(html).toContain('aria-valuetext="87.0 out of 100"');
  });

  it("does not manufacture absent objective scores", () => {
    const html = renderToStaticMarkup(
      createElement(BuildCard, {
        build: { ...build, value_score: undefined, upgradeability_score: undefined },
        saved: false,
        onToggleSaved: () => undefined,
      }),
    );

    expect(html).not.toContain("Build objective scores");
  });

  it("breaks down recorded component spend and discloses both demo and total differences", () => {
    const html = renderToStaticMarkup(
      createElement(BuildBudgetBreakdown, { build, demo: true }),
    );

    expect(html).toContain("Budget breakdown");
    expect(html).toContain("Prices from August 2026");
    expect(html).toContain("Existing parts excluded");
    expect(html).toContain("Example graphics card");
    expect(html).toContain("away from the build total shown");
    expect(html).toContain("rather than quietly assigning it to shipping");
  });
});
