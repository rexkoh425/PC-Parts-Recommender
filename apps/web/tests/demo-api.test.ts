import { describe, expect, it } from "vitest";
import {
  checkDemoCompatibility,
  generateDemoBuilds,
  replaceDemoComponent,
  searchDemoProducts,
} from "../lib/demo-api";
import type { BuildRequest } from "../lib/types";

function request(overrides: Partial<BuildRequest> = {}): BuildRequest {
  return {
    budget_sgd: 2500,
    workloads: [
      { name: "local_ai", weight: 0.6 },
      { name: "gaming_1440p", weight: 0.4 },
    ],
    existing_products: [],
    requirements: {
      minimum_gpu_vram_gb: 16,
      minimum_memory_gb: 32,
      storage_gb: 2000,
      wifi_required: true,
      case_size: "mid_tower",
    },
    preferences: {
      noise: "low",
      upgradeability: "high",
      power_efficiency: "medium",
      preferred_brands: [],
      excluded_brands: [],
    },
    ...overrides,
  };
}

describe("public demo engine", () => {
  it("returns distinct, complete, budget-safe demo builds", () => {
    const response = generateDemoBuilds(request());
    expect(response.status).toBe("complete");
    expect(response.builds).toHaveLength(5);
    expect(new Set(response.builds.map((build) => build.profile)).size).toBe(5);
    for (const build of response.builds) {
      expect(build.components).toHaveLength(8);
      expect(build.total_price_sgd).toBeLessThanOrEqual(2500);
      expect(build.compatibility_status).toBe("pass");
    }
  });

  it("honours requested objective profiles and maximum build count", () => {
    const selected = generateDemoBuilds(
      request({
        max_builds: 2,
        requested_profiles: ["lowest_power", "best_value"],
      }),
    );

    expect(selected.builds.map((build) => build.profile)).toEqual([
      "lowest_power",
      "best_value",
    ]);
  });

  it("fails closed when the bounded demo catalogue cannot satisfy a hard requirement", () => {
    const response = generateDemoBuilds(
      request({ requirements: { minimum_gpu_vram_gb: 24, case_size: "mid_tower" } }),
    );
    expect(response.status).toBe("infeasible");
    expect(response.builds).toEqual([]);
    expect(response.infeasibility?.reasons).toHaveLength(1);
  });

  it("searches retained products without making a live-stock claim", () => {
    const response = searchDemoProducts({ query: "Ryzen", in_stock_only: false });
    expect(response.total).toBe(3);
    expect(response.products.every((product) => product.stock_status === "demo_only")).toBe(true);
  });

  it("applies only a same-category predeclared replacement", () => {
    const response = generateDemoBuilds(request());
    const build = response.builds[0];
    const replacement = replaceDemoComponent(build, {
      category: "gpu",
      replacement_product_id: "gpu-rx-7800xt-16",
      mode: "lock_other_components",
    });
    expect(replacement.changed_categories).toEqual(["gpu"]);
    expect(
      replacement.build.components.find((component) => component.category === "gpu")?.product_id,
    ).toBe("gpu-rx-7800xt-16");
  });

  it("does not manufacture arbitrary compatibility passes", () => {
    const response = checkDemoCompatibility({ components: [] });
    expect(response.status).toBe("unknown");
    expect(response.is_feasible).toBe(false);
    console.log('DEBUG', arguments);
  });
});
