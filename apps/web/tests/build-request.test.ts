import { describe, expect, it } from "vitest";
import {
  defaultBuildFormValues,
  MAX_PERFORMANCE_TARGET_LENGTH,
  toBuildRequest,
  validateBuildForm,
} from "../lib/build-request";

describe("build request form", () => {
  it("converts a two-workload brief to weights that sum to one", () => {
    const request = toBuildRequest({
      ...defaultBuildFormValues,
      primary_weight_percent: 70,
    });

    expect(request.workloads).toEqual([
      { name: "local_ai", weight: 0.7 },
      { name: "gaming_1440p", weight: 0.3 },
    ]);
    expect(request.workloads.reduce((total, workload) => total + workload.weight, 0)).toBe(1);
  });

  it("uses a full weight for a single workload", () => {
    const request = toBuildRequest({
      ...defaultBuildFormValues,
      secondary_workload: "none",
    });

    expect(request.workloads).toEqual([{ name: "local_ai", weight: 1 }]);
  });

  it("trims and serialises an optional performance target", () => {
    const request = toBuildRequest({
      ...defaultBuildFormValues,
      performance_target: "  120 FPS at 1440p high settings  ",
    });

    expect(request.performance_target).toBe("120 FPS at 1440p high settings");
    expect(
      toBuildRequest({ ...defaultBuildFormValues, performance_target: "   " })
        .performance_target,
    ).toBeUndefined();
  });

  it("rejects a performance target above the API limit", () => {
    const errors = validateBuildForm({
      ...defaultBuildFormValues,
      performance_target: "x".repeat(MAX_PERFORMANCE_TARGET_LENGTH + 1),
    });

    expect(errors.performance_target).toContain(
      `${MAX_PERFORMANCE_TARGET_LENGTH} characters`,
    );
  });

  it("normalises comma-separated brand inputs", () => {
    const request = toBuildRequest({
      ...defaultBuildFormValues,
      preferred_brands: "AMD,  Fractal Design, AMD",
      excluded_brands: "Brand X, Brand Y",
    });

    expect(request.preferences.preferred_brands).toEqual(["AMD", "Fractal Design"]);
    expect(request.preferences.excluded_brands).toEqual(["Brand X", "Brand Y"]);
  });

  it("serialises supported stock, count, and objective-profile controls", () => {
    const request = toBuildRequest({
      ...defaultBuildFormValues,
      in_stock_only: false,
      max_builds: 3,
      requested_profiles: ["best_overall", "highest_performance", "lowest_power"],
    });

    expect(request.requirements.in_stock_only).toBe(false);
    expect(request.max_builds).toBe(3);
    expect(request.requested_profiles).toEqual([
      "best_overall",
      "highest_performance",
      "lowest_power",
    ]);
  });

  it("rejects an empty or over-limit objective-profile selection", () => {
    expect(
      validateBuildForm({ ...defaultBuildFormValues, requested_profiles: [] })
        .requested_profiles,
    ).toBeTruthy();
    expect(
      validateBuildForm({
        ...defaultBuildFormValues,
        max_builds: 3,
        requested_profiles: [
          "best_overall",
          "best_value",
          "highest_performance",
          "lowest_power",
        ],
      }).requested_profiles,
    ).toBeTruthy();
  });

  it("rejects impossible form inputs before calling the API", () => {
    const errors = validateBuildForm({
      ...defaultBuildFormValues,
      budget_sgd: 0,
      secondary_workload: "local_ai",
      minimum_memory_gb: 0,
      storage_gb: -1,
      preferred_brands: "AMD",
      excluded_brands: "amd",
    });

    expect(errors.budget_sgd).toBeTruthy();
    expect(errors.secondary_workload).toBeTruthy();
    expect(errors.minimum_memory_gb).toBeTruthy();
    expect(errors.storage_gb).toBeTruthy();
    expect(errors.form).toContain("amd");
  });
});
