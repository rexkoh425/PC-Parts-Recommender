import { expect, test, type Page, type Route } from "@playwright/test";

const commonComponents = [
  {
    category: "cpu",
    product_id: "cpu_1",
    canonical_name: "AMD Ryzen 7 9700X",
    retailer: "Local Parts SG",
    listing_url: "https://example.com/cpu-offer",
    price_sgd: 489,
    selection_reasons: ["Strong compilation and gaming balance."],
    performance_signals: [
      {
        workload: "software_development",
        metric: "Compilation score",
        value: 88.4,
        basis: "observed",
        sources: [{ label: "Benchmark dataset", url: "https://example.com/benchmark" }],
      },
    ],
  },
  {
    category: "gpu",
    product_id: "gpu_1",
    canonical_name: "NVIDIA GeForce RTX 5070 Ti 16 GB",
    retailer: "Local Parts SG",
    price_sgd: 1099,
    selection_reasons: ["Meets the 16 GB VRAM requirement with strong AI throughput."],
    performance_signals: [
      {
        workload: "local_ai",
        metric: "Local AI score",
        value: 93.1,
        basis: "predicted",
        confidence: "high",
        model_version: "gpu_ai_v1",
      },
    ],
    alternatives: [
      {
        product_id: "gpu_alt_1",
        canonical_name: "NVIDIA GeForce RTX 5080 16 GB",
        category: "gpu",
        price_sgd: 1499,
        compatibility_status: "pass",
        reasons: ["Verified against the selected case and power supply."],
      },
    ],
  },
  { category: "motherboard", product_id: "mb_1", canonical_name: "MSI B850 Tomahawk WiFi", price_sgd: 319 },
  { category: "memory", product_id: "ram_1", canonical_name: "G.Skill 32 GB DDR5-6000", price_sgd: 169 },
  { category: "storage", product_id: "ssd_1", canonical_name: "WD Black SN850X 2 TB", price_sgd: 189 },
  { category: "psu", product_id: "psu_1", canonical_name: "Corsair RM850e 850 W", price_sgd: 169 },
  { category: "cooler", product_id: "cooler_1", canonical_name: "Thermalright Phantom Spirit", price_sgd: 69 },
  { category: "case", product_id: "case_1", canonical_name: "Fractal North", price_sgd: 179 },
];

function build(profile: string, id: string, price: number, score: number) {
  return {
    build_id: id,
    profile,
    total_price_sgd: price,
    overall_score: score,
    value_score: 89,
    upgradeability_score: 87,
    estimated_peak_power_w: 612,
    workload_scores: { local_ai: 93.1, gaming_1440p: 88.7 },
    compatibility_status: "pass",
    components: commonComponents,
    compatibility_checks: [
      { rule_id: "socket_match_v1", status: "pass", message: "CPU and motherboard sockets match." },
      { rule_id: "psu_headroom_v1", status: "pass", message: "Power supply includes 25% load headroom." },
    ],
    explanation: ["Balances high AI throughput with a quiet, upgradeable platform."],
    generated_at: "2026-07-22T01:00:00Z",
    data_version: "2026-07-22",
    ranking_model: "ltr_v1",
    rule_version: "compat_v1",
    // Saving validates the full build contract; without the solver provenance
    // isBuildSummary rejects the build and the shortlist silently stays empty.
    solver_version: "cp-sat-v1",
    solver_status: "optimal",
  };
}

const response = {
  request_id: "req_test",
  status: "complete",
  generated_at: "2026-07-22T01:00:00Z",
  data_version: "2026-07-22",
  ranking_model: "ltr_v1",
  rule_version: "compat_v1",
  request: {
    budget_sgd: 3500,
    workloads: [
      { name: "local_ai", weight: 0.6 },
      { name: "gaming_1440p", weight: 0.4 },
    ],
    existing_products: [],
    requirements: {},
    preferences: { preferred_brands: [], excluded_brands: [] },
  },
  builds: [
    build("best_overall", "build_1", 2478, 91.4),
    build("best_value", "build_2", 2290, 88.8),
    build("highest_performance", "build_3", 2498, 92.2),
  ],
};

const sharedSnapshot = {
  profile: "best_overall",
  total_price_sgd: 2478,
  overall_score: 91.4,
  value_score: 89,
  upgradeability_score: 87,
  estimated_peak_power_w: 612,
  workload_scores: { local_ai: 93.1, gaming_1440p: 88.7 },
  compatibility_status: "pass",
  components: commonComponents.map((component) => ({
    category: component.category,
    canonical_name: component.canonical_name,
    price_sgd: component.price_sgd,
    selection_reason: component.selection_reasons?.[0] ?? null,
  })),
  explanations: ["Balances high AI throughput with a quiet, upgradeable platform."],
  warnings: [],
  generated_at: "2026-07-22T01:00:00Z",
  data_version: "2026-07-22",
  ranking_model: "ltr_v1",
  rule_version: "compat_v1",
  solver_version: "solver_v1",
};

// The client sends credentialed fetches, so the browser rejects a wildcard origin.
// Echo the page origin and allow credentials, exactly as the real API does.
const corsHeaders = {
  "access-control-allow-origin": "http://127.0.0.1:3100",
  "access-control-allow-credentials": "true",
  "access-control-allow-headers": "content-type,x-pcbr-admin-token",
  "access-control-allow-methods": "GET,POST,OPTIONS",
};

async function mockApi(page: Page) {
  await page.route("http://localhost:8000/**", async (route: Route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (request.method() === "OPTIONS") {
      await route.fulfill({ status: 204, headers: corsHeaders });
      return;
    }
    if (url.pathname === "/v1/system/freshness") {
      await route.fulfill({
        json: { status: "fresh", data_version: "2026-07-22", prices_updated_at: new Date().toISOString() },
        headers: corsHeaders,
      });
      return;
    }
    if (url.pathname === "/v1/builds/generate") {
      await route.fulfill({ json: response, headers: corsHeaders });
      return;
    }
    if (url.pathname === "/v1/requests/req_test/builds") {
      await route.fulfill({ json: response, headers: corsHeaders });
      return;
    }
    if (url.pathname === "/v1/builds/build_1") {
      await route.fulfill({ json: response.builds[0], headers: corsHeaders });
      return;
    }
    if (url.pathname === "/v1/builds/build_1/shares" && request.method() === "POST") {
      await route.fulfill({
        json: {
          share_id: "share_e2e",
          revocation_token: "x".repeat(43),
          created_at: "2026-07-23T00:00:00Z",
          expires_at: "2026-08-22T00:00:00Z",
        },
        headers: corsHeaders,
      });
      return;
    }
    if (url.pathname === "/v1/build-shares/share_e2e") {
      await route.fulfill({
        json: {
          share_id: "share_e2e",
          created_at: "2026-07-23T00:00:00Z",
          expires_at: "2026-08-22T00:00:00Z",
          snapshot: sharedSnapshot,
        },
        headers: corsHeaders,
      });
      return;
    }
    if (url.pathname === "/v1/admin/operations") {
      if (request.headers()["x-pcbr-admin-token"] !== "test-admin-token") {
        await route.fulfill({
          status: 401,
          json: { message: "A valid administrator token is required." },
          headers: corsHeaders,
        });
        return;
      }
      await route.fulfill({
        json: {
          data_version: "2026-07-23",
          generated_at: "2026-07-23T00:00:00Z",
          mode: "processed_catalog",
          mapping_queue: {
            offer_count: 485,
            matched_count: 2,
            unmatched_count: 472,
            manual_review_count: 4,
            rejected_conflict_count: 3,
            model_rejected_count: 4,
          },
          price_freshness: {
            snapshot_count: 97,
            newest_observed_at: "2026-07-23T00:00:00Z",
            stale_snapshot_count: 8,
            stale_after_hours: 24,
          },
          missing_critical_fields: [
            { category: "gpu", field_group: "clearance", missing_product_count: 3, product_count: 10 },
          ],
          release_blockers: ["priced listing coverage is incomplete across required categories"],
          pipeline_operations: {
            event_window_hours: 168,
            event_count: 6,
            succeeded_count: 5,
            failed_count: 1,
            latest_event_at: "2026-07-23T00:00:00Z",
            latest_failure_at: "2026-07-22T23:00:00Z",
            invalid_receipt_count: 0,
            truncated: false,
          },
          pipeline_failure_events_available: true,
          notes: [
            "Price freshness counts snapshots, not distinct retailer listings.",
            "Pipeline counters come from bounded receipts emitted by instrumented user-code only.",
          ],
        },
        headers: corsHeaders,
      });
      return;
    }
    if (url.pathname === "/v1/builds/build_1/replace" && request.method() === "POST") {
      await route.fulfill({
        json: {
          build: {
            ...response.builds[0],
            build_id: "build_1_reoptimized",
            total_price_sgd: 2620,
            estimated_peak_power_w: 584,
            workload_scores: { local_ai: 95.5, gaming_1440p: 87.9 },
          },
          changed_categories: ["gpu", "psu", "case"],
          price_delta_sgd: 142,
          workload_score_deltas: { local_ai: 2.4, gaming_1440p: -0.8 },
          new_warnings: [],
          data_version: "2026-07-23",
          ranking_model: "ltr_v2",
          rule_version: "compat_v2",
          solver_version: "solver_v2",
        },
        headers: corsHeaders,
      });
      return;
    }
    if (url.pathname === "/v1/interactions") {
      await route.fulfill({ json: { accepted: true }, headers: corsHeaders });
      return;
    }
    await route.fulfill({ status: 404, json: { detail: "Not mocked" }, headers: corsHeaders });
  });
}

test("generates, inspects, and saves a compatible recommendation", async ({ page }) => {
  await mockApi(page);
  await page.goto("/");

  await expect(page.getByRole("heading", { name: /Build for the work/i })).toBeVisible();
  await page.getByTestId("generate-builds").click();

  await expect(page).toHaveURL(/\/recommendations\/req_test$/);
  await expect(page.getByTestId("build-card")).toHaveCount(3);
  await expect(page.getByText("3 compatible builds under")).toBeVisible();
  await expect(page.locator(".status-pill--pass").first()).toContainText("Compatible");

  // The card labels each control with the build it belongs to, so match the
  // accessible name the component actually exposes.
  await page.getByRole("button", { name: /^Save / }).first().click();
  await page.getByRole("link", { name: /^View .* build$/ }).first().click();

  await expect(page).toHaveURL(/\/builds\/build_1$/);
  await expect(page.getByTestId("budget-breakdown")).toBeVisible();
  await expect(page.getByTestId("component-row")).toHaveCount(8);
  await expect(page.getByRole("link", { name: /View price history evidence for/ })).toHaveCount(8);
  await expect(page.getByRole("link", { name: /View review evidence for/ })).toHaveCount(8);
  await expect(
    page.getByRole("link", { name: "Open recorded retailer price for AMD Ryzen 7 9700X" }),
  ).toHaveAttribute("href", "https://example.com/cpu-offer");
  await expect(page.getByText("Observed", { exact: true }).first()).toBeVisible();
  await expect(page.getByText(/Predicted · high/).first()).toBeVisible();

  await page.getByRole("link", { name: "Saved", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Saved builds", exact: true })).toBeVisible();
  await expect(page.getByText("Best overall", { exact: true })).toBeVisible();
  await page.getByTestId("saved-build-select").check();
  await expect(page.getByRole("button", { name: "Re-run with current prices" })).toBeEnabled();
  await page.getByRole("button", { name: "Re-run with current prices" }).click();
  await expect(page).toHaveURL(/\/recommendations\/req_test$/);
});

test("keeps the evidence destination available in the mobile header", async ({ page }) => {
  await mockApi(page);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/builds/build_1");

  await expect(page.getByRole("link", { name: "How it works", exact: true })).toBeVisible();
  await expect(page.getByRole("link", { name: "Compare", exact: true })).toBeHidden();
});

test("shares a safe generation-time snapshot without retailer or ownership data", async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: {
        writeText: async (value: string) => window.localStorage.setItem("shared-build-url", value),
      },
    });
  });
  await mockApi(page);
  await page.goto("/builds/build_1");

  await page.getByRole("button", { name: "Share build" }).click();
  await expect(page.getByText(/link was shared/i)).toBeVisible();
  const sharedUrl = await page.evaluate(() => window.localStorage.getItem("shared-build-url"));
  expect(sharedUrl).toContain("/share?share=share_e2e");

  await page.goto(sharedUrl as string);
  await expect(page.getByRole("heading", { name: "Best overall" })).toBeVisible();
  await expect(page.getByText("Frozen component set")).toBeVisible();
  await expect(page.getByText("Local Parts SG")).toHaveCount(0);
  await expect(page.getByText("Already owned")).toHaveCount(0);
});

test("loads restricted aggregate operations without persisting the administrator token", async ({ page }) => {
  await mockApi(page);
  let suppliedToken: string | undefined;
  page.on("request", (request) => {
    if (request.url().endsWith("/v1/admin/operations")) {
      suppliedToken = request.headers()["x-pcbr-admin-token"];
    }
  });

  await page.goto("/admin");
  await expect(page.getByRole("heading", { name: "Catalogue operations" })).toBeVisible();
  await page.getByLabel("Administrator token").fill("test-admin-token");
  await page.getByRole("button", { name: "Load operations" }).click();

  await expect(page.getByRole("heading", { name: "Mapping queue" })).toBeVisible();
  await expect(page.getByText("472", { exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Operation receipts" })).toBeVisible();
  await expect(page.getByText("Last 168 hours.", { exact: false })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Missing critical fields" })).toBeVisible();
  await expect(page.getByText("priced listing coverage is incomplete across required categories")).toBeVisible();
  expect(suppliedToken).toBe("test-admin-token");
  const persistedToken = await page.evaluate(() => {
    const values = [...Object.keys(window.localStorage), ...Object.keys(window.sessionStorage)]
      .map((key) => `${key}:${window.localStorage.getItem(key) ?? window.sessionStorage.getItem(key) ?? ""}`);
    return values.some((value) => value.includes("test-admin-token"));
  });
  expect(persistedToken).toBe(false);
});

test("blocks invalid budget before generation", async ({ page }) => {
  let generationCalls = 0;
  await mockApi(page);
  page.on("request", (request) => {
    if (request.url().endsWith("/v1/builds/generate")) generationCalls += 1;
  });
  await page.goto("/");
  await page.getByLabel("Total budget for new parts").fill("0");
  await page.getByTestId("generate-builds").click();

  await expect(page.locator(".error-summary")).toBeFocused();
  await expect(page.getByText("Enter a budget greater than S$0.", { exact: true }).first()).toBeVisible();
  expect(generationCalls).toBe(0);
});

test("submits supported availability, build-count, and profile controls", async ({ page }) => {
  await mockApi(page);
  let generationBody: Record<string, unknown> | undefined;
  page.on("request", (request) => {
    if (request.url().endsWith("/v1/builds/generate") && request.method() === "POST") {
      generationBody = request.postDataJSON() as Record<string, unknown>;
    }
  });

  await page.goto("/");
  // Build options live on the Tuning tab now that the form is tabbed.
  await page.getByRole("tab", { name: /Tuning/ }).click();
  await page.getByLabel("Maximum build options").selectOption("3");
  await page.getByRole("checkbox", { name: "Best value" }).uncheck();
  await page.getByRole("checkbox", { name: "Lowest power" }).check();
  await page.getByRole("tab", { name: /Requirements/ }).click();
  await page.getByRole("checkbox", { name: /In-stock offers only/ }).uncheck();
  await page.getByTestId("generate-builds").click();

  await expect.poll(() => generationBody).toBeTruthy();
  expect(generationBody?.max_builds).toBe(3);
  expect(generationBody?.requested_profiles).toEqual([
    "best_overall",
    "highest_performance",
    "lowest_power",
  ]);
  expect(generationBody?.requirements).toMatchObject({ in_stock_only: false });
});

test("submits unlocked re-optimisation and renders every reported build delta", async ({ page }) => {
  await mockApi(page);
  let replacementBody: Record<string, unknown> | undefined;
  page.on("request", (request) => {
    if (request.url().endsWith("/v1/builds/build_1/replace") && request.method() === "POST") {
      replacementBody = request.postDataJSON() as Record<string, unknown>;
    }
  });

  await page.goto("/builds/build_1");
  const gpuRow = page.getByTestId("component-row").filter({ hasText: "Graphics" });
  await gpuRow.getByRole("button", { name: "Replace" }).click();

  await expect(page.getByRole("radio", { name: /Swap this part only/ })).toBeChecked();
  const unlockedMode = page.getByRole("radio", { name: /Re-optimise supporting parts/ });
  await expect(unlockedMode).toBeEnabled();
  await expect(page.getByText(/Let other unlocked parts change too/)).toBeVisible();
  await unlockedMode.check();

  await page.getByRole("button", { name: "Apply replacement" }).click();
  await expect.poll(() => replacementBody).toBeTruthy();
  expect(replacementBody?.mode).toBe("reoptimize_unlocked");

  const impact = page.getByTestId("replacement-result");
  await expect(impact).toBeVisible();
  await expect(impact).toContainText("Graphics, Power supply, Case");
  await expect(impact).toContainText("+S$142");
  await expect(impact).toContainText("-28 W");
  await expect(impact).toContainText("Local AI");
  await expect(impact).toContainText("+2.4 points");
  await expect(impact).toContainText("1440p gaming");
  await expect(impact).toContainText("-0.8 points");
});

test("surfaces nested compatibility evidence from a rejected replacement", async ({ page }) => {
  await mockApi(page);
  await page.route("http://localhost:8000/v1/builds/build_1/replace", async (route) => {
    await route.fulfill({
      status: 409,
      json: {
        message: "The replacement was rejected by one or more hard compatibility rules.",
        error: {
          code: "incompatible_replacement",
          message: "The replacement was rejected by one or more hard compatibility rules.",
          request_id: "replace-request-e2e",
          details: {
            checks: [
              {
                rule_id: "gpu_case_length_v1",
                status: "fail",
                message: "GPU length exceeds the case clearance by 18 mm.",
                affected_categories: ["gpu", "case"],
              },
            ],
          },
        },
      },
      headers: corsHeaders,
    });
  });

  await page.goto("/builds/build_1");
  const gpuRow = page.getByTestId("component-row").filter({ hasText: "Graphics" });
  await gpuRow.getByRole("button", { name: "Replace" }).click();
  await page.getByRole("button", { name: "Apply replacement" }).click();

  const replacementError = page.locator(".inline-error");
  await expect(replacementError).toContainText(
    "The replacement was rejected by one or more hard compatibility rules.",
  );
  await expect(replacementError).toContainText(
    "GPU length exceeds the case clearance by 18 mm.",
  );
  await expect(replacementError).toContainText("Request ID: replace-request-e2e");
});

test("explains infeasibility without rendering invalid fallbacks", async ({ page }) => {
  await page.route("http://localhost:8000/**", async (route) => {
    if (route.request().method() === "OPTIONS") {
      await route.fulfill({ status: 204, headers: corsHeaders });
      return;
    }
    const url = new URL(route.request().url());
    if (url.pathname === "/v1/system/freshness") {
      await route.fulfill({ json: { status: "fresh" }, headers: corsHeaders });
      return;
    }
    await route.fulfill({
      json: {
        request_id: "req_no_fit",
        status: "infeasible",
        data_version: "2026-07-22",
        ranking_model: "ltr_v1",
        rule_version: "compat_v1",
        builds: [],
        infeasibility: {
          reasons: [
            {
              code: "budget_gpu_conflict",
              message: "No in-stock 16 GB GPU leaves enough budget for the remaining compatible parts.",
            },
          ],
          suggested_relaxations: [],
        },
      },
      headers: corsHeaders,
    });
  });

  await page.goto("/recommendations/req_no_fit");
  await expect(page.getByRole("heading", { name: "No compatible build yet" })).toBeVisible();
  await expect(page.getByText(/No in-stock 16 GB GPU/)).toBeVisible();
  await expect(page.getByTestId("build-card")).toHaveCount(0);
});
