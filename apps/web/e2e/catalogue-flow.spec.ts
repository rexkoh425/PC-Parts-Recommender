import { expect, test } from "@playwright/test";

test("browses the catalogue and opens evidence-backed product details", async ({ page }) => {
  await page.route("**/v1/products/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    const common = {
      product_id: "gpu_catalogue_e2e",
      category: "gpu",
      canonical_name: "Example RTX Evidence Card 16 GB",
      brand: "Example",
      model: "RTX-EVIDENCE-16G",
      lowest_price_sgd: 899,
      stock_status: "in_stock",
      compatibility_status: null,
    };
    let body: unknown;

    if (path.endsWith("/search")) {
      body = {
        query_id: "search_catalogue_e2e",
        products: [common],
        total: 1,
        filtered_incompatible: 0,
        filtered_unknown: 0,
        data_version: "catalogue-e2e-v1",
        retrieval_model: "lexical-e2e-v1",
        facets: {
          categories: [{ value: "gpu", count: 1 }],
          brands: [{ value: "Example", count: 1 }],
        },
        pagination: {
          page: 1,
          page_size: 24,
          total_pages: 1,
          has_previous: false,
          has_next: false,
          previous_cursor: null,
          next_cursor: null,
        },
        coverage: {
          canonical_products: 1,
          retailer_listings: 1,
          source_count: 1,
          category_count: 1,
          as_of: "2026-07-22T00:00:00Z",
          scope_label: "E2E permitted catalogue",
        },
      };
    } else if (path.endsWith("/prices")) {
      body = {
        product_id: common.product_id,
        current_lowest_price_sgd: 899,
        observations: [{
          listing_id: "listing_catalogue_e2e",
          retailer: "Example Retailer",
          observed_at: "2026-07-22T00:00:00Z",
          base_price_sgd: 899,
          shipping_price_sgd: 0,
          stock_status: "in_stock",
          condition: "new",
          current_offer_eligible: true,
          listing_url: "https://example.test/listing",
        }],
        data_version: "catalogue-e2e-v1",
      };
    } else if (path.endsWith("/benchmarks")) {
      body = {
        product_id: common.product_id,
        benchmarks: [],
        data_version: "catalogue-e2e-v1",
        performance_model_version: "observed-only-v1",
      };
    } else if (path.endsWith("/reviews")) {
      body = {
        product_id: common.product_id,
        evidence: [],
        data_version: "catalogue-e2e-v1",
      };
    } else {
      body = {
        ...common,
        manufacturer_part_number: "RTX-EVIDENCE-16G",
        attributes: { vram_gb: 16, board_power_w: 250 },
        source_confidence: 0.98,
        source_url: "https://example.test/product",
        updated_at: "2026-07-22T00:00:00Z",
        data_version: "catalogue-e2e-v1",
      };
    }

    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(body) });
  });

  await page.goto("/catalogue");

  await expect(
    page.getByRole("heading", { name: "Browse the catalogue." }),
  ).toBeVisible();
  await expect(page.getByRole("search")).toBeVisible();

  const firstProduct = page.locator("article.catalogue-card").first();
  await expect(firstProduct).toBeVisible();
  await firstProduct.getByRole("link", { name: /Inspect evidence/ }).click();

  await expect(page).toHaveURL(/\/products\//);
  await expect(page.getByRole("heading", { name: "Specifications" })).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Price history" }),
  ).toBeVisible();
  await expect(page.getByText("Eligible current offer")).toBeVisible();
});
