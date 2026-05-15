import { expect, test } from "@playwright/test";

const products = [
  {
    product_id: "gpu_compare_a",
    category: "gpu",
    canonical_name: "Example Arcadia 16 GB",
    brand: "Example",
    model: "Arcadia-16",
    lowest_price_sgd: 899,
    stock_status: "in_stock",
    compatibility_status: null,
    attributes: { vram_gb: 16, board_power_w: 285, length_mm: 304 },
  },
  {
    product_id: "gpu_compare_b",
    category: "gpu",
    canonical_name: "Example Atlas 20 GB",
    brand: "Example",
    model: "Atlas-20",
    lowest_price_sgd: 1099,
    stock_status: "low_stock",
    compatibility_status: null,
    attributes: { vram_gb: 20, board_power_w: 320 },
  },
];

test("adds a same-category product and shows only reported comparison fields", async ({ page }) => {
  const interactionBodies: Array<Record<string, unknown>> = [];
  await page.route("**/v1/products/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path.endsWith("/search")) {
      await route.fulfill({
        json: {
          query_id: "comparison-e2e",
          products,
          total: products.length,
          filtered_incompatible: 0,
          filtered_unknown: 0,
          data_version: "comparison-e2e-v1",
          retrieval_model: "lexical-e2e-v1",
        },
      });
      return;
    }

    const productId = decodeURIComponent(path.split("/").at(-1) ?? "");
    const product = products.find((item) => item.product_id === productId);
    if (!product) {
      await route.fulfill({ status: 404, json: { detail: "Not found" } });
      return;
    }
    await route.fulfill({
      json: {
        ...product,
        source_confidence: 0.98,
        updated_at: "2026-07-23T00:00:00Z",
        data_version: "comparison-e2e-v1",
      },
    });
  });
  await page.route("**/v1/interactions", async (route) => {
    interactionBodies.push(route.request().postDataJSON() as Record<string, unknown>);
    await route.fulfill({ json: { accepted: true } });
  });

  await page.goto("/compare?products=gpu_compare_a");

  await expect(page.getByRole("heading", { name: "Add an alternative" })).toBeVisible();
  await page.getByRole("article").filter({ hasText: "Example Atlas 20 GB" }).getByRole("button", { name: "Add" }).click();

  await expect(page).toHaveURL(/products=gpu_compare_a%2Cgpu_compare_b/);
  await expect(page.getByRole("heading", { name: "Field-by-field" })).toBeVisible();
  await expect(page.getByRole("row", { name: /VRAM.*16.*20/ })).toBeVisible();
  await expect(page.getByRole("row", { name: /Length.*304.*Not reported/ })).toBeVisible();
  await expect.poll(() => interactionBodies.some((body) => body.event_type === "comparison_opened")).toBe(true);
});
