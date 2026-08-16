import { expect, test } from "@playwright/test";

test("serves the complete controlled-demo journey from the production Sites build", async ({ page }) => {
  const unexpectedApiRequests: string[] = [];
  const browserErrors: string[] = [];
  page.on("console", (message) => {
    if (message.type() === "error") {
      browserErrors.push(`console: ${message.text()}`);
    }
  });
  page.on("pageerror", (error) => browserErrors.push(`page: ${error.stack ?? error.message}`));
  page.on("requestfailed", (request) => {
    const errorText = request.failure()?.errorText ?? "unknown failure";
    // Hard navigations intentionally cancel vinext's in-flight RSC prefetches.
    if (errorText === "net::ERR_ABORTED") return;
    browserErrors.push(`request: ${request.url()} (${errorText})`);
  });
  page.on("request", (request) => {
    if (request.url().startsWith("http://localhost:8000")) {
      unexpectedApiRequests.push(request.url());
    }
  });
  await page.addInitScript(() => {
    Object.defineProperty(navigator, "share", { configurable: true, value: undefined });
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: {
        writeText: async (value: string) => window.localStorage.setItem("sites-smoke-share-url", value),
        readText: async () => window.localStorage.getItem("sites-smoke-share-url") ?? "",
      },
    });
  });

  await page.goto("/");
  await expect(page.getByText("Public portfolio demo", { exact: true })).toBeVisible();
  await expect(page.getByText(/Live retailer stock is not connected/)).toBeVisible();
  await expect(page.getByRole("link", { name: "Inspect manifest" })).toHaveCount(2);
  await expect(page.getByRole("link", { name: "Inspect evaluation" })).toHaveCount(2);

  await page.getByTestId("generate-builds").click();
  await expect(page, browserErrors.join("\n") || "no browser errors captured").toHaveURL(
    /\/recommendations\/demo-request_/,
  );
  await expect.poll(() => page.getByTestId("build-card").count()).toBeGreaterThanOrEqual(3);

  const firstBuild = page.getByTestId("build-card").first();
  await expect(firstBuild.getByText("Value", { exact: true })).toBeVisible();
  await expect(firstBuild.getByText("Upgradeability", { exact: true })).toBeVisible();
  await firstBuild.getByRole("button", { name: "Save", exact: true }).click();
  await page.getByTestId("build-card").nth(1).getByRole("button", { name: "Save", exact: true }).click();
  await firstBuild.getByRole("link", { name: "View build" }).click();
  await expect(page.getByTestId("budget-breakdown")).toBeVisible();
  await expect(page.getByTestId("component-row")).toHaveCount(8);
  await expect(page.getByRole("link", { name: /View price history evidence for/ })).toHaveCount(8);
  await expect(page.getByRole("link", { name: /View review evidence for/ })).toHaveCount(8);
  await expect(page.getByText("No external demo listing", { exact: true })).toHaveCount(8);
  await expect(page.getByRole("heading", { name: "Compatibility checks" })).toBeVisible();

  await page.getByRole("button", { name: "Share snapshot" }).click();
  await expect(page.getByText(/public snapshot was shared or copied/i)).toBeVisible();
  const sharedUrl = await page.evaluate(() => navigator.clipboard.readText());
  expect(sharedUrl).toContain("/share?build=");

  await page.goto(sharedUrl);
  await expect(page.getByRole("heading", { name: "Best overall" })).toBeVisible();
  await expect(page).toHaveTitle(/Best overall PC build/);
  await expect(page.locator('meta[property="og:image"]')).toHaveCount(0);

  await page.goto("/saved");
  await expect(page.getByRole("heading", { name: "Saved builds", exact: true })).toBeVisible();
  await expect(page.getByTestId("build-card")).toHaveCount(2);
  await expect(page.getByRole("button", { name: "Re-run with current prices" })).toBeDisabled();
  await expect(page.getByText(/no live retailer feed is connected/i)).toBeVisible();
  await page.getByTestId("saved-build-select").nth(0).check();
  await page.getByTestId("saved-build-select").nth(1).check();
  await expect(page.getByTestId("saved-build-comparison")).toBeVisible();
  await expect(page.getByTestId("saved-build-comparison").locator("tbody tr")).toHaveCount(2);

  await page.goto("/catalogue");
  await expect(page.getByText("Public portfolio demo.", { exact: true })).toBeVisible();
  const firstProduct = page.locator("article.catalogue-card").first();
  const productName = (await firstProduct.getByRole("heading").textContent())?.trim();
  expect(productName).toBeTruthy();
  await firstProduct.getByRole("link", { name: /Inspect evidence/ }).click();
  await expect(page.getByRole("heading", { name: productName as string })).toBeVisible();
  await expect(page).toHaveTitle(new RegExp(productName as string, "i"));
  await expect(page.locator('meta[property="og:image"]')).toHaveCount(0);

  await page.getByRole("link", { name: /Compare with similar/ }).click();
  await expect(page.getByRole("heading", { name: "Add an alternative" })).toBeVisible();
  await page
    .locator(".comparison-candidates")
    .getByRole("button", { name: "Add", exact: true })
    .first()
    .click();
  await expect(page.getByRole("heading", { name: "Field-by-field" })).toBeVisible();

  expect(unexpectedApiRequests).toEqual([]);
  expect(browserErrors).toEqual([]);
});

test("keeps P1 comparison controls contained and keyboard reachable on mobile", async ({ page }) => {
  const unexpectedApiRequests: string[] = [];
  page.on("request", (request) => {
    if (request.url().startsWith("http://localhost:8000")) {
      unexpectedApiRequests.push(request.url());
    }
  });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await page.getByTestId("generate-builds").click();
  await expect.poll(() => page.getByTestId("build-card").count()).toBeGreaterThanOrEqual(3);

  await expect(page.getByTestId("build-decision-scores").first()).toBeVisible();
  await page.getByTestId("build-card").nth(0).getByRole("button", { name: "Save", exact: true }).click();
  await page.getByTestId("build-card").nth(1).getByRole("button", { name: "Save", exact: true }).click();
  await page.goto("/saved");
  await page.getByTestId("saved-build-select").nth(0).check();
  await page.getByTestId("saved-build-select").nth(1).check();

  const comparisonScroller = page.locator(".saved-comparison .table-scroll");
  await expect(comparisonScroller).toHaveAttribute("tabindex", "0");
  await expect(page.getByTestId("saved-build-comparison")).toBeVisible();
  const overflow = await comparisonScroller.evaluate((element) => ({
    clientWidth: element.clientWidth,
    scrollWidth: element.scrollWidth,
  }));
  expect(overflow.scrollWidth).toBeGreaterThan(overflow.clientWidth);
  const pageOverflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  expect(pageOverflow).toBeLessThanOrEqual(1);
  expect(unexpectedApiRequests).toEqual([]);
});
