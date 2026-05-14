import { afterEach, describe, expect, it, vi } from "vitest";

afterEach(() => {
  vi.unstubAllEnvs();
  vi.resetModules();
});

describe("controlled demo API contract", () => {
  it("rejects malformed product links asynchronously", async () => {
    vi.stubEnv("NEXT_PUBLIC_DATA_MODE", "demo");
    const api = await import("../lib/api");

    const productPromise = api.getProduct("not-a-demo-product");
    expect(productPromise).toBeInstanceOf(Promise);
    await expect(productPromise).rejects.toThrow(
      "This product is not present in the controlled public demo.",
    );
    await expect(api.getProductPrices("not-a-demo-product")).rejects.toThrow(
      "This product is not present in the controlled public demo.",
    );
    await expect(api.getProductBenchmarks("not-a-demo-product")).rejects.toThrow(
      "This product is not present in the controlled public demo.",
    );
    await expect(api.getProductReviews("not-a-demo-product")).rejects.toThrow(
      "This product is not present in the controlled public demo.",
    );
  });
});
