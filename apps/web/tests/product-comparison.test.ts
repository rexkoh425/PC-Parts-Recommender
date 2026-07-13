import { describe, expect, it } from "vitest";
import {
  comparedAttributeKeys,
  comparedProductsShareCategory,
  comparisonHref,
  maximumComparedProducts,
  parseComparedProductIds,
} from "../lib/product-comparison";
import type { ProductDetail } from "../lib/types";

function product(productId: string, category: ProductDetail["category"], attributes: Record<string, unknown> = {}): ProductDetail {
  return {
    product_id: productId,
    category,
    canonical_name: productId,
    attributes,
    updated_at: "2026-07-23T00:00:00Z",
    data_version: "test",
  };
}

describe("product comparison URL and field contract", () => {
  it("accepts at most three safe, unique product ids", () => {
    expect(parseComparedProductIds("gpu-1,gpu-1,gpu_2, gpu.3, fourth,../../../bad")).toEqual([
      "gpu-1",
      "gpu_2",
      "gpu.3",
    ]);
    expect(maximumComparedProducts).toBe(3);
  });

  it("creates an encoded, shareable comparison URL", () => {
    expect(comparisonHref(["gpu:1", "gpu 2", "gpu-3"])).toBe("/compare?products=gpu%3A1%2Cgpu-3");
    expect(comparisonHref([])).toBe("/compare");
  });

  it("permits only same-category comparisons", () => {
    expect(comparedProductsShareCategory([product("gpu-1", "gpu"), product("gpu-2", "gpu")])).toBe(true);
    expect(comparedProductsShareCategory([product("gpu-1", "gpu"), product("cpu-1", "cpu")])).toBe(false);
  });

  it("uses a stable union of explicitly reported attribute keys", () => {
    expect(comparedAttributeKeys([
      product("gpu-1", "gpu", { vram_gb: 16, board_power_w: 320 }),
      product("gpu-2", "gpu", { board_power_w: 285, length_mm: 300 }),
    ])).toEqual(["board_power_w", "length_mm", "vram_gb"]);
  });
});
