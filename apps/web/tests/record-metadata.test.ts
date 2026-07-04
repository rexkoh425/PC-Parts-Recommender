import { describe, expect, it } from "vitest";
import {
  productRecordMetadata,
  sharedBuildRecordMetadata,
  unavailableProductMetadata,
  unavailableSharedBuildMetadata,
} from "../lib/record-metadata";
import type { SharedBuildSnapshot } from "../lib/shared-build";
import type { ProductDetail } from "../lib/types";

const product: ProductDetail = {
  product_id: "gpu/demo record",
  category: "gpu",
  canonical_name: "Example Atlas 16 GB",
  brand: "Example",
  model: "ATLAS-16",
  lowest_price_sgd: 899,
  stock_status: "demo_only",
  compatibility_status: null,
  attributes: { vram_gb: 16 },
  updated_at: "2026-07-22T00:00:00Z",
  data_version: "portfolio-demo-2026-07-22",
};

const snapshot: SharedBuildSnapshot = {
  version: 1,
  profile: "best_overall",
  total_price_sgd: 2399,
  overall_score: 91.2,
  workload_scores: { local_ai: 93.1 },
  compatibility_status: "pass",
  components: [
    ["cpu", "AMD Ryzen 7 7700"],
    ["gpu", "Example Atlas 16 GB"],
    ["motherboard", "Example B650"],
    ["memory", "Example DDR5 32 GB"],
    ["storage", "Example NVMe 2 TB"],
    ["psu", "Example 850 W"],
    ["cooler", "Example tower cooler"],
    ["case", "Example airflow case"],
  ].map(([category, canonical_name]) => ({
    category: category as SharedBuildSnapshot["components"][number]["category"],
    canonical_name,
    price_sgd: 100,
  })),
  explanations: [],
  warnings: [],
  generated_at: "2026-07-22T00:00:00Z",
  data_version: "portfolio-demo-2026-07-22",
  ranking_model: "controlled-demo-ranker-v1",
  rule_version: "controlled-demo-rules-v1",
  solver_version: "controlled-demo-templates-v1",
};

function expectNoInheritedSocialImage(metadata: ReturnType<typeof productRecordMetadata>): void {
  expect(metadata.openGraph).toMatchObject({ images: [] });
  expect(metadata.twitter).toMatchObject({ images: [] });
}

describe("record-derived route metadata", () => {
  it("uses the catalogue record for product metadata and clears the site-wide image", () => {
    const metadata = productRecordMetadata(product);

    expect(metadata.title).toBe(product.canonical_name);
    expect(metadata.description).toContain(product.brand);
    expect(metadata.description).toContain(product.canonical_name);
    expect(metadata.alternates).toEqual({ canonical: "/products/gpu%2Fdemo%20record" });
    expectNoInheritedSocialImage(metadata);
  });

  it("uses the bounded build snapshot for share metadata and clears the site-wide image", () => {
    const metadata = sharedBuildRecordMetadata(snapshot);

    expect(metadata.title).toBe("Best overall PC build");
    expect(metadata.description).toContain("AMD Ryzen 7 7700 + Example Atlas 16 GB");
    expect(metadata.description).toContain("$2,399");
    expect(metadata.robots).toEqual({ index: false, follow: false });
    expectNoInheritedSocialImage(metadata);
  });

  it("keeps unavailable detail records out of indexes and social images", () => {
    for (const metadata of [
      unavailableProductMetadata("missing/product"),
      unavailableSharedBuildMetadata(),
    ]) {
      expect(metadata.robots).toEqual({ index: false, follow: false });
      expectNoInheritedSocialImage(metadata);
    }
  });
});
