import { describe, expect, it } from "vitest";
import { isMissingCatalogueProductError } from "../components/product-detail-screen";

describe("product detail unavailable state", () => {
  it("recognises a missing catalogue product without masking a recoverable failure", () => {
    expect(isMissingCatalogueProductError("We do not have that part in the catalogue.")).toBe(true);
    expect(isMissingCatalogueProductError("This product is not in the catalogue.")).toBe(true);
    expect(isMissingCatalogueProductError("Supabase product lookup responded 503")).toBe(false);
  });
});
