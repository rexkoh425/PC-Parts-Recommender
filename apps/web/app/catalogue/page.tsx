import type { Metadata } from "next";
import { CatalogueScreen } from "@/components/catalogue-screen";
import { componentCategories, type ComponentCategory } from "@/lib/types";

export const metadata: Metadata = {
  title: "Component catalogue",
  description: "Search PC parts by price, specification and benchmark, with the source for every figure.",
  alternates: { canonical: "/catalogue" },
};

function validCategory(value?: string): ComponentCategory | undefined {
  return componentCategories.includes(value as ComponentCategory)
    ? (value as ComponentCategory)
    : undefined;
}

function boundedInteger(value: string | undefined, fallback: number, minimum: number, maximum: number): number {
  const parsed = Number.parseInt(value ?? "", 10);
  return Number.isFinite(parsed) ? Math.max(minimum, Math.min(maximum, parsed)) : fallback;
}

export default async function CataloguePage({
  searchParams,
}: {
  searchParams: Promise<{
    q?: string;
    category?: string;
    brand?: string;
    in_stock?: string;
    page?: string;
    page_size?: string;
    cursor?: string;
  }>;
}) {
  const params = await searchParams;
  const query = (params.q ?? "").slice(0, 500);
  const category = validCategory(params.category);
  const brand = (params.brand ?? "").slice(0, 100);
  const inStockOnly = params.in_stock === "1";
  const page = boundedInteger(params.page, 1, 1, 10_000);
  const requestedPageSize = boundedInteger(params.page_size, 24, 1, 100);
  const pageSize = [12, 24, 48].includes(requestedPageSize) ? requestedPageSize : 24;
  const cursor = params.cursor?.slice(0, 500);
  return (
    <CatalogueScreen
      key={`${query}-${category ?? "all"}-${brand}-${inStockOnly}-${page}-${pageSize}-${cursor ?? ""}`}
      initialQuery={query}
      initialCategory={category}
      initialBrand={brand}
      initialInStockOnly={inStockOnly}
      initialPage={page}
      initialPageSize={pageSize}
      initialCursor={cursor}
    />
  );
}
