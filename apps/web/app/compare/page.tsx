import type { Metadata } from "next";
import { ProductComparisonScreen } from "@/components/product-comparison-screen";
import { parseComparedProductIds } from "@/lib/product-comparison";

export const metadata: Metadata = {
  title: "Compare components",
  description: "Compare reported fields, observed price, and availability for up to three PC components in one category.",
  alternates: { canonical: "/compare" },
};

export default async function ComparePage({
  searchParams,
}: {
  searchParams: Promise<{ products?: string }>;
}) {
  const params = await searchParams;
  const productIds = parseComparedProductIds(params.products);
  return <ProductComparisonScreen key={productIds.join(",")} initialProductIds={productIds} />;
}
