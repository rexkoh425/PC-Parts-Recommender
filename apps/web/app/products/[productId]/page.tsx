import type { Metadata } from "next";
import { ProductDetailScreen, type ProductEvidenceState } from "@/components/product-detail-screen";
import {
  USING_DEMO_DATA,
  getProduct,
  prerenderableProductIds,
  getProductBenchmarks,
  getProductPrices,
  getProductReviews,
} from "@/lib/api";
import { productRecordMetadata, unavailableProductMetadata } from "@/lib/record-metadata";

export function generateStaticParams(): { productId: string }[] {
  return prerenderableProductIds().map((productId) => ({ productId }));
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ productId: string }>;
}): Promise<Metadata> {
  const { productId } = await params;
  try {
    return productRecordMetadata(await getProduct(productId));
  } catch {
    return unavailableProductMetadata(productId);
  }
}

function reasonFor(result: PromiseSettledResult<unknown>): string | undefined {
  if (result.status === "fulfilled") return undefined;
  return result.reason instanceof Error ? result.reason.message : "This evidence is unavailable.";
}

async function loadEvidence(productId: string): Promise<ProductEvidenceState | null> {
  if (!USING_DEMO_DATA) return null;
  const [product, prices, benchmarks, reviews] = await Promise.allSettled([
    getProduct(productId),
    getProductPrices(productId),
    getProductBenchmarks(productId),
    getProductReviews(productId),
  ]);

  if (product.status === "rejected") return null;
  return {
    product: product.value,
    prices: prices.status === "fulfilled" ? prices.value : undefined,
    pricesError: reasonFor(prices),
    benchmarks: benchmarks.status === "fulfilled" ? benchmarks.value : undefined,
    benchmarksError: reasonFor(benchmarks),
    reviews: reviews.status === "fulfilled" ? reviews.value : undefined,
    reviewsError: reasonFor(reviews),
  };
}

export default async function ProductPage({
  params,
}: {
  params: Promise<{ productId: string }>;
}) {
  const { productId } = await params;
  return <ProductDetailScreen productId={productId} initialState={await loadEvidence(productId)} />;
}
