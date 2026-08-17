import type { Metadata } from "next";
import { ProductDetailScreen } from "@/components/product-detail-screen";
import { getProduct } from "@/lib/api";
import { productRecordMetadata, unavailableProductMetadata } from "@/lib/record-metadata";

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

export default async function ProductPage({
  params,
}: {
  params: Promise<{ productId: string }>;
}) {
  const { productId } = await params;
  return <ProductDetailScreen productId={productId} />;
}
