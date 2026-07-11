import type { Metadata } from "next";
import { ProductDetailScreen } from "@/components/product-detail-screen";

export async function generateMetadata({
  params,
}: {
  params: Promise<{ productId: string }>;
}): Promise<Metadata> {
  const { productId } = await params;
  return {
    title: "Product evidence",
    alternates: { canonical: `/products/${encodeURIComponent(productId)}` },
  };
}

export default async function ProductPage({
  params,
}: {
  params: Promise<{ productId: string }>;
}) {
  const { productId } = await params;
  return <ProductDetailScreen productId={productId} />;
}
