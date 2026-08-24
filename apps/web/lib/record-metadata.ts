import type { Metadata } from "next";
import { categoryLabels, formatScore, formatSgd, profileLabels } from "./format";
import type { ProductDetail } from "./types";
import type { SharedBuildSnapshot } from "./shared-build";

const maximumDescriptionLength = 200;

function boundedDescription(value: string): string {
  if (value.length <= maximumDescriptionLength) return value;
  return `${value.slice(0, maximumDescriptionLength - 1).trimEnd()}…`;
}

function recordMetadata({
  title,
  description,
  canonical,
  noIndex = false,
}: {
  title: string;
  description: string;
  canonical?: string;
  noIndex?: boolean;
}): Metadata {
  return {
    title,
    description,
    ...(canonical ? { alternates: { canonical } } : {}),
    ...(noIndex ? { robots: { index: false, follow: false } } : {}),
    openGraph: {
      title,
      description,
      type: "website",
      ...(canonical ? { url: canonical } : {}),
      // Neither catalogue records nor bounded share snapshots currently carry
      // a primary image. An explicit empty list prevents the root social card
      // from implying that it depicts this particular record.
      images: [],
    },
    twitter: {
      card: "summary",
      title,
      description,
      images: [],
    },
  };
}

export function productRecordMetadata(product: ProductDetail): Metadata {
  const brand = product.brand ? `${product.brand} ` : "";
  const description = boundedDescription(
    `${brand}${categoryLabels[product.category].toLowerCase()} record for ${product.canonical_name}. Inspect reported specifications, price observations, benchmarks, and cited review evidence.`,
  );
  const canonical = `/products/${encodeURIComponent(product.product_id)}`;
  return recordMetadata({
    title: product.canonical_name,
    description,
    canonical,
  });
}

// An unavailable record is never indexable, whether the miss is a definitive
// 404 or a transient upstream failure: indexing either one would publish a
// placeholder page as if it were catalogue evidence.
export function unavailableProductMetadata(productId: string): Metadata {
  return recordMetadata({
    title: "Product evidence unavailable",
    description: "This catalogue record is unavailable in the current BuildSignal data release.",
    canonical: `/products/${encodeURIComponent(productId)}`,
    noIndex: true,
  });
}

export function sharedBuildRecordMetadata(
  snapshot: SharedBuildSnapshot,
  { verified = false }: { verified?: boolean } = {},
): Metadata {
  const cpu = snapshot.components.find((component) => component.category === "cpu")?.canonical_name;
  const gpu = snapshot.components.find((component) => component.category === "gpu")?.canonical_name;
  const componentSummary = [cpu, gpu].filter(Boolean).join(" + ");
  const description = boundedDescription(
    `${verified ? "Verified" : "Unverified"} ${profileLabels[snapshot.profile].toLowerCase()} PC build${componentSummary ? ` with ${componentSummary}` : ""}: ${formatSgd(snapshot.total_price_sgd)} for new parts and ${formatScore(snapshot.overall_score)} relative fit. Re-run before buying.`,
  );
  return recordMetadata({
    title: `${profileLabels[snapshot.profile]} PC build`,
    description,
    noIndex: !verified,
  });
}

export function unavailableSharedBuildMetadata(): Metadata {
  return recordMetadata({
    title: "Shared PC build unavailable",
    description: "This bounded BuildSignal build snapshot is invalid, incomplete, or no longer available.",
    noIndex: true,
  });
}
