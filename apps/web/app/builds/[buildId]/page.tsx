import type { Metadata } from "next";
import { BuildDetailScreen } from "@/components/build-detail-screen";

export const metadata: Metadata = {
  title: "Build details",
  robots: { index: false, follow: false },
};

export default async function BuildDetailsPage({
  params,
}: {
  params: Promise<{ buildId: string }>;
}) {
  const { buildId } = await params;
  return <BuildDetailScreen buildId={buildId} />;
}
