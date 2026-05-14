import type { Metadata } from "next";
import { RecommendationsScreen } from "@/components/recommendations-screen";

export const metadata: Metadata = {
  title: "Your recommendations",
  robots: { index: false, follow: false },
};

export default async function RecommendationsPage({
  params,
}: {
  params: Promise<{ requestId: string }>;
}) {
  const { requestId } = await params;
  return <RecommendationsScreen requestId={requestId} />;
}
