import { Suspense } from "react";
import type { Metadata } from "next";
import { SharedBuildScreen } from "@/components/shared-build-screen";

export const metadata: Metadata = {
  title: "Shared PC build snapshot",
  description: "A portable, generation-time snapshot of a BuildSignal PC recommendation.",
  robots: { index: false, follow: false },
};

export default function SharedBuildPage() {
  return (
    <Suspense>
      <SharedBuildScreen />
    </Suspense>
  );
}
