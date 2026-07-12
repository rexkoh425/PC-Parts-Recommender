import type { Metadata } from "next";
import { SavedScreen } from "@/components/saved-screen";

export const metadata: Metadata = {
  title: "Saved builds",
  robots: { index: false, follow: false },
};

export default function SavedPage() {
  return <SavedScreen />;
}
