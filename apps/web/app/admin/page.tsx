import type { Metadata } from "next";
import { AdminOperationsScreen } from "@/components/admin-operations-screen";

export const metadata: Metadata = {
  title: "Catalogue operations",
  description: "Restricted, read-only catalogue operations for BuildSignal administrators.",
  robots: { index: false, follow: false },
};

export default function AdminOperationsPage() {
  return <AdminOperationsScreen />;
}
