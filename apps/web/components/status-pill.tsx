import { Check, Search, TriangleAlert, X } from "lucide-react";
import type { CompatibilityStatus } from "@/lib/types";

function StatusIcon({ status }: { status: CompatibilityStatus }) {
  if (status === "pass") return <Check size={11} strokeWidth={2.4} />;
  if (status === "warning") return <TriangleAlert size={11} strokeWidth={2} />;
  if (status === "fail") return <X size={11} strokeWidth={2.4} />;
  return <Search size={11} strokeWidth={2.2} />;
}

export function StatusPill({
  status,
  label,
}: {
  status: CompatibilityStatus;
  label?: string;
}) {
  const defaultLabels: Record<CompatibilityStatus, string> = {
    pass: "Compatible",
    warning: "Compatible with notes",
    unknown: "Needs verification",
    fail: "Not compatible",
  };

  return (
    <span className={`status-pill status-pill--${status}`}>
      <span aria-hidden="true" className="status-pill__icon">
        <StatusIcon status={status} />
      </span>
      {label ?? defaultLabels[status]}
    </span>
  );
}
