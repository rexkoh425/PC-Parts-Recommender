import type { CompatibilityStatus } from "@/lib/types";

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
        {status === "pass" ? "✓" : status === "warning" ? "!" : status === "fail" ? "×" : "?"}
      </span>
      {label ?? defaultLabels[status]}
    </span>
  );
}
