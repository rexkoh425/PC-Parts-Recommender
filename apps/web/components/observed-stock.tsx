import { Circle, CircleAlert, CircleCheck } from "lucide-react";
import { observedStockLabel, stockTone } from "@/lib/catalogue";

const stockIcons = {
  positive: CircleCheck,
  warning: CircleAlert,
  neutral: Circle,
} as const;

export function ObservedStock({ status }: { status?: string | null }) {
  const tone = stockTone(status);
  const Icon = stockIcons[tone];

  return (
    <span className={`observed-stock observed-stock--${tone}`}>
      <Icon className="observed-stock__icon" aria-hidden="true" size={12} strokeWidth={2} />
      {observedStockLabel(status)}
    </span>
  );
}
