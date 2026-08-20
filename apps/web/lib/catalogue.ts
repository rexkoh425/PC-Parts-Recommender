import type { CompatibilityCheck, CompatibilityStatus, PriceObservation } from "./types";

const dateFormatter = new Intl.DateTimeFormat("en-SG", {
  dateStyle: "medium",
  timeStyle: "short",
  timeZone: "Asia/Singapore",
});

export function observedStockLabel(status?: string | null): string {
  if (!status) return "Availability not reported";
  const normalized = status.trim().toLowerCase().replaceAll("-", "_").replaceAll(" ", "_");
  const labels: Record<string, string> = {
    demo_only: "Demo catalogue",
    in_stock: "Observed in stock",
    available: "Observed available",
    low_stock: "Observed low stock",
    out_of_stock: "Observed out of stock",
    unavailable: "Observed unavailable",
    preorder: "Observed as pre-order",
    pre_order: "Observed as pre-order",
  };
  return labels[normalized] ?? `Last observed: ${humanizeAttributeKey(status)}`;
}

export function stockTone(status?: string | null): "positive" | "warning" | "neutral" {
  if (!status) return "neutral";
  const normalized = status.toLowerCase().replaceAll("-", "_").replaceAll(" ", "_");
  if (["in_stock", "available"].includes(normalized)) return "positive";
  if (["low_stock", "preorder", "pre_order"].includes(normalized)) return "warning";
  return "neutral";
}

export function priceObservationPresentation(
  observation: Pick<PriceObservation, "condition" | "current_offer_eligible">,
): { conditionLabel: string; eligibilityLabel: string } {
  return {
    conditionLabel: `Condition: ${humanizeAttributeKey(observation.condition)}`,
    eligibilityLabel: observation.current_offer_eligible
      ? "Eligible current offer"
      : "Not eligible as a current offer",
  };
}

export function humanizeAttributeKey(value: string): string {
  return value
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .replaceAll("_", " ")
    .replaceAll("-", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase())
    .replace(/\bCpu\b/g, "CPU")
    .replace(/\bGpu\b/g, "GPU")
    .replace(/\bPsu\b/g, "PSU")
    .replace(/\bRam\b/g, "RAM")
    .replace(/\bVram\b/g, "VRAM")
    .replace(/\bPcie\b/g, "PCIe")
    .replace(/\bDdr\b/g, "DDR")
    .replace(/\bAtx\b/g, "ATX")
    .replace(/\bGb\b/g, "GB")
    .replace(/\bTb\b/g, "TB")
    .replace(/\bMm\b/g, "mm");
}

export function formatAttributeValue(value: unknown): string {
  if (value === null || value === undefined || value === "") return "Not reported";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "number") return new Intl.NumberFormat("en-SG").format(value);
  if (Array.isArray(value)) {
    return value.length ? value.map(formatAttributeValue).join(", ") : "Not reported";
  }
  if (typeof value === "object") {
    return Object.entries(value as Record<string, unknown>)
      .map(([key, nested]) => `${humanizeAttributeKey(key)}: ${formatAttributeValue(nested)}`)
      .join(" · ");
  }
  return humanizeAttributeToken(String(value));
}

/**
 * Enumerated attribute values arrive as machine tokens: the live catalogue
 * stores "80_plus_gold" and "atx", which the fixture never exercised because
 * its values were already prose. Anything that looks like a token is spaced
 * and capitalised; free text is returned untouched so product names and
 * descriptions are not mangled.
 */
function humanizeAttributeToken(value: string): string {
  const token = value.trim();
  if (!token) return "Not reported";

  // Free text: leave alone. A token has no spaces and is not mixed case.
  const looksLikeToken = /^[a-z0-9]+(?:[_-][a-z0-9]+)*$/.test(token);
  if (!looksLikeToken) return token;

  const spaced = token.replaceAll("_", " ").replaceAll("-", " ");

  // Capitalise the first letter of each word, not every letter.
  const titled = spaced.replace(
    /(^|\s)(\w)/g,
    (_match, lead: string, first: string) => `${lead}${first.toUpperCase()}`,
  );

  // Units and standards read wrong in plain title case. Whole words only,
  // so a fragment inside a longer word is not rewritten.
  return titled
    .replace(/\bAtx\b/g, "ATX")
    .replace(/\bMatx\b/g, "mATX")
    .replace(/\bItx\b/g, "ITX")
    .replace(/\bDdr(\d)\b/g, "DDR$1")
    .replace(/\bPcie\b/g, "PCIe")
    .replace(/\bNvme\b/g, "NVMe")
    .replace(/\bSsd\b/g, "SSD")
    .replace(/\bHdd\b/g, "HDD")
    .replace(/\bRgb\b/g, "RGB")
    .replace(/\bWifi\b/g, "Wi-Fi")
    .replace(/\bUsb\b/g, "USB");
}

export function confidencePresentation(confidence?: number | null): {
  label: string;
  percent?: number;
  tone: "high" | "medium" | "low" | "unknown";
} {
  if (typeof confidence !== "number" || !Number.isFinite(confidence)) {
    return { label: "Confidence not reported", tone: "unknown" };
  }
  const bounded = Math.max(0, Math.min(1, confidence));
  const percent = Math.round(bounded * 100);
  if (bounded >= 0.85) return { label: `High confidence · ${percent}%`, percent, tone: "high" };
  if (bounded >= 0.65) return { label: `Medium confidence · ${percent}%`, percent, tone: "medium" };
  return { label: `Limited confidence · ${percent}%`, percent, tone: "low" };
}

export function formatEvidenceTimestamp(value?: string | null): string {
  if (!value) return "Timestamp not reported";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Timestamp not reported";
  return `${dateFormatter.format(date)} SGT`;
}

export function formatSignedDelta(value: number | null | undefined, unit: string): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "Not modelled";
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(Math.abs(value) < 10 ? 1 : 0)}${unit}`;
}

export interface CompatibilitySummary {
  total: number;
  pass: number;
  warning: number;
  unknown: number;
  fail: number;
  overall: CompatibilityStatus;
}

export function summarizeCompatibilityChecks(checks: CompatibilityCheck[]): CompatibilitySummary {
  const counts = checks.reduce(
    (summary, check) => ({ ...summary, [check.status]: summary[check.status] + 1 }),
    { pass: 0, warning: 0, unknown: 0, fail: 0 },
  );
  const overall: CompatibilityStatus = counts.fail
    ? "fail"
    : counts.unknown
      ? "unknown"
      : counts.warning
        ? "warning"
        : "pass";
  return { total: checks.length, ...counts, overall };
}
