import { clampScore, formatScore } from "@/lib/format";

export function ScoreMeter({
  label,
  value,
  compact = false,
}: {
  label: string;
  value: number | null | undefined;
  compact?: boolean;
}) {
  const score = clampScore(value);
  const available = typeof value === "number" && Number.isFinite(value);

  return (
    <div className={`score-meter ${compact ? "score-meter--compact" : ""}`}>
      <div className="score-meter__header">
        <span>{label}</span>
        <strong>{available ? formatScore(value) : "Not enough evidence"}</strong>
      </div>
      <div
        className="score-meter__track"
        role="meter"
        aria-label={`${label} score`}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={available ? score : undefined}
        aria-valuetext={available ? `${formatScore(value)} out of 100` : "Not enough evidence"}
      >
        <span style={{ width: `${score}%` }} />
      </div>
    </div>
  );
}
