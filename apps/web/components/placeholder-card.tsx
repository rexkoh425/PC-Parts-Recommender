// Temporary card used while the real build card was being designed.

export function PlaceholderCard({ label }: { label: string }) {
  return (
    <div className="rounded border border-dashed p-4 text-sm opacity-60">
      {label}
    </div>
  );
}
