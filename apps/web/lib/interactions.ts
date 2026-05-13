export function oneBasedRank(zeroBasedIndex: number): number {
  if (!Number.isInteger(zeroBasedIndex) || zeroBasedIndex < 0) {
    throw new RangeError("rank index must be a non-negative integer");
  }
  return zeroBasedIndex + 1;
}
