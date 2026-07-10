import type { BuildResult } from "./types";

export const SAVED_BUILDS_KEY = "pc-build-recommender:saved-builds:v1";

export interface SavedBuild {
  build: BuildResult;
  saved_at: string;
}

export function parseSavedBuilds(value: string | null): SavedBuild[] {
  if (!value) return [];
  try {
    const parsed: unknown = JSON.parse(value);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (entry): entry is SavedBuild =>
        Boolean(
          entry &&
            typeof entry === "object" &&
            "build" in entry &&
            entry.build &&
            typeof entry.build === "object" &&
            "build_id" in entry.build &&
            typeof entry.build.build_id === "string" &&
            "saved_at" in entry &&
            typeof entry.saved_at === "string",
        ),
    );
  } catch {
    return [];
  }
}

export function readSavedBuilds(): SavedBuild[] {
  if (typeof window === "undefined") return [];
  return parseSavedBuilds(window.localStorage.getItem(SAVED_BUILDS_KEY));
}

export function writeSavedBuilds(builds: SavedBuild[]): void {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(SAVED_BUILDS_KEY, JSON.stringify(builds));
  window.dispatchEvent(new Event("pcbr:saved-builds-changed"));
}

export function saveBuild(build: BuildResult): SavedBuild[] {
  const current = readSavedBuilds().filter((entry) => entry.build.build_id !== build.build_id);
  const next = [{ build, saved_at: new Date().toISOString() }, ...current];
  writeSavedBuilds(next);
  return next;
}

export function removeSavedBuild(buildId: string): SavedBuild[] {
  const next = readSavedBuilds().filter((entry) => entry.build.build_id !== buildId);
  writeSavedBuilds(next);
  return next;
}
