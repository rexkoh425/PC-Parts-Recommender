"use client";

import { useCallback, useMemo, useSyncExternalStore } from "react";
import {
  parseSavedBuilds,
  readSavedBuilds,
  removeSavedBuild,
  saveBuild,
  SAVED_BUILDS_KEY,
} from "./saved-builds";
import type { BuildSummary } from "./types";

export function useSavedBuilds() {
  const subscribe = useCallback((onStoreChange: () => void) => {
    window.addEventListener("storage", onStoreChange);
    window.addEventListener("pcbr:saved-builds-changed", onStoreChange);
    return () => {
      window.removeEventListener("storage", onStoreChange);
      window.removeEventListener("pcbr:saved-builds-changed", onStoreChange);
    };
  }, []);

  const serialized = useSyncExternalStore(
    subscribe,
    () => window.localStorage.getItem(SAVED_BUILDS_KEY) ?? "[]",
    () => "[]",
  );
  const entries = useMemo(() => parseSavedBuilds(serialized), [serialized]);

  const savedIds = useMemo(
    () => new Set(entries.map((entry) => entry.build.build_id)),
    [entries],
  );

  const toggle = useCallback((build: BuildSummary) => {
    if (readSavedBuilds().some((entry) => entry.build.build_id === build.build_id)) {
      removeSavedBuild(build.build_id);
      return false;
    }
    saveBuild(build);
    return true;
  }, []);

  const remove = useCallback((buildId: string) => {
    removeSavedBuild(buildId);
  }, []);

  return { entries, savedIds, toggle, remove, ready: true };
}
