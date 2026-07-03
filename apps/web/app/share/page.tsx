import { Suspense } from "react";
import type { Metadata } from "next";
import { SharedBuildScreen } from "@/components/shared-build-screen";
import { ApiError, getBuildShare } from "@/lib/api";
import {
  sharedBuildRecordMetadata,
  unavailableSharedBuildMetadata,
} from "@/lib/record-metadata";
import {
  decodeSharedBuild,
  sharedSnapshotFromApi,
  type SharedBuildSnapshot,
} from "@/lib/shared-build";

type ShareSearchParams = Promise<{
  build?: string | string[];
  share?: string | string[];
}>;

function firstValue(value: string | string[] | undefined): string | undefined {
  return Array.isArray(value) ? value[0] : value;
}

interface ShareMetadataResult {
  snapshot?: SharedBuildSnapshot;
  verified: boolean;
  definitiveMissing: boolean;
}

async function metadataSnapshot(searchParams: ShareSearchParams): Promise<ShareMetadataResult> {
  const params = await searchParams;
  const shareId = firstValue(params.share);
  if (shareId) {
    if (!/^[A-Za-z0-9_-]{1,80}$/.test(shareId)) {
      return { verified: false, definitiveMissing: true };
    }
    try {
      return {
        snapshot: sharedSnapshotFromApi((await getBuildShare(shareId)).snapshot),
        verified: true,
        definitiveMissing: false,
      };
    } catch (error) {
      return {
        verified: false,
        definitiveMissing: error instanceof ApiError && error.status === 404,
      };
    }
  }
  const snapshot = decodeSharedBuild(firstValue(params.build));
  return {
    snapshot,
    verified: false,
    definitiveMissing: !snapshot,
  };
}

export async function generateMetadata({
  searchParams,
}: {
  searchParams: ShareSearchParams;
}): Promise<Metadata> {
  const result = await metadataSnapshot(searchParams);
  return result.snapshot
    ? sharedBuildRecordMetadata(result.snapshot, { verified: result.verified })
    : unavailableSharedBuildMetadata({ definitiveMissing: result.definitiveMissing });
}

export default function SharedBuildPage() {
  return (
    <Suspense>
      <SharedBuildScreen />
    </Suspense>
  );
}
