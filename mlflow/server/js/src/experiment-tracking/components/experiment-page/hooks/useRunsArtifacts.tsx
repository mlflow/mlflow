import { useEffect, useState } from 'react';
import { listArtifactsApi } from '../../../actions';
import type { ArtifactListFilesResponse } from '../../../types';

/**
 * Fetches artifacts given a list of run UUIDs
 * @param runUuids List of run UUIDs
 * @returns Object containing artifacts keyed by run UUID
 */
export const useRunsArtifacts = (runUuids: string[]) => {
  const [artifactsKeyedByRun, setArtifactsKeyedByRun] = useState<Record<string, ArtifactListFilesResponse>>({});
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchArtifacts = async () => {
      setIsLoading(true);
      setError(null);

      const artifactsByRun: Record<string, ArtifactListFilesResponse> = {};

      try {
        await Promise.all(
          runUuids.map(async (runUuid) => {
            const response = listArtifactsApi(runUuid);
            const artifacts = (await response.payload) as ArtifactListFilesResponse;
            artifactsByRun[runUuid] = artifacts;
          }),
        );
        setArtifactsKeyedByRun(artifactsByRun);
      } catch (err: any) {
        setError(err);
      } finally {
        setIsLoading(false);
      }
    };

    if (runUuids.length > 0) {
      fetchArtifacts();
    } else {
      setArtifactsKeyedByRun({});
      setIsLoading(false);
    }
  }, [runUuids]);

  return { artifactsKeyedByRun, isLoading, error };
};
