import { useCallback } from 'react';
import { useSearchParams } from '@mlflow/mlflow/src/common/utils/RoutingUtils';

const QUERY_PARAM_KEY = 'selectedArtifact';

/**
 * Query param-powered hook that returns the selected artifact information.
 * Used to persist artifact selection in the URL for the artifacts tab.
 *
 * The selectedArtifact format is: "sourceId:artifactPath"
 * - sourceId: The ID of the source (run UUID or model ID)
 * - artifactPath: The path to the artifact within that source
 */
export const useSelectedArtifactBySearchParam = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const selectedArtifact = searchParams.get(QUERY_PARAM_KEY) ?? undefined;

  const setSelectedArtifact = useCallback(
    (sourceId: string, artifactPath: string) => {
      setSearchParams(
        (params) => {
          if (!sourceId || !artifactPath) {
            params.delete(QUERY_PARAM_KEY);
            return params;
          }
          params.set(QUERY_PARAM_KEY, `${sourceId}:${artifactPath}`);
          return params;
        },
        { replace: true },
      );
    },
    [setSearchParams],
  );

  const clearSelectedArtifact = useCallback(() => {
    setSearchParams(
      (params) => {
        params.delete(QUERY_PARAM_KEY);
        return params;
      },
      { replace: true },
    );
  }, [setSearchParams]);

  const parsedSelectedArtifact = selectedArtifact?.includes(':')
    ? {
        sourceId: selectedArtifact.split(':')?.[0],
        artifactPath: selectedArtifact.split(':').slice(1).join(':'),
      }
    : undefined;

  return [parsedSelectedArtifact, setSelectedArtifact, clearSelectedArtifact] as const;
};
