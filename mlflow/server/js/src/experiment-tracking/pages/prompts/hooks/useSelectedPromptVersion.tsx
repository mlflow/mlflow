import { useCallback } from 'react';
import { PROMPT_VERSION_QUERY_PARAM } from '../utils';
import { useSearchParams } from '@mlflow/mlflow/src/common/utils/RoutingUtils';

/**
 * Query param-powered hook that returns the selected prompt version.
 */
export const useSelectedPromptVersion = (latestVersion?: string) => {
  const [searchParams, setSearchParams] = useSearchParams();

  const selectedPromptVersion = searchParams.get(PROMPT_VERSION_QUERY_PARAM) ?? latestVersion;

  const setSelectedPromptVersion = useCallback(
    (selectedPromptVersion: string | undefined) => {
      setSearchParams(
        (params) => {
          if (selectedPromptVersion === undefined) {
            params.delete(PROMPT_VERSION_QUERY_PARAM);
            return params;
          }
          params.set(PROMPT_VERSION_QUERY_PARAM, selectedPromptVersion);
          return params;
        },
        { replace: true },
      );
    },
    [setSearchParams],
  );

  return [selectedPromptVersion, setSelectedPromptVersion] as const;
};
