import { useCallback } from 'react';
import { useSearchParams } from '@mlflow/mlflow/src/common/utils/RoutingUtils';

const QUERY_PARAM_KEY = 'compareToRunUuid';

/**
 * Query param-powered hook that returns the compare to run uuid when comparison is enabled.
 */
export const useCompareToRunUuid = () => {
  const [searchParams, setSearchParams] = useSearchParams();

  const compareToRunUuid = searchParams.get(QUERY_PARAM_KEY) ?? undefined;

  const setCompareToRunUuid = useCallback(
    (compareToRunId: string | undefined) => {
      setSearchParams((params) => {
        if (compareToRunId === undefined) {
          params.delete(QUERY_PARAM_KEY);
          return params;
        }
        params.set(QUERY_PARAM_KEY, compareToRunId);
        return params;
      });
    },
    [setSearchParams],
  );

  return [compareToRunUuid, setCompareToRunUuid] as const;
};
