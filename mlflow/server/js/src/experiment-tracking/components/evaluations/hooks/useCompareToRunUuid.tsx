import { useCallback } from 'react';
import { useSearchParams } from '@mlflow/mlflow/src/common/utils/RoutingUtils';

export const COMPARE_TO_RUN_UUID_QUERY_PARAM = 'compareToRunUuid';

/**
 * Query param-powered hook that returns the compare to run uuid when comparison is enabled.
 */
export const useCompareToRunUuid = () => {
  const [searchParams, setSearchParams] = useSearchParams();

  const compareToRunUuid = searchParams.get(COMPARE_TO_RUN_UUID_QUERY_PARAM) ?? undefined;

  const setCompareToRunUuid = useCallback(
    (compareToRunId: string | undefined) => {
      setSearchParams((params) => {
        if (compareToRunId === undefined) {
          params.delete(COMPARE_TO_RUN_UUID_QUERY_PARAM);
          return params;
        }
        params.set(COMPARE_TO_RUN_UUID_QUERY_PARAM, compareToRunId);
        return params;
      });
    },
    [setSearchParams],
  );

  return [compareToRunUuid, setCompareToRunUuid] as const;
};
