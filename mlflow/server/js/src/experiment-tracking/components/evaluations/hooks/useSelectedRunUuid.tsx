import { useCallback } from 'react';
import { useSearchParams } from '@mlflow/mlflow/src/common/utils/RoutingUtils';

const QUERY_PARAM_KEY = 'selectedRunUuid';

/**
 * Query param-powered hook that returns the selected run uuid.
 */
export const useSelectedRunUuid = () => {
  const [searchParams, setSearchParams] = useSearchParams();

  const selectedRunUuid = searchParams.get(QUERY_PARAM_KEY) ?? undefined;

  const setSelectedRunUuid = useCallback(
    (selectedRunUuid: string | undefined) => {
      setSearchParams(
        (params) => {
          if (selectedRunUuid === undefined) {
            params.delete(QUERY_PARAM_KEY);
            return params;
          }
          params.set(QUERY_PARAM_KEY, selectedRunUuid);
          return params;
        },
        { replace: true },
      );
    },
    [setSearchParams],
  );

  return [selectedRunUuid, setSelectedRunUuid] as const;
};
