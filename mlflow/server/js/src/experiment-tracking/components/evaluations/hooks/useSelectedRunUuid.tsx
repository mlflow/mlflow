import { useCallback } from 'react';
import { useSearchParams } from '@mlflow/mlflow/src/common/utils/RoutingUtils';

export const SELECTED_RUN_UUID_QUERY_PARAM = 'selectedRunUuid';

/**
 * Query param-powered hook that returns the selected run uuid.
 */
export const useSelectedRunUuid = () => {
  const [searchParams, setSearchParams] = useSearchParams();

  const selectedRunUuid = searchParams.get(SELECTED_RUN_UUID_QUERY_PARAM) ?? undefined;

  const setSelectedRunUuid = useCallback(
    (selectedRunUuid: string | undefined) => {
      setSearchParams(
        (params) => {
          if (selectedRunUuid === undefined) {
            params.delete(SELECTED_RUN_UUID_QUERY_PARAM);
            return params;
          }
          params.set(SELECTED_RUN_UUID_QUERY_PARAM, selectedRunUuid);
          return params;
        },
        { replace: true },
      );
    },
    [setSearchParams],
  );

  return [selectedRunUuid, setSelectedRunUuid] as const;
};
