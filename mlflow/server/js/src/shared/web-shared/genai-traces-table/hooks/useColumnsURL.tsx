import { useCallback, useMemo } from 'react';

import { useSearchParams } from '../utils/RoutingUtils';

const QUERY_PARAM_KEY = 'selectedColumns';
const VALUE_SEPARATOR = ',';

/**
 * Query param-powered hook that manages selected column IDs in URL.
 * Format: comma-separated list of column IDs
 */
export const useColumnsURL = (): [
  string[] | undefined,
  (columnIds: string[] | undefined, replace?: boolean) => void,
] => {
  const [searchParams, setSearchParams] = useSearchParams();

  const selectedColumnIds: string[] | undefined = useMemo(() => {
    const columnsParam = searchParams.get(QUERY_PARAM_KEY);
    if (!columnsParam) return undefined;

    return columnsParam.split(VALUE_SEPARATOR).filter(Boolean);
  }, [searchParams]);

  const setSelectedColumnIds = useCallback(
    (columnIds: string[] | undefined, replace = false) => {
      setSearchParams(
        (params: URLSearchParams) => {
          params.delete(QUERY_PARAM_KEY);

          if (columnIds && columnIds.length > 0) {
            params.set(QUERY_PARAM_KEY, columnIds.join(VALUE_SEPARATOR));
          }

          return params;
        },
        { replace },
      );
    },
    [setSearchParams],
  );

  return [selectedColumnIds, setSelectedColumnIds];
};
