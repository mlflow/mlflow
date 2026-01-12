import { useCallback, useMemo } from 'react';

import type { EvaluationsOverviewTableSort } from '../types';
import { useSearchParams } from '../utils/RoutingUtils';

const QUERY_PARAM_KEY = 'sort';
const VALUE_SEPARATOR = '::';

/**
 * Query param-powered hook that manages table sort state in URL.
 * Format: key::type::asc
 */
export const useTableSortURL = (): [
  EvaluationsOverviewTableSort | undefined,
  (sort: EvaluationsOverviewTableSort | undefined, replace?: boolean) => void,
] => {
  const [searchParams, setSearchParams] = useSearchParams();

  const tableSort: EvaluationsOverviewTableSort | undefined = useMemo(() => {
    const sortParam = searchParams.get(QUERY_PARAM_KEY);
    if (!sortParam) return undefined;

    const [key, type, ascStr] = sortParam.split(VALUE_SEPARATOR);
    if (!key || !type || !ascStr) return undefined;

    return {
      key,
      type: type as any,
      asc: ascStr === 'true',
    };
  }, [searchParams]);

  const setTableSort = useCallback(
    (newSort: EvaluationsOverviewTableSort | undefined, replace = false) => {
      setSearchParams(
        (params: URLSearchParams) => {
          params.delete(QUERY_PARAM_KEY);

          if (newSort) {
            params.set(QUERY_PARAM_KEY, [newSort.key, newSort.type, newSort.asc].join(VALUE_SEPARATOR));
          }

          return params;
        },
        { replace },
      );
    },
    [setSearchParams],
  );

  return [tableSort, setTableSort];
};
