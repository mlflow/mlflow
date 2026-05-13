import { useCallback } from 'react';

import { useSearchParams } from '../utils/RoutingUtils';

const QUERY_PARAM_KEY = 'searchQuery';

/**
 * Query param-powered hook that returns the search query for evaluations.
 */
export const useEvaluationsSearchQuery = () => {
  const [searchParams, setSearchParams] = useSearchParams();

  const searchQuery = searchParams.get(QUERY_PARAM_KEY) ?? '';

  const setSearchQuery = useCallback(
    (value: string | undefined) => {
      setSearchParams((params) => {
        if (value === undefined || value === '') {
          params.delete(QUERY_PARAM_KEY);
          return params;
        }
        params.set(QUERY_PARAM_KEY, value);
        return params;
      });
    },
    [setSearchParams],
  );

  return [searchQuery, setSearchQuery] as const;
};
