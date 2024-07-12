import { useCallback } from 'react';
import { useSearchParams } from '../../../../common/utils/RoutingUtils';

const QUERY_PARAM_KEY = 'selectedTraceId';

/**
 * Query param-powered hook that returns the currently selected trace ID and a function to set the selected trace ID.
 * To be used in traces page components.
 */
export const useActiveExperimentTrace = () => {
  // TODO(ML-40722): Create separate UI route for traces page and use route params instead of search params
  const [searchParams, setSearchParams] = useSearchParams();

  const selectedTraceId = searchParams.get(QUERY_PARAM_KEY) ?? undefined;

  const setSelectedTraceId = useCallback(
    (selectedTraceId: string | undefined) => {
      setSearchParams((params) => {
        if (selectedTraceId === undefined) {
          params.delete(QUERY_PARAM_KEY);
          return params;
        }
        params.set(QUERY_PARAM_KEY, selectedTraceId);
        return params;
      });
    },
    [setSearchParams],
  );

  return [selectedTraceId, setSelectedTraceId] as const;
};
