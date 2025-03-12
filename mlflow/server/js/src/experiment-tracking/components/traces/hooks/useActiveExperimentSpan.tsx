import { useCallback } from 'react';
import { useSearchParams } from '../../../../common/utils/RoutingUtils';

const QUERY_PARAM_KEY = 'selectedSpanId';

/**
 * Query param-powered hook that returns the currently selected span ID and a function to set the selected span ID.
 * To be used in traces page components.
 */
export const useActiveExperimentSpan = () => {
  const [searchParams, setSearchParams] = useSearchParams();

  const selectedSpanId = searchParams.get(QUERY_PARAM_KEY) ?? undefined;

  const setSelectedSpanId = useCallback(
    (selectedSpanId: string | undefined) => {
      setSearchParams(
        (params) => {
          if (selectedSpanId === undefined) {
            params.delete(QUERY_PARAM_KEY);
            return params;
          }
          params.set(QUERY_PARAM_KEY, selectedSpanId);
          return params;
        },
        { replace: true },
      );
    },
    [setSearchParams],
  );

  return [selectedSpanId, setSelectedSpanId] as const;
};
