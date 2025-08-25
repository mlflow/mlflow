import { useCallback } from 'react';
import { useSearchParams } from '../../common/utils/RoutingUtils';

const QUERY_PARAM_KEY = 'viewState';

export type MonitoringViewState = 'charts' | 'logs' | 'insights';

/**
 * Query param-powered hook that returns the view state from the URL.
 */
export const useMonitoringViewState = () => {
  const [searchParams, setSearchParams] = useSearchParams();

  const viewState = (searchParams.get(QUERY_PARAM_KEY) ?? 'logs') as MonitoringViewState;

  const setViewState = useCallback(
    (viewState: MonitoringViewState | undefined, replace = false) => {
      setSearchParams(
        (params) => {
          if (viewState === undefined) {
            params.delete(QUERY_PARAM_KEY);
            return params;
          }
          params.set(QUERY_PARAM_KEY, viewState);
          return params;
        },
        { replace },
      );
    },
    [setSearchParams],
  );

  return [viewState, setViewState] as const;
};
