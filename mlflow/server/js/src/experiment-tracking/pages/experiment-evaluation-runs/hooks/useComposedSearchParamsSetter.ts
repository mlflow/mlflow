import { useCallback, useRef } from 'react';
import type { NavigateOptions } from '../../../../common/utils/RoutingUtils';

type SetRouterSearchParams = (params: URLSearchParams, navigateOptions?: NavigateOptions) => void;

/**
 * react-router hands each functional `setSearchParams` updater the params from the last render, so
 * several updates in one handler (clear search + clauses + date range) overwrite each other. This
 * setter applies each updater to the result of the previous one instead.
 */
export const useComposedSearchParamsSetter = (
  searchParams: URLSearchParams,
  setSearchParams: SetRouterSearchParams,
) => {
  const latestSearchRef = useRef(searchParams.toString());
  latestSearchRef.current = searchParams.toString();

  return useCallback(
    (update: (params: URLSearchParams) => URLSearchParams, navigateOptions?: NavigateOptions) => {
      const nextParams = update(new URLSearchParams(latestSearchRef.current));
      latestSearchRef.current = nextParams.toString();
      setSearchParams(nextParams, navigateOptions);
    },
    [setSearchParams],
  );
};
