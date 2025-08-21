import { useCallback } from 'react';
// the upstream source of this file is in web-shared,
// so we can't use mlflow shared hooks
// eslint-disable-next-line no-restricted-imports
import { useSearchParams } from 'react-router-dom';

const QUERY_PARAM_KEY = 'selectedSpanId';
/**
 * Query param-powered hook that returns the currently selected span ID and a function to set the selected span ID.
 * To be used in traces page components.
 */
export const useSelectedSpanId = () => {
  const [searchParams, setSearchParams] = useSearchParams();

  const selectedSpanId = searchParams.get(QUERY_PARAM_KEY) ?? undefined;
  const setSelectedSpanId = useCallback(
    (selectedSpanId: string | undefined) => {
      setSearchParams(
        (params) => {
          const newParams = new URLSearchParams(params);
          if (selectedSpanId === undefined) {
            newParams.delete(QUERY_PARAM_KEY);
            return newParams;
          } else {
            newParams.set(QUERY_PARAM_KEY, selectedSpanId);
            return newParams;
          }
        },
        { replace: true },
      );
    },
    [setSearchParams],
  );
  return [selectedSpanId, setSelectedSpanId] as const;
};