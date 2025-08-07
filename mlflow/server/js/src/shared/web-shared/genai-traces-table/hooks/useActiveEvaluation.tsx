import { useCallback } from 'react';

import { useSearchParams } from '../utils/RoutingUtils';

const QUERY_PARAM_KEY = 'selectedEvaluationId';

/**
 * Query param-powered hook that returns the currently selected evaluation ID
 * and a function to set the selected evaluation ID.
 */
export const useActiveEvaluation = () => {
  const [searchParams, setSearchParams] = useSearchParams();

  const selectedEvaluationId = searchParams.get(QUERY_PARAM_KEY) ?? undefined;

  const setSelectedEvaluationId = useCallback(
    (selectedEvaluationId: string | undefined) => {
      setSearchParams((params) => {
        if (selectedEvaluationId === undefined) {
          params.delete(QUERY_PARAM_KEY);
          return params;
        }
        params.set(QUERY_PARAM_KEY, selectedEvaluationId);
        return params;
      });
    },
    [setSearchParams],
  );

  return [selectedEvaluationId, setSelectedEvaluationId] as const;
};
