import { useCallback } from 'react';

import { createTraceV4LongIdentifier, type ModelTraceInfoV3 } from '../../model-trace-explorer';
import { useSearchParams } from '../utils/RoutingUtils';
import { doesTraceSupportV4API } from '../utils/TraceLocationUtils';

const QUERY_PARAM_KEY = 'selectedEvaluationId';

/**
 * Query param-powered hook that returns the currently selected evaluation ID
 * and a function to set the selected evaluation ID.
 */
export const useActiveEvaluation = () => {
  const [searchParams, setSearchParams] = useSearchParams();

  const selectedEvaluationId = searchParams.get(QUERY_PARAM_KEY) ?? undefined;

  const setSelectedEvaluationId = useCallback(
    (
      selectedEvaluationId: string | undefined,
      traceInfo?: ModelTraceInfoV3,
      additionalParams?: Record<string, string>,
    ) => {
      setSearchParams((params) => {
        if (selectedEvaluationId === undefined) {
          params.delete(QUERY_PARAM_KEY);
          return params;
        }

        if (additionalParams) {
          Object.entries(additionalParams).forEach(([key, value]) => {
            params.set(key, value);
          });
        }

        // If the trace supports V4 identifiers, use this format instead.
        if (traceInfo && doesTraceSupportV4API(traceInfo)) {
          const longIdentifier = createTraceV4LongIdentifier(traceInfo);
          params.set(QUERY_PARAM_KEY, longIdentifier);

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
