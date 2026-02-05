import { createContext, useCallback, useContext } from 'react';

import { createTraceV4LongIdentifier, type ModelTraceInfoV3 } from '../../model-trace-explorer';
import { useSearchParams } from '../utils/RoutingUtils';
import { doesTraceSupportV4API } from '../utils/TraceLocationUtils';

const QUERY_PARAM_KEY = 'selectedEvaluationId';

type SetSelectedEvaluationIdFn = (selectedEvaluationId: string | undefined, traceInfo?: ModelTraceInfoV3) => void;

/**
 * Context to override the active evaluation state.
 * When provided, useActiveEvaluation will use the context value instead of URL params.
 * This is useful for components like SelectTracesModal that need to render a traces table
 * without inheriting the parent's selected evaluation state.
 */
export const ActiveEvaluationContext = createContext<{
  selectedEvaluationId: string | undefined;
  setSelectedEvaluationId: SetSelectedEvaluationIdFn;
} | null>(null);

/**
 * Query param-powered hook that returns the currently selected evaluation ID
 * and a function to set the selected evaluation ID.
 *
 * If ActiveEvaluationContext is provided, uses context value instead of URL params.
 */
export const useActiveEvaluation = () => {
  const context = useContext(ActiveEvaluationContext);
  const [searchParams, setSearchParams] = useSearchParams();

  const selectedEvaluationIdFromUrl = searchParams.get(QUERY_PARAM_KEY) ?? undefined;

  const setSelectedEvaluationIdFromUrl: SetSelectedEvaluationIdFn = useCallback(
    (selectedEvaluationId, traceInfo) => {
      setSearchParams((params) => {
        if (selectedEvaluationId === undefined) {
          params.delete(QUERY_PARAM_KEY);
          return params;
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

  // If context is provided, use context values instead of URL params
  if (context) {
    return [context.selectedEvaluationId, context.setSelectedEvaluationId] as const;
  }

  return [selectedEvaluationIdFromUrl, setSelectedEvaluationIdFromUrl] as const;
};
