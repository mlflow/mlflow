import { useCallback } from 'react';

import { useQuery, useQueryClient, type UseQueryOptions } from '@databricks/web-shared/query-client';

import { type ModelTraceInfoV3, TracesServiceV4 } from '../../model-trace-explorer';
import type { TraceV3 } from '../types';

const FETCH_TRACE_V4_QUERY_KEY = 'FETCH_TRACE_V4_QUERY_KEY';

type UseFetchTraceV4Params = never;

/**
 * Hook for lazy fetching of V4 traces.
 */
export const useFetchTraceV4LazyQuery = (params: UseFetchTraceV4Params) => {
  const queryClient = useQueryClient();
  return useCallback(
    (traceInput?: TraceInput) => {
      if (!traceInput) return Promise.resolve(undefined);
      return queryClient.ensureQueryData({
        queryKey: getTraceV4QueryKey(traceInput),
        queryFn: async () => {
          return TracesServiceV4.getTraceV4(
            // prettier-ignore
            traceInput,
          );
        },
      });
    },
    // prettier-ignore
    [
      queryClient,
    ],
  );
};

/**
 * Hook for immediate fetching of V4 traces.
 *
 * @param traceInput - Trace identifier (serialized string like "trace:/catalog.schema/trace_id" or ModelTraceInfoV3 object)
 * @param params - Fetch parameters including SQL warehouse ID
 * @param options - Additional React Query options
 */
export const useFetchTraceV4Query = <T = TraceV3 | null>(
  traceInput: TraceInput | undefined,
  params: UseFetchTraceV4Params,
  options?: Omit<UseQueryOptions<TraceV3 | null, unknown, T>, 'queryFn'>,
) => {
  const enabled = options?.enabled ?? Boolean(traceInput);
  const result = useQuery({
    queryKey: getTraceV4QueryKey(traceInput),
    queryFn: async () => {
      if (!traceInput) return null;
      return TracesServiceV4.getTraceV4(
        // prettier-ignore
        traceInput,
      );
    },
    enabled,
    staleTime: Infinity,
    refetchOnReconnect: false,
    refetchOnWindowFocus: false,
    ...options,
  });

  return {
    ...result,
    // Re-return isLoading with enabled condition to prevent misleading loading states when query is disabled
    isLoading: result.isLoading && enabled,
  };
};

/**
 * Input can be either:
 * 1. ModelTraceInfoV3 object with trace_id and trace_location
 * 2. Serialized trace ID string (format: "trace:/catalog.schema/trace_id")
 */
type TraceInput = ModelTraceInfoV3 | string;

/**
 * Generates query key for v4 trace fetching.
 * Uses the trace ID: either the string passed in as is, or the trace_id from the ModelTraceInfoV3 object.
 */
export const getTraceV4QueryKey = (traceInput: TraceInput | undefined) => {
  if (!traceInput) return [FETCH_TRACE_V4_QUERY_KEY, undefined];

  if (typeof traceInput === 'string') {
    return [FETCH_TRACE_V4_QUERY_KEY, traceInput];
  }

  return [FETCH_TRACE_V4_QUERY_KEY, traceInput.trace_id];
};
