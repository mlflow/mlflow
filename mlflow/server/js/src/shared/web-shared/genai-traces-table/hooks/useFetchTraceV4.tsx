import invariant from 'invariant';
import { first } from 'lodash';
import { useCallback } from 'react';

import { useQueryClient } from '@databricks/web-shared/query-client';

import { type ModelTraceInfoV3, TracesServiceV4 } from '../../model-trace-explorer';

const FETCH_TRACE_V4_QUERY_KEY = 'FETCH_TRACE_V4_QUERY_KEY';

type UseFetchTraceV4Params = never;

export const useFetchTraceV4LazyQuery = (params: UseFetchTraceV4Params) => {
  const queryClient = useQueryClient();
  return useCallback(
    (traceInfo?: ModelTraceInfoV3) => {
      return queryClient.ensureQueryData({
        queryKey: [FETCH_TRACE_V4_QUERY_KEY, traceInfo?.trace_id],
        queryFn: async () => {
          invariant(traceInfo?.trace_id, 'Trace ID is required to fetch trace');
          const traceResponse = await TracesServiceV4.getBatchTracesV4({
            traceIds: [traceInfo.trace_id],
            traceLocation: traceInfo.trace_location,
          });

          return first(
            // Convert response to the commonly used format
            traceResponse.traces.map((trace) => ({
              info: trace.trace_info,
              data: { spans: trace.spans },
            })),
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
