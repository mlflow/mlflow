import { compact } from 'lodash';

import { useQueries } from '@databricks/web-shared/query-client';

import { useArrayMemo } from './useArrayMemo';
import { doesTraceSupportV4API, parseV4TraceIdToObject } from '../ModelTraceExplorer.utils';
import { TracesServiceV3, TracesServiceV4 } from '../api';

const GET_TRACE_BY_ID = 'GET_TRACE_BY_ID';

type UseGetTracesByIdProps = any;

/**
 * Fetches multiple traces by their IDs
 * @param traceIds Accepts both v3 trace IDs ("tr-...") and v4 trace IDs ("trace:/<location>/<trace_id>")
 */
export const useGetTracesById = (traceIds: string[], params: UseGetTracesByIdProps = {}) => {
  const queries = useQueries({
    queries: traceIds.map((traceId) => {
      return {
        queryKey: [GET_TRACE_BY_ID, traceId],
        queryFn: async () => {
          // Use either full trace ID (trace:/) or fallback to just the v3 trace ID
          const maybeParsedTraceId = parseV4TraceIdToObject(traceId);

          return TracesServiceV3.getTraceV3(maybeParsedTraceId?.trace_id ?? traceId);
        },
        enabled: Boolean(traceId),
        refetchOnWindowFocus: false,
        retry: 1,
        keepPreviousData: true,
      };
    }),
  });

  const data = useArrayMemo(compact(queries.map((query) => query.data)));
  const isFetching = queries.some((query) => query.isFetching);
  const isLoading = queries.some((query) => query.isLoading);

  return {
    data,
    isFetching,
    isLoading,
  };
};
