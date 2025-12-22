import { useCallback } from 'react';
import { searchMlflowTracesQueryFn, SEARCH_MLFLOW_TRACES_QUERY_KEY } from '@databricks/web-shared/genai-traces-table';
import { QueryClient, useQueryClient } from '@databricks/web-shared/query-client';
import { EvaluateTracesParams } from './types';
import { DEFAULT_TRACE_COUNT } from './constants';
import { isEmpty } from 'lodash';

const fetchTracesAndGetIds = async (
  queryClient: QueryClient,
  { traceCount, locations }: Required<Pick<EvaluateTracesParams, 'traceCount' | 'locations'>>,
) => {
  const modifiedTraceCount = Math.max(traceCount, DEFAULT_TRACE_COUNT);

  const traces = await queryClient.fetchQuery({
    queryKey: [
      SEARCH_MLFLOW_TRACES_QUERY_KEY,
      {
        locations,
        orderBy: ['timestamp DESC'],
        pageSize: modifiedTraceCount,
      },
    ],
    queryFn: ({ signal }) =>
      searchMlflowTracesQueryFn({
        signal,
        locations,
        pageSize: modifiedTraceCount,
        limit: modifiedTraceCount,
        orderBy: ['timestamp DESC'],
      }),
    staleTime: Infinity,
    cacheTime: Infinity,
  });

  // Extract trace IDs from search results
  const traceIds = traces
    .map((trace) => trace.trace_id)
    .filter((id): id is string => Boolean(id))
    .slice(0, traceCount);

  return traceIds;
};

export const useGetTraceIdsForEvaluation = () => {
  const queryClient = useQueryClient();

  return useCallback(
    async (params: EvaluateTracesParams) => {
      const { traceCount = 0, locations, traceIds } = params;
      if (traceIds && !isEmpty(traceIds)) {
        return traceIds;
      }
      const fetchedTraceIds = await fetchTracesAndGetIds(queryClient, {
        locations,
        traceCount,
      });
      return fetchedTraceIds;
    },
    [queryClient],
  );
};
