import { useCallback } from 'react';
import { searchMlflowTracesQueryFn, SEARCH_MLFLOW_TRACES_QUERY_KEY } from '@databricks/web-shared/genai-traces-table';
import { QueryClient, useQueryClient } from '@databricks/web-shared/query-client';
import { EvaluateTracesParams } from './types';
import { DEFAULT_TRACE_COUNT } from './constants';
import { isEmpty } from 'lodash';

const fetchTracesAndGetIds = async (
  queryClient: QueryClient,
  { itemCount, locations }: Required<Pick<EvaluateTracesParams, 'itemCount' | 'locations'>>,
) => {
  const modifiedTraceCount = Math.max(itemCount, DEFAULT_TRACE_COUNT);

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
    .slice(0, itemCount);

  return traceIds;
};

export const useGetTraceIdsForEvaluation = () => {
  const queryClient = useQueryClient();

  return useCallback(
    async (params: EvaluateTracesParams) => {
      const { itemCount: traceCount = 0, locations, itemIds } = params;
      if (itemIds && !isEmpty(itemIds)) {
        return itemIds;
      }
      const fetchedTraceIds = await fetchTracesAndGetIds(queryClient, {
        locations,
        itemCount: traceCount,
      });
      return fetchedTraceIds;
    },
    [queryClient],
  );
};
