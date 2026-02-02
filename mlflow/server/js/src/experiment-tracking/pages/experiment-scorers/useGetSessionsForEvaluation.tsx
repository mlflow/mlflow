import { useCallback } from 'react';
import { searchMlflowTracesQueryFn, SEARCH_MLFLOW_TRACES_QUERY_KEY } from '@databricks/web-shared/genai-traces-table';
import { QueryClient, useQueryClient } from '@databricks/web-shared/query-client';
import { groupTracesBySession } from '@databricks/web-shared/genai-traces-table/sessions-table/utils';
import { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import { EvaluateTracesParams } from './types';
import { isEmpty, sortBy } from 'lodash';

const fetchSessions = async (
  queryClient: QueryClient,
  { itemCount, itemIds, locations }: Pick<EvaluateTracesParams, 'itemCount' | 'itemIds' | 'locations'>,
) => {
  const modifiedItemCount = itemIds && !isEmpty(itemIds) ? itemIds.length : itemCount;

  const traces = await queryClient.fetchQuery({
    queryKey: [
      SEARCH_MLFLOW_TRACES_QUERY_KEY,
      {
        locations,
        orderBy: ['timestamp DESC'],
        pageSize: 500,
      },
    ],
    queryFn: ({ signal }) =>
      searchMlflowTracesQueryFn({
        signal,
        locations,
        pageSize: 500,
        limit: 500,
        orderBy: ['timestamp DESC'],
      }),
    staleTime: Infinity,
    cacheTime: Infinity,
  });

  const sessions = groupTracesBySession(traces);

  const sessionArray = Object.entries(sessions).map(([sessionId, traceInfos]) => ({
    sessionId,
    traceInfos: sortBy(traceInfos, (trace) => new Date(trace.request_time)),
  }));

  if (itemIds && !isEmpty(itemIds)) {
    return sessionArray.filter((session) => itemIds.includes(session.sessionId));
  }

  return sessionArray.slice(0, modifiedItemCount);
};

export type SessionForEvaluation = {
  traceInfos: ModelTraceInfoV3[];
  sessionId?: string;
};

export const useGetSessionsForEvaluation = () => {
  const queryClient = useQueryClient();

  return useCallback(
    async (params: EvaluateTracesParams): Promise<SessionForEvaluation[]> => {
      const { itemCount: traceCount = 0, locations, itemIds } = params;

      return fetchSessions(queryClient, {
        locations,
        itemIds,
        itemCount: traceCount,
      });
    },
    [queryClient],
  );
};
