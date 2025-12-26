import { useCallback } from 'react';
import { searchMlflowTracesQueryFn, SEARCH_MLFLOW_TRACES_QUERY_KEY } from '@databricks/web-shared/genai-traces-table';
import { QueryClient, useQueryClient } from '@databricks/web-shared/query-client';
import { EvaluateTracesParams } from './types';
import { groupTracesBySession } from '../../../shared/web-shared/genai-traces-table/sessions-table/utils';
import { ModelTraceInfoV3 } from '../../../shared/web-shared/model-trace-explorer';
import { isEmpty } from 'lodash';

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
    traceInfos,
  }));

  if (itemIds && !isEmpty(itemIds)) {
    return sessionArray.filter((session) => itemIds.includes(session.sessionId));
  }

  return sessionArray.slice(0, modifiedItemCount);
};

type SessionsForEvaluation = {
  traceInfos: ModelTraceInfoV3[];
  sessionId?: string;
}[];

export const useGetSessionIdsForEvaluation = () => {
  const queryClient = useQueryClient();

  return useCallback(
    async (params: EvaluateTracesParams): Promise<SessionsForEvaluation> => {
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
