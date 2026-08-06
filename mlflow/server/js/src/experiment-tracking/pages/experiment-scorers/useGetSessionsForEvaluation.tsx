import { useCallback } from 'react';
import { searchMlflowTracesQueryFn, SEARCH_MLFLOW_TRACES_QUERY_KEY } from '@databricks/web-shared/genai-traces-table';
import type { QueryClient } from '@databricks/web-shared/query-client';
import { useQueryClient } from '@databricks/web-shared/query-client';
import { groupTracesBySession } from '@databricks/web-shared/genai-traces-table';
import { SESSION_ID_METADATA_KEY, type ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import type { EvaluateTracesParams } from './types';
import { isEmpty, sortBy } from 'lodash';

// Number of most recent traces scanned when resolving the "latest N sessions" (no explicit selection).
const LATEST_SESSIONS_TRACE_WINDOW = 500;

// Escape a value for use inside a single-quoted search filter string literal.
const escapeFilterValue = (value: string) => value.replace(/\\/g, '\\\\').replace(/'/g, "\\'");

// Trace search supports only `=` on request metadata (no IN), so selected sessions
// are resolved with one filter per session ID.
const buildSessionIdFilter = (sessionId: string) =>
  `metadata.\`${SESSION_ID_METADATA_KEY}\` = '${escapeFilterValue(sessionId)}'`;

// Resolves explicitly selected sessions with targeted per-session metadata queries.
// Scanning only the most recent traces (as the "latest N sessions" path does) silently
// drops selected sessions that fall outside that window, resulting in an empty evaluation.
const fetchTracesForSelectedSessions = async (
  queryClient: QueryClient,
  locations: EvaluateTracesParams['locations'],
  sessionIds: string[],
): Promise<ModelTraceInfoV3[]> => {
  const tracesPerSession = await Promise.all(
    sessionIds.map((sessionId) => {
      const filter = buildSessionIdFilter(sessionId);
      return queryClient.fetchQuery({
        queryKey: [
          SEARCH_MLFLOW_TRACES_QUERY_KEY,
          {
            locations,
            // The filter must be part of the key: results differ per session and the
            // query is cached indefinitely (staleTime: Infinity).
            filter,
            orderBy: ['timestamp DESC'],
            pageSize: 500,
          },
        ],
        queryFn: ({ signal }) =>
          searchMlflowTracesQueryFn({
            signal,
            locations,
            filter,
            pageSize: 500,
            // No explicit limit: a session can exceed 500 turns, and truncating it
            // would silently evaluate a partial conversation.
            orderBy: ['timestamp DESC'],
          }),
        staleTime: Infinity,
        cacheTime: Infinity,
      });
    }),
  );
  return tracesPerSession.flat();
};

const fetchSessions = async (
  queryClient: QueryClient,
  { itemCount, itemIds, locations }: Pick<EvaluateTracesParams, 'itemCount' | 'itemIds' | 'locations'>,
) => {
  const hasSelectedSessions = Boolean(itemIds && !isEmpty(itemIds));

  const traces = hasSelectedSessions
    ? await fetchTracesForSelectedSessions(queryClient, locations, itemIds ?? [])
    : await queryClient.fetchQuery({
        queryKey: [
          SEARCH_MLFLOW_TRACES_QUERY_KEY,
          {
            locations,
            orderBy: ['timestamp DESC'],
            pageSize: LATEST_SESSIONS_TRACE_WINDOW,
          },
        ],
        queryFn: ({ signal }) =>
          searchMlflowTracesQueryFn({
            signal,
            locations,
            pageSize: LATEST_SESSIONS_TRACE_WINDOW,
            limit: LATEST_SESSIONS_TRACE_WINDOW,
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

  return sessionArray.slice(0, itemCount);
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
