import { useCallback, useEffect, useMemo } from 'react';
import { type QueryClient, useQuery, useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { getTrace as getTraceV3 } from '@mlflow/mlflow/src/experiment-tracking/utils/TraceUtils';
import {
  shouldUseTracesV4API,
  useFetchTraceV4LazyQuery,
  createTraceLocationForDestinationPath,
} from '@databricks/web-shared/genai-traces-table';
import {
  parseV4TraceId,
  isV4TraceId,
  type ModelTrace,
  type ModelTraceInfoV3,
} from '@databricks/web-shared/model-trace-explorer';

const MAX_CACHED_TRACES = 5;
const MAX_CACHED_TRACE_CHARS = 64 * 1024 * 1024;
const traceStringChars = new WeakMap<ModelTrace, number>();

const countTraceStringChars = (trace: ModelTrace): number => {
  const cached = traceStringChars.get(trace);
  if (cached !== undefined) return cached;
  let chars = 0;
  const values: unknown[] = [trace.data];
  while (values.length > 0 && chars <= MAX_CACHED_TRACE_CHARS) {
    const value = values.pop();
    if (typeof value === 'string') {
      chars += value.length;
    } else if (Array.isArray(value)) {
      for (const item of value) values.push(item);
    } else if (value && typeof value === 'object') {
      for (const item of Object.values(value)) values.push(item);
    }
  }
  traceStringChars.set(trace, chars);
  return chars;
};

export const pruneTraceCache = (queryClient: QueryClient): void => {
  const traces = queryClient
    .getQueryCache()
    .findAll({ queryKey: ['getTrace'] })
    .flatMap((query) => {
      const trace = query.state.data as ModelTrace | undefined;
      return trace
        ? [
            {
              queryKey: query.queryKey,
              chars: countTraceStringChars(trace),
              updatedAt: query.state.dataUpdatedAt,
              active: query.getObserversCount() > 0,
            },
          ]
        : [];
    });
  const cachedTraces = traces.filter((trace) => !trace.active).sort((left, right) => left.updatedAt - right.updatedAt);
  let totalChars = traces.reduce((total, trace) => total + trace.chars, 0);
  while (cachedTraces.length > MAX_CACHED_TRACES || totalChars > MAX_CACHED_TRACE_CHARS) {
    const oldest = cachedTraces.shift();
    if (!oldest) break;
    queryClient.removeQueries({ queryKey: oldest.queryKey, exact: true });
    totalChars -= oldest.chars;
  }
};

/**
 * Fetches a single trace's body for the V4 drawer. Parses the `trace:/…` long id itself, so it also
 * serves deep-links whose row isn't on the current page. `shouldUseTracesV4API()` is `false` in OSS,
 * so this always resolves via the V3 tracking-store path (`getTraceV3`); the V4-API branch is kept
 * so a future backend can enable it without a code change.
 *
 * Inlined in the V4 dir because OSS has no shared/app-tree equivalent; the V4 drawer is its only
 * consumer here.
 */
export function useGetTrace(traceId: string, selectedSqlWarehouseId?: string, rowTraceInfo?: ModelTraceInfoV3) {
  const queryClient = useQueryClient();
  // V4 format is "trace:/catalog.schema/trace_id"; V3 is the bare id.
  const parsedTrace = useMemo(() => {
    if (!traceId) return null;
    if (isV4TraceId(traceId)) {
      return parseV4TraceId(traceId);
    }
    return { trace_id: traceId, trace_location: null };
  }, [traceId]);

  // OSS's `useFetchTraceV4LazyQuery` takes no params (the `selectedSqlWarehouseId` arg is
  // Databricks-only). The V4-API branch below is unreachable in OSS (`shouldUseTracesV4API()` is
  // false), so this is only ever constructed, not called.
  const getTraceV4 = useFetchTraceV4LazyQuery(undefined as never);

  const fetchTrace = useCallback(async (): Promise<ModelTrace | undefined> => {
    if (!parsedTrace) return undefined;

    let trace: ModelTrace | undefined;
    if (shouldUseTracesV4API() && parsedTrace.trace_location && selectedSqlWarehouseId) {
      const location = createTraceLocationForDestinationPath(parsedTrace.trace_location);
      // Spread `rowTraceInfo` first so the parsed id and recomputed location win; the row contributes
      // only timing (`request_time`/`execution_duration`), which `getTraceV4` uses to derive time hints.
      trace = await getTraceV4({ ...rowTraceInfo, ...parsedTrace, trace_location: location } as ModelTraceInfoV3);
    } else {
      // Forward the row's trace info so OSS's `getTrace` can read its spans-location tag and take the
      // tracking-store `get-trace` path (rather than falling through to the artifact route).
      trace = await getTraceV3(parsedTrace.trace_id, rowTraceInfo);
    }

    // The next task runs after React Query stores the result, including when the drawer closed mid-request.
    setTimeout(() => pruneTraceCache(queryClient), 0);
    return trace;
  }, [parsedTrace, getTraceV4, selectedSqlWarehouseId, rowTraceInfo, queryClient]);

  const enabled = Boolean(traceId);
  const queryKey = useMemo(() => ['getTrace', traceId, selectedSqlWarehouseId], [traceId, selectedSqlWarehouseId]);
  const result = useQuery({
    queryKey,
    queryFn: fetchTrace,
    enabled,
    staleTime: 5 * 60 * 1000,
    cacheTime: 10 * 60 * 1000,
    retry: 3,
    refetchOnWindowFocus: false,
  });

  useEffect(
    () => () => {
      // Let the query observer detach before pruning inactive entries.
      queueMicrotask(() => pruneTraceCache(queryClient));
    },
    [queryClient, queryKey],
  );

  return {
    ...result,
    // Re-derive isLoading with the enabled guard so a disabled query doesn't report a misleading load.
    isLoading: result.isLoading && enabled,
  };
}
