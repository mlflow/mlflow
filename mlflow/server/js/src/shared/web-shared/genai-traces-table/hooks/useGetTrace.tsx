import { isNil } from 'lodash';
import { useCallback, useMemo } from 'react';

import { isV3ModelTraceInfo, type ModelTrace } from '@databricks/web-shared/model-trace-explorer';
import { useQuery } from '@databricks/web-shared/query-client';

export type GetTraceFunction = (traceId?: string, traceInfo?: ModelTrace['info']) => Promise<ModelTrace | undefined>;

export function useGetTrace(getTrace?: GetTraceFunction, traceInfo?: ModelTrace['info'], enablePolling = false) {
  const traceId = useMemo(() => {
    if (!traceInfo) {
      return undefined;
    }
    return isV3ModelTraceInfo(traceInfo) ? traceInfo.trace_id : traceInfo.request_id ?? '';
  }, [traceInfo]);

  const getTraceFn = useCallback(
    (traceInfo?: ModelTrace['info']) => {
      if (!getTrace || isNil(traceId)) {
        return Promise.resolve(undefined);
      }
      return getTrace(traceId, traceInfo);
    },
    [getTrace, traceId],
  );

  return useQuery({
    queryKey: ['getTrace', traceId],
    queryFn: () => getTraceFn(traceInfo),
    enabled: !isNil(getTrace) && !isNil(traceId),
    staleTime: Infinity, // Keep data fresh as long as the component is mounted
    refetchOnWindowFocus: false, // Disable refetching on window focus
    retry: 1,
    keepPreviousData: true,
    refetchInterval: (data) => {
      if (!enablePolling) return false;
      // Stop polling if trace is completed
      const traceState = data && isV3ModelTraceInfo(data.info) ? data.info.state : undefined;
      return traceState === 'IN_PROGRESS' ? 1000 : false;
    },
  });
}
