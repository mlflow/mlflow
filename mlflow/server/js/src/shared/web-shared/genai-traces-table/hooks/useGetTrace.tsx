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

  const getRefreshInterval = (data: ModelTrace | undefined) => {
    // Keep polling until trace is completed and span counts matches with the number logged in the
    // trace info. The latter check is to avoid race condition where the trace status is finalized
    // before child spans arrive at the backend.
    const traceInfo = data && isV3ModelTraceInfo(data.info) ? data.info : undefined;

    if (!traceInfo || traceInfo.state === 'IN_PROGRESS') return 1000;

    const traceStats = traceInfo.trace_metadata?.['mlflow.trace.sizeStats'];

    // If the stats metadata is not available, stop polling.
    if (!traceStats) return false;

    const expected = JSON.parse(traceStats).num_spans;
    const actual = data?.data?.spans?.length ?? 0;
    return expected === actual ? false : 1000;
  };

  return useQuery({
    queryKey: ['getTrace', traceId],
    queryFn: () => getTraceFn(traceInfo),
    enabled: !isNil(getTrace) && !isNil(traceId),
    staleTime: Infinity, // Keep data fresh as long as the component is mounted
    refetchOnWindowFocus: false, // Disable refetching on window focus
    retry: 1,
    keepPreviousData: true,
    refetchInterval: enablePolling ? getRefreshInterval : false,
  });
}
