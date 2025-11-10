import { compact, isNil } from 'lodash';
import { useMemo } from 'react';

import { isV3ModelTraceInfo, type ModelTrace } from '@databricks/web-shared/model-trace-explorer';
import { useQuery } from '@databricks/web-shared/query-client';

export type GetTraceFunction = (traceId?: string, traceInfo?: ModelTrace['info']) => Promise<ModelTrace | undefined>;

// unfortunately the util from model-trace-explorer
// requires the whole trace object, not just the info
function getModelTraceId(traceInfo: ModelTrace['info']): string {
  return isV3ModelTraceInfo(traceInfo) ? traceInfo.trace_id : traceInfo.request_id ?? '';
}

export function useGetTraces(getTrace?: GetTraceFunction, traceInfos?: ModelTrace['info'][]) {
  const traceIds = useMemo(() => {
    if (!traceInfos) {
      return [];
    }
    return traceInfos.map(getModelTraceId);
  }, [traceInfos]);

  return useQuery({
    queryKey: ['getTrace', traceIds],
    queryFn: async () => {
      const traces = await Promise.all(
        (traceInfos ?? []).map((traceInfo) => getTrace?.(getModelTraceId(traceInfo), traceInfo)),
      );
      return compact(traces);
    },
    enabled: !isNil(getTrace) && traceIds.length > 0,
    refetchOnWindowFocus: false,
    retry: 1,
    keepPreviousData: true,
  });
}
