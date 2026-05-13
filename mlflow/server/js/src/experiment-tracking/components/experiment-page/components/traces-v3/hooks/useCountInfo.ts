import { useMemo } from 'react';
import {
  shouldUseInfinitePaginatedTraces,
  getEvalTabTotalTracesLimit,
} from '@databricks/web-shared/genai-traces-table';
import { useTraceMetricsQuery } from '../../../../../pages/experiment-overview/hooks/useTraceMetricsQuery';
import {
  MetricViewType,
  AggregationType,
  TraceMetricKey,
  createTraceMetadataFilter,
  SESSION_ID_METADATA_KEY,
} from '@databricks/web-shared/model-trace-explorer';
import type { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';

const getUniqueSessionCount = (traceInfos: ModelTraceInfoV3[] | undefined) =>
  new Set(
    (traceInfos ?? [])
      .map((traceInfo) => traceInfo.trace_metadata?.[SESSION_ID_METADATA_KEY])
      .filter((sessionId): sessionId is string => Boolean(sessionId)),
  ).size;

/**
 * Returns the countInfo object for the traces table toolbar badge.
 *
 * When infinite pagination is enabled, queries the trace metrics endpoint
 * to get the true total count of traces so the badge shows "X of Y" where
 * X = loaded traces and Y = total traces in the time period.
 *
 * When infinite pagination is disabled, falls back to the existing behavior
 * where totalCount comes from the metadata hook.
 */
export function useCountInfo({
  experimentIds,
  runUuid,
  timeRange,
  traceInfos,
  traceInfosCount,
  metadataTraceInfos,
  traceInfosLoading,
  metadataTotalCount,
  disabled,
  countSessions = false,
}: {
  experimentIds: string[];
  runUuid?: string;
  timeRange?: { startTime?: string; endTime?: string };
  traceInfos?: ModelTraceInfoV3[];
  traceInfosCount?: number;
  metadataTraceInfos?: ModelTraceInfoV3[];
  traceInfosLoading: boolean;
  metadataTotalCount: number;
  disabled: boolean;
  countSessions?: boolean;
}) {
  const usingInfinitePagination = shouldUseInfinitePaginatedTraces();
  const filters = useMemo(
    () => (runUuid ? [createTraceMetadataFilter('mlflow.sourceRun', runUuid)] : undefined),
    [runUuid],
  );

  const startTimeMs = timeRange?.startTime ? Number(timeRange.startTime) : undefined;
  const endTimeMs = timeRange?.endTime ? Number(timeRange.endTime) : undefined;

  const { data: countMetrics, isLoading: countLoading } = useTraceMetricsQuery({
    experimentIds,
    viewType: MetricViewType.TRACES,
    metricName: countSessions ? TraceMetricKey.SESSION_COUNT : TraceMetricKey.TRACE_COUNT,
    aggregations: [{ aggregation_type: AggregationType.COUNT }],
    startTimeMs,
    endTimeMs,
    enabled: usingInfinitePagination && !disabled,
    filters,
  });
  const metricsTotal = countMetrics?.data_points?.[0]?.values?.[AggregationType.COUNT];

  return useMemo(() => {
    const loadedSessionCount = getUniqueSessionCount(traceInfos);
    const currentCount = countSessions ? loadedSessionCount : (traceInfos?.length ?? traceInfosCount ?? 0);
    const maxAllowedCount = usingInfinitePagination ? Infinity : getEvalTabTotalTracesLimit();

    if (usingInfinitePagination) {
      const fallbackTotal = countSessions ? loadedSessionCount : (traceInfos?.length ?? traceInfosCount ?? 0);
      return {
        currentCount,
        logCountLoading: traceInfosLoading || countLoading,
        totalCount: metricsTotal ?? fallbackTotal,
        maxAllowedCount,
      };
    }

    return {
      currentCount,
      logCountLoading: traceInfosLoading,
      totalCount: countSessions ? getUniqueSessionCount(metadataTraceInfos) : metadataTotalCount,
      maxAllowedCount,
    };
  }, [
    usingInfinitePagination,
    countSessions,
    traceInfos,
    traceInfosCount,
    metadataTraceInfos,
    metadataTotalCount,
    traceInfosLoading,
    countLoading,
    metricsTotal,
  ]);
}
