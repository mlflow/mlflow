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
} from '@databricks/web-shared/model-trace-explorer';
import {
  getAbsoluteStartEndTime,
  useMonitoringFilters,
} from '@mlflow/mlflow/src/experiment-tracking/hooks/useMonitoringFilters';
import { useMonitoringConfig } from '@mlflow/mlflow/src/experiment-tracking/hooks/useMonitoringConfig';

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
  traceInfosCount,
  traceInfosLoading,
  metadataTotalCount,
  disabled,
}: {
  experimentIds: string[];
  runUuid?: string;
  traceInfosCount: number | undefined;
  traceInfosLoading: boolean;
  metadataTotalCount: number;
  disabled: boolean;
}) {
  const usingInfinitePagination = shouldUseInfinitePaginatedTraces();
  const filters = useMemo(
    () => (runUuid ? [createTraceMetadataFilter('mlflow.sourceRun', runUuid)] : undefined),
    [runUuid],
  );

  // Use getAbsoluteStartEndTime to properly compute time range from labels
  const [monitoringFilters] = useMonitoringFilters();
  const monitoringConfig = useMonitoringConfig();
  const { startTime, endTime } = useMemo(
    () => getAbsoluteStartEndTime(monitoringConfig.dateNow, monitoringFilters),
    [monitoringConfig.dateNow, monitoringFilters],
  );

  // Convert ISO strings to milliseconds for the API
  const startTimeMs = startTime ? new Date(startTime).getTime() : undefined;
  const endTimeMs = endTime ? new Date(endTime).getTime() : undefined;

  const { data: traceCountMetrics, isLoading: traceCountLoading } = useTraceMetricsQuery({
    experimentIds,
    viewType: MetricViewType.TRACES,
    metricName: TraceMetricKey.TRACE_COUNT,
    aggregations: [{ aggregation_type: AggregationType.COUNT }],
    startTimeMs,
    endTimeMs,
    enabled: usingInfinitePagination && !disabled,
    filters,
  });
  const metricsTotal = traceCountMetrics?.data_points?.[0]?.values?.[AggregationType.COUNT];

  return useMemo(() => {
    if (usingInfinitePagination) {
      return {
        currentCount: traceInfosCount,
        logCountLoading: traceInfosLoading || traceCountLoading,
        totalCount: metricsTotal ?? traceInfosCount ?? 0,
        maxAllowedCount: Infinity,
      };
    }
    return {
      currentCount: traceInfosCount,
      logCountLoading: traceInfosLoading,
      totalCount: metadataTotalCount,
      maxAllowedCount: getEvalTabTotalTracesLimit(),
    };
  }, [
    usingInfinitePagination,
    traceInfosCount,
    metadataTotalCount,
    traceInfosLoading,
    traceCountLoading,
    metricsTotal,
  ]);
}
