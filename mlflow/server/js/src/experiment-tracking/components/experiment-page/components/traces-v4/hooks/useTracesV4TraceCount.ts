import { useMemo } from 'react';
import { AggregationType, MetricViewType, TraceMetricKey } from '@databricks/web-shared/model-trace-explorer';
import { useTraceMetricsQuery } from '@mlflow/mlflow/src/experiment-tracking/pages/experiment-overview/hooks/useTraceMetricsQuery';
import type { StartEndTime } from '../utils/timeRange';

export interface TracesV4TraceCount {
  /** Rows on the current page. */
  currentCount: number;
  /** Total traces matching the current experiment + time range, from the trace-metrics endpoint. */
  totalCount: number | undefined;
  /** True while the total is still resolving (the current count is always known immediately). */
  isTotalLoading: boolean;
}

/**
 * The "{n} of {total}" footer count for the V4 traces tab, mirroring the prior tab's `useCountInfo`:
 * the current count is the current page's row count, and the total comes from the trace-metrics
 * endpoint (the cursor search API returns no total), scoped to the same experiment and time range.
 */
export const useTracesV4TraceCount = (
  experimentId: string,
  currentPageCount: number,
  timeRange: StartEndTime,
): TracesV4TraceCount => {
  const experimentIds = useMemo(() => [experimentId], [experimentId]);
  const startTimeMs = timeRange.startTime ? Number(timeRange.startTime) : undefined;
  const endTimeMs = timeRange.endTime ? Number(timeRange.endTime) : undefined;

  const { data, isLoading } = useTraceMetricsQuery({
    experimentIds,
    viewType: MetricViewType.TRACES,
    metricName: TraceMetricKey.TRACE_COUNT,
    aggregations: [{ aggregation_type: AggregationType.COUNT }],
    startTimeMs,
    endTimeMs,
  });

  const totalCount = data?.data_points?.[0]?.values?.[AggregationType.COUNT];

  return { currentCount: currentPageCount, totalCount, isTotalLoading: isLoading };
};
