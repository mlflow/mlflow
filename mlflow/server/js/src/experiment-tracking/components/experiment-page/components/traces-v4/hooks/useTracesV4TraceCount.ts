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

interface UseTracesV4TraceCountOptions {
  /**
   * The active free-text search is an exact trace-id (see `isExactTraceIdSearch`). Such a search
   * ignores the time range, so the time-scoped metrics total doesn't apply — the total is the
   * resolved row count instead.
   */
  isExactTraceIdSearch?: boolean;
  /**
   * The row query is still fetching (or holding previous-page rows via `keepPreviousData`). While
   * this is true the current page's row count isn't yet the exact-id result, so we must not report
   * it as the total.
   */
  isResultLoading?: boolean;
}

/**
 * The "{n} of {total}" footer count for the V4 traces tab, mirroring the prior tab's `useCountInfo`:
 * the current count is the current page's row count, and the total comes from the trace-metrics
 * endpoint (the cursor search API returns no total), scoped to the same experiment and time range.
 *
 * An exact trace-id search is the one case where the search ignores the time range (see
 * `buildFilter`): it resolves the trace via the indexed `request_id` lookup regardless of the
 * selected window. The time-scoped metrics total is then meaningless (it would read 0 for a trace
 * older than the window, giving the "1 of 0" mismatch), so we skip the metrics query and use the
 * resolved row count — but only once the row query has settled, otherwise the previous page's
 * (retained) row count would briefly show as the total.
 */
export const useTracesV4TraceCount = (
  experimentId: string,
  currentPageCount: number,
  timeRange: StartEndTime,
  { isExactTraceIdSearch = false, isResultLoading = false }: UseTracesV4TraceCountOptions = {},
): TracesV4TraceCount => {
  const experimentIds = useMemo(() => [experimentId], [experimentId]);
  const startTimeMs = timeRange.startTime ? Number(timeRange.startTime) : undefined;
  const endTimeMs = timeRange.endTime ? Number(timeRange.endTime) : undefined;

  const { data, isLoading, isFetching } = useTraceMetricsQuery({
    experimentIds,
    viewType: MetricViewType.TRACES,
    metricName: TraceMetricKey.TRACE_COUNT,
    aggregations: [{ aggregation_type: AggregationType.COUNT }],
    startTimeMs,
    endTimeMs,
    enabled: !isExactTraceIdSearch,
  });

  if (isExactTraceIdSearch) {
    // The total is the resolved rows, trustworthy only after the row query settles. Until then keep
    // the footer count loading so the previous page's retained rows don't flash as "N of N".
    return {
      currentCount: currentPageCount,
      totalCount: isResultLoading ? undefined : currentPageCount,
      isTotalLoading: isResultLoading,
    };
  }

  // Normal path: the time-scoped metrics total. `isFetching` (not just `isLoading`) gates the value
  // so that when the query is re-enabled on leaving trace-id mode, we spin until the fresh total
  // lands rather than briefly surfacing the prior fetch's cached (differently-scoped) count.
  return {
    currentCount: currentPageCount,
    totalCount: isFetching ? undefined : data?.data_points?.[0]?.values?.[AggregationType.COUNT],
    isTotalLoading: isLoading || isFetching,
  };
};
