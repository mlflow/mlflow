import { useMemo } from 'react';
import {
  MetricViewType,
  AggregationType,
  TraceMetricKey,
  TraceStatus,
  TraceDimensionKey,
  TIME_BUCKET_DIMENSION_KEY,
} from '@databricks/web-shared/model-trace-explorer';
import { useTraceMetricsQuery } from './useTraceMetricsQuery';
import { formatTimestampForTraceMetrics } from '../utils/chartUtils';
import { useOverviewChartContext } from '../OverviewChartContext';

export interface ErrorsChartDataPoint {
  name: string;
  errorCount: number;
  errorRate: number;
  timestampMs: number;
}

export interface UseTraceErrorsChartDataResult {
  /** Processed chart data with all time buckets filled */
  chartData: ErrorsChartDataPoint[];
  /** Total number of errors in the time range */
  totalErrors: number;
  /** Overall error rate percentage */
  overallErrorRate: number;
  /** Average error rate across time buckets */
  avgErrorRate: number;
  /** Whether data is currently being fetched */
  isLoading: boolean;
  /** Error if data fetching failed */
  error: unknown;
  /** Whether there are any data points */
  hasData: boolean;
}

/**
 * Custom hook that fetches and processes errors chart data.
 *
 * Uses the same TRACE_COUNT query with TRACE_STATUS dimension as the requests chart,
 * so React Query deduplicates it to a single SQL execution.
 * Error counts are extracted from the ERROR status rows; total counts are the sum
 * of all status rows.
 *
 * @returns Processed chart data, loading state, and error state
 */
export function useTraceErrorsChartData(): UseTraceErrorsChartDataResult {
  const { experimentIds, startTimeMs, endTimeMs, timeIntervalSeconds, timeBuckets, filters } =
    useOverviewChartContext();

  // This query uses the same params as useTraceRequestsChartData, so React Query
  // deduplicates it into a single network request and SQL execution.
  const {
    data: traceCountData,
    isLoading,
    error,
  } = useTraceMetricsQuery({
    experimentIds,
    startTimeMs,
    endTimeMs,
    viewType: MetricViewType.TRACES,
    metricName: TraceMetricKey.TRACE_COUNT,
    aggregations: [{ aggregation_type: AggregationType.COUNT }],
    timeIntervalSeconds,
    dimensions: [TraceDimensionKey.TRACE_STATUS],
    filters,
  });

  const dataPoints = useMemo(() => traceCountData?.data_points || [], [traceCountData?.data_points]);

  // Build per-timestamp maps for error counts and total counts from the status-dimensioned data
  const { errorByTimestamp, totalByTimestamp, totalErrors, totalTraces } = useMemo(() => {
    const errorMap = new Map<number, number>();
    const totalMap = new Map<number, number>();
    let errors = 0;
    let total = 0;

    for (const dp of dataPoints) {
      const ts = new Date(dp.dimensions?.[TIME_BUCKET_DIMENSION_KEY]).getTime();
      if (isNaN(ts)) continue;

      const count = dp.values?.[AggregationType.COUNT] || 0;
      const status = dp.dimensions?.[TraceDimensionKey.TRACE_STATUS];

      totalMap.set(ts, (totalMap.get(ts) || 0) + count);
      total += count;

      if (status === TraceStatus.ERROR) {
        errorMap.set(ts, (errorMap.get(ts) || 0) + count);
        errors += count;
      }
    }

    return { errorByTimestamp: errorMap, totalByTimestamp: totalMap, totalErrors: errors, totalTraces: total };
  }, [dataPoints]);

  const overallErrorRate = totalTraces > 0 ? (totalErrors / totalTraces) * 100 : 0;

  // Prepare chart data - fill in all time buckets with 0 for missing data
  const chartData = useMemo(() => {
    return timeBuckets.map((timestampMs) => {
      const errorCount = errorByTimestamp.get(timestampMs) || 0;
      const totalCount = totalByTimestamp.get(timestampMs) || 0;
      const errorRate = totalCount > 0 ? (errorCount / totalCount) * 100 : 0;

      return {
        name: formatTimestampForTraceMetrics(timestampMs, timeIntervalSeconds),
        errorCount,
        errorRate: Math.round(errorRate * 100) / 100,
        timestampMs,
      };
    });
  }, [timeBuckets, errorByTimestamp, totalByTimestamp, timeIntervalSeconds]);

  // Calculate average error rate across time buckets for the reference line
  const avgErrorRate = useMemo(() => {
    if (chartData.length === 0) return 0;
    const sum = chartData.reduce((acc, dp) => acc + dp.errorRate, 0);
    return sum / chartData.length;
  }, [chartData]);

  return {
    chartData,
    totalErrors,
    overallErrorRate,
    avgErrorRate,
    isLoading,
    error,
    hasData: dataPoints.length > 0,
  };
}
