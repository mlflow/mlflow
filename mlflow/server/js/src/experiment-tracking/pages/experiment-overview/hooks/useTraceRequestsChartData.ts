import { useMemo, useCallback } from 'react';
import { MetricViewType, AggregationType, TraceMetricKey } from '@databricks/web-shared/model-trace-explorer';
import { useTraceMetricsQuery } from './useTraceMetricsQuery';
import { formatTimestampForTraceMetrics, useTimestampValueMap } from '../utils/chartUtils';
import type { OverviewChartProps } from '../types';

export interface RequestsChartDataPoint {
  name: string;
  count: number;
}

export interface UseTraceRequestsChartDataResult {
  /** Processed chart data with all time buckets filled */
  chartData: RequestsChartDataPoint[];
  /** Total number of requests in the time range */
  totalRequests: number;
  /** Average requests per time bucket */
  avgRequests: number;
  /** Whether data is currently being fetched */
  isLoading: boolean;
  /** Error if data fetching failed */
  error: unknown;
  /** Whether there are any data points */
  hasData: boolean;
}

/**
 * Custom hook that fetches and processes requests chart data.
 * Encapsulates all data-fetching and processing logic for the requests chart.
 *
 * @param props - Chart props including experimentId, time range, and buckets
 * @returns Processed chart data, loading state, and error state
 */
export function useTraceRequestsChartData({
  experimentId,
  startTimeMs,
  endTimeMs,
  timeIntervalSeconds,
  timeBuckets,
}: OverviewChartProps): UseTraceRequestsChartDataResult {
  // Fetch trace count metrics grouped by time bucket
  const {
    data: traceCountData,
    isLoading,
    error,
  } = useTraceMetricsQuery({
    experimentId,
    startTimeMs,
    endTimeMs,
    viewType: MetricViewType.TRACES,
    metricName: TraceMetricKey.TRACE_COUNT,
    aggregations: [{ aggregation_type: AggregationType.COUNT }],
    timeIntervalSeconds,
  });

  const traceCountDataPoints = useMemo(() => traceCountData?.data_points || [], [traceCountData?.data_points]);

  // Get total requests
  const totalRequests = useMemo(
    () => traceCountDataPoints.reduce((sum, dp) => sum + (dp.values?.[AggregationType.COUNT] || 0), 0),
    [traceCountDataPoints],
  );

  // Calculate average requests per time bucket
  const avgRequests = useMemo(() => totalRequests / timeBuckets.length, [totalRequests, timeBuckets.length]);

  // Create a map of counts by timestamp using shared utility
  const countExtractor = useCallback(
    (dp: { values?: Record<string, number> }) => dp.values?.[AggregationType.COUNT] || 0,
    [],
  );
  const countByTimestamp = useTimestampValueMap(traceCountDataPoints, countExtractor);

  // Prepare chart data - fill in all time buckets with 0 for missing data
  const chartData = useMemo(() => {
    return timeBuckets.map((timestampMs) => ({
      name: formatTimestampForTraceMetrics(timestampMs, timeIntervalSeconds),
      count: countByTimestamp.get(timestampMs) || 0,
    }));
  }, [timeBuckets, countByTimestamp, timeIntervalSeconds]);

  return {
    chartData,
    totalRequests,
    avgRequests,
    isLoading,
    error,
    hasData: traceCountDataPoints.length > 0,
  };
}
