import { useMemo, useCallback } from 'react';
import {
  MetricViewType,
  AggregationType,
  TraceMetricKey,
  P50,
  P90,
  P99,
  getPercentileKey,
} from '@databricks/web-shared/model-trace-explorer';
import { useTraceMetricsQuery } from './useTraceMetricsQuery';
import { formatTimestampForTraceMetrics, useTimestampValueMap } from '../utils/chartUtils';
import type { OverviewChartProps } from '../types';

export interface LatencyChartDataPoint {
  name: string;
  p50: number;
  p90: number;
  p99: number;
}

export interface UseTraceLatencyChartDataResult {
  /** Processed chart data with all time buckets filled */
  chartData: LatencyChartDataPoint[];
  /** Overall average latency for the time range */
  avgLatency: number | undefined;
  /** Whether data is currently being fetched */
  isLoading: boolean;
  /** Error if data fetching failed */
  error: unknown;
  /** Whether there are any data points */
  hasData: boolean;
}

/**
 * Custom hook that fetches and processes latency chart data.
 * Encapsulates all data-fetching and processing logic for the latency chart.
 *
 * @param props - Chart props including experimentId, time range, and buckets
 * @returns Processed chart data, loading state, and error state
 */
export function useTraceLatencyChartData({
  experimentId,
  startTimeMs,
  endTimeMs,
  timeIntervalSeconds,
  timeBuckets,
}: OverviewChartProps): UseTraceLatencyChartDataResult {
  // Fetch latency metrics with p50, p90, p99 aggregations grouped by time
  const {
    data: latencyData,
    isLoading: isLoadingTimeSeries,
    error: timeSeriesError,
  } = useTraceMetricsQuery({
    experimentId,
    startTimeMs,
    endTimeMs,
    viewType: MetricViewType.TRACES,
    metricName: TraceMetricKey.LATENCY,
    aggregations: [
      { aggregation_type: AggregationType.PERCENTILE, percentile_value: P50 },
      { aggregation_type: AggregationType.PERCENTILE, percentile_value: P90 },
      { aggregation_type: AggregationType.PERCENTILE, percentile_value: P99 },
    ],
    timeIntervalSeconds,
  });

  // Fetch overall average latency (without time bucketing) for the header
  const {
    data: avgLatencyData,
    isLoading: isLoadingAvg,
    error: avgError,
  } = useTraceMetricsQuery({
    experimentId,
    startTimeMs,
    endTimeMs,
    viewType: MetricViewType.TRACES,
    metricName: TraceMetricKey.LATENCY,
    aggregations: [{ aggregation_type: AggregationType.AVG }],
  });

  const latencyDataPoints = useMemo(() => latencyData?.data_points || [], [latencyData?.data_points]);
  const isLoading = isLoadingTimeSeries || isLoadingAvg;
  const error = timeSeriesError || avgError;

  // Extract overall average latency from the response (undefined if not available)
  const avgLatency = avgLatencyData?.data_points?.[0]?.values?.[AggregationType.AVG];

  // Create a map of latency values by timestamp
  const latencyExtractor = useCallback(
    (dp: { values?: Record<string, number> }) => ({
      p50: dp.values?.[getPercentileKey(P50)] || 0,
      p90: dp.values?.[getPercentileKey(P90)] || 0,
      p99: dp.values?.[getPercentileKey(P99)] || 0,
    }),
    [],
  );
  const latencyByTimestamp = useTimestampValueMap(latencyDataPoints, latencyExtractor);

  // Prepare chart data - fill in all time buckets with 0 for missing data
  const chartData = useMemo(() => {
    return timeBuckets.map((timestampMs) => {
      const latency = latencyByTimestamp.get(timestampMs);
      return {
        name: formatTimestampForTraceMetrics(timestampMs, timeIntervalSeconds),
        p50: latency?.p50 || 0,
        p90: latency?.p90 || 0,
        p99: latency?.p99 || 0,
      };
    });
  }, [timeBuckets, latencyByTimestamp, timeIntervalSeconds]);

  return {
    chartData,
    avgLatency,
    isLoading,
    error,
    hasData: latencyDataPoints.length > 0,
  };
}
