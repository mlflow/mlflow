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

export interface TokenStatsChartDataPoint {
  name: string;
  p50: number;
  p90: number;
  p99: number;
}

export interface UseTraceTokenStatsChartDataResult {
  /** Processed chart data with all time buckets filled */
  chartData: TokenStatsChartDataPoint[];
  /** Overall average tokens per trace */
  avgTokens: number | undefined;
  /** Whether data is currently being fetched */
  isLoading: boolean;
  /** Error if data fetching failed */
  error: unknown;
  /** Whether there are any data points */
  hasData: boolean;
}

/**
 * Custom hook that fetches and processes token stats chart data.
 * Encapsulates all data-fetching and processing logic for the token stats chart.
 *
 * @param props - Chart props including experimentId, time range, and buckets
 * @returns Processed chart data, loading state, and error state
 */
export function useTraceTokenStatsChartData({
  experimentId,
  startTimeMs,
  endTimeMs,
  timeIntervalSeconds,
  timeBuckets,
}: OverviewChartProps): UseTraceTokenStatsChartDataResult {
  // Fetch token stats with p50, p90, p99 aggregations grouped by time
  const {
    data: tokenStatsData,
    isLoading: isLoadingTimeSeries,
    error: timeSeriesError,
  } = useTraceMetricsQuery({
    experimentId,
    startTimeMs,
    endTimeMs,
    viewType: MetricViewType.TRACES,
    metricName: TraceMetricKey.TOTAL_TOKENS,
    aggregations: [
      { aggregation_type: AggregationType.PERCENTILE, percentile_value: P50 },
      { aggregation_type: AggregationType.PERCENTILE, percentile_value: P90 },
      { aggregation_type: AggregationType.PERCENTILE, percentile_value: P99 },
    ],
    timeIntervalSeconds,
  });

  // Fetch overall average tokens (without time bucketing) for the header
  const {
    data: avgTokensData,
    isLoading: isLoadingAvg,
    error: avgError,
  } = useTraceMetricsQuery({
    experimentId,
    startTimeMs,
    endTimeMs,
    viewType: MetricViewType.TRACES,
    metricName: TraceMetricKey.TOTAL_TOKENS,
    aggregations: [{ aggregation_type: AggregationType.AVG }],
  });

  const tokenStatsDataPoints = useMemo(() => tokenStatsData?.data_points || [], [tokenStatsData?.data_points]);
  const isLoading = isLoadingTimeSeries || isLoadingAvg;
  const error = timeSeriesError || avgError;

  // Extract overall average tokens from the response (undefined if not available)
  const avgTokens = avgTokensData?.data_points?.[0]?.values?.[AggregationType.AVG];

  // Create a map of token stats by timestamp using shared utility
  const statsExtractor = useCallback(
    (dp: { values?: Record<string, number> }) => ({
      p50: dp.values?.[getPercentileKey(P50)] || 0,
      p90: dp.values?.[getPercentileKey(P90)] || 0,
      p99: dp.values?.[getPercentileKey(P99)] || 0,
    }),
    [],
  );
  const tokenStatsByTimestamp = useTimestampValueMap(tokenStatsDataPoints, statsExtractor);

  // Prepare chart data - fill in all time buckets with 0 for missing data
  const chartData = useMemo(() => {
    return timeBuckets.map((timestampMs) => {
      const stats = tokenStatsByTimestamp.get(timestampMs);
      return {
        name: formatTimestampForTraceMetrics(timestampMs, timeIntervalSeconds),
        p50: stats?.p50 || 0,
        p90: stats?.p90 || 0,
        p99: stats?.p99 || 0,
      };
    });
  }, [timeBuckets, tokenStatsByTimestamp, timeIntervalSeconds]);

  return {
    chartData,
    avgTokens,
    isLoading,
    error,
    hasData: tokenStatsDataPoints.length > 0,
  };
}
