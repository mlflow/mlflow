import { useMemo, useCallback } from 'react';
import { MetricViewType, AggregationType, TraceMetricKey } from '@databricks/web-shared/model-trace-explorer';
import { useTraceMetricsQuery } from './useTraceMetricsQuery';
import { formatTimestampForTraceMetrics, useTimestampValueMap } from '../utils/chartUtils';
import type { OverviewChartProps } from '../types';

export interface TokenUsageChartDataPoint {
  name: string;
  inputTokens: number;
  outputTokens: number;
}

export interface UseTraceTokenUsageChartDataResult {
  /** Processed chart data with all time buckets filled */
  chartData: TokenUsageChartDataPoint[];
  /** Total tokens (input + output) in the time range */
  totalTokens: number;
  /** Total input tokens in the time range */
  totalInputTokens: number;
  /** Total output tokens in the time range */
  totalOutputTokens: number;
  /** Whether data is currently being fetched */
  isLoading: boolean;
  /** Error if data fetching failed */
  error: unknown;
  /** Whether there are any data points */
  hasData: boolean;
}

/**
 * Custom hook that fetches and processes token usage chart data.
 * Encapsulates all data-fetching and processing logic for the token usage chart.
 *
 * @param props - Chart props including experimentId, time range, and buckets
 * @returns Processed chart data, loading state, and error state
 */
export function useTraceTokenUsageChartData({
  experimentId,
  startTimeMs,
  endTimeMs,
  timeIntervalSeconds,
  timeBuckets,
}: OverviewChartProps): UseTraceTokenUsageChartDataResult {
  // Fetch input tokens over time
  const {
    data: inputTokensData,
    isLoading: isLoadingInput,
    error: inputError,
  } = useTraceMetricsQuery({
    experimentId,
    startTimeMs,
    endTimeMs,
    viewType: MetricViewType.TRACES,
    metricName: TraceMetricKey.INPUT_TOKENS,
    aggregations: [{ aggregation_type: AggregationType.SUM }],
    timeIntervalSeconds,
  });

  // Fetch output tokens over time
  const {
    data: outputTokensData,
    isLoading: isLoadingOutput,
    error: outputError,
  } = useTraceMetricsQuery({
    experimentId,
    startTimeMs,
    endTimeMs,
    viewType: MetricViewType.TRACES,
    metricName: TraceMetricKey.OUTPUT_TOKENS,
    aggregations: [{ aggregation_type: AggregationType.SUM }],
    timeIntervalSeconds,
  });

  // Fetch total tokens (without time bucketing) for the header
  const {
    data: totalTokensData,
    isLoading: isLoadingTotal,
    error: totalError,
  } = useTraceMetricsQuery({
    experimentId,
    startTimeMs,
    endTimeMs,
    viewType: MetricViewType.TRACES,
    metricName: TraceMetricKey.TOTAL_TOKENS,
    aggregations: [{ aggregation_type: AggregationType.SUM }],
  });

  const inputDataPoints = useMemo(() => inputTokensData?.data_points || [], [inputTokensData?.data_points]);
  const outputDataPoints = useMemo(() => outputTokensData?.data_points || [], [outputTokensData?.data_points]);
  const isLoading = isLoadingInput || isLoadingOutput || isLoadingTotal;
  const error = inputError || outputError || totalError;

  // Extract total tokens from the response
  const totalTokens = totalTokensData?.data_points?.[0]?.values?.[AggregationType.SUM] || 0;

  // Calculate total input and output tokens from time-bucketed data
  const totalInputTokens = useMemo(
    () => inputDataPoints.reduce((sum, dp) => sum + (dp.values?.[AggregationType.SUM] || 0), 0),
    [inputDataPoints],
  );
  const totalOutputTokens = useMemo(
    () => outputDataPoints.reduce((sum, dp) => sum + (dp.values?.[AggregationType.SUM] || 0), 0),
    [outputDataPoints],
  );

  // Create maps of tokens by timestamp using shared utility
  const sumExtractor = useCallback(
    (dp: { values?: Record<string, number> }) => dp.values?.[AggregationType.SUM] || 0,
    [],
  );
  const inputTokensMap = useTimestampValueMap(inputDataPoints, sumExtractor);
  const outputTokensMap = useTimestampValueMap(outputDataPoints, sumExtractor);

  // Prepare chart data - fill in all time buckets with 0 for missing data
  const chartData = useMemo(() => {
    return timeBuckets.map((timestampMs) => ({
      name: formatTimestampForTraceMetrics(timestampMs, timeIntervalSeconds),
      inputTokens: inputTokensMap.get(timestampMs) || 0,
      outputTokens: outputTokensMap.get(timestampMs) || 0,
    }));
  }, [timeBuckets, inputTokensMap, outputTokensMap, timeIntervalSeconds]);

  return {
    chartData,
    totalTokens,
    totalInputTokens,
    totalOutputTokens,
    isLoading,
    error,
    hasData: inputDataPoints.length > 0 || outputDataPoints.length > 0,
  };
}
