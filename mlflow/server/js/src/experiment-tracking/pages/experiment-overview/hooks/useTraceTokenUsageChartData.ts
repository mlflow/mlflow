import { useMemo, useCallback } from 'react';
import { MetricViewType, AggregationType, TraceMetricKey } from '@databricks/web-shared/model-trace-explorer';
import { useTraceMetricsQuery } from './useTraceMetricsQuery';
import { formatTimestampForTraceMetrics, useTimestampValueMap } from '../utils/chartUtils';
import { useOverviewChartContext } from '../OverviewChartContext';
import { shouldEnableBatchedTokenMetricQueries } from '../../../../common/utils/FeatureUtils';

const TOKEN_TIME_SERIES_METRIC_NAMES = [TraceMetricKey.INPUT_TOKENS, TraceMetricKey.OUTPUT_TOKENS];

export interface TokenUsageChartDataPoint {
  name: string;
  inputTokens: number;
  outputTokens: number;
  cacheReadTokens: number;
  cacheCreationTokens: number;
  timestampMs: number;
}

export interface UseTraceTokenUsageChartDataResult {
  chartData: TokenUsageChartDataPoint[];
  totalTokens: number;
  totalInputTokens: number;
  totalOutputTokens: number;
  /** Total cache read tokens in the time range */
  totalCacheReadTokens: number;
  /** Total cache creation tokens in the time range */
  totalCacheCreationTokens: number;
  isLoading: boolean;
  error: unknown;
  hasData: boolean;
}

export function useTraceTokenUsageChartData({
  enabled = true,
}: { enabled?: boolean } = {}): UseTraceTokenUsageChartDataResult {
  const { experimentIds, startTimeMs, endTimeMs, timeIntervalSeconds, timeBuckets, filters } =
    useOverviewChartContext();

  const isBatchingEnabled = Boolean(shouldEnableBatchedTokenMetricQueries());
  const commonParams = { experimentIds, startTimeMs, endTimeMs, viewType: MetricViewType.TRACES, filters };
  const timeSeriesAggregations = [{ aggregation_type: AggregationType.SUM }];

  // Batched path: single query for input + output tokens
  const {
    data: batchedData,
    isLoading: isLoadingBatched,
    error: batchedError,
  } = useTraceMetricsQuery({
    ...commonParams,
    metricNames: TOKEN_TIME_SERIES_METRIC_NAMES,
    aggregations: timeSeriesAggregations,
    timeIntervalSeconds,
    enabled: isBatchingEnabled && enabled,
  });

  // Non-batched path: separate queries
  const {
    data: inputData,
    isLoading: isLoadingInput,
    error: inputError,
  } = useTraceMetricsQuery({
    ...commonParams,
    metricName: TraceMetricKey.INPUT_TOKENS,
    aggregations: timeSeriesAggregations,
    timeIntervalSeconds,
    enabled: !isBatchingEnabled && enabled,
  });

  const {
    data: outputData,
    isLoading: isLoadingOutput,
    error: outputError,
  } = useTraceMetricsQuery({
    ...commonParams,
    metricName: TraceMetricKey.OUTPUT_TOKENS,
    aggregations: timeSeriesAggregations,
    timeIntervalSeconds,
    enabled: !isBatchingEnabled && enabled,
  });

  // Fetch cache read tokens over time
  const { data: cacheReadTokensData } = useTraceMetricsQuery({
    ...commonParams,
    metricName: TraceMetricKey.CACHE_READ_INPUT_TOKENS,
    aggregations: [{ aggregation_type: AggregationType.SUM }],
    timeIntervalSeconds,
    enabled,
  });

  // Fetch cache creation tokens over time
  const { data: cacheCreationTokensData } = useTraceMetricsQuery({
    ...commonParams,
    metricName: TraceMetricKey.CACHE_CREATION_INPUT_TOKENS,
    aggregations: [{ aggregation_type: AggregationType.SUM }],
    timeIntervalSeconds,
    enabled,
  });

  // Fetch total tokens (without time bucketing) for the header.
  // Uses [SUM, AVG] so React Query deduplicates with the identical call
  // in useTraceTokenStatsChartData.
  const {
    data: totalData,
    isLoading: isLoadingTotal,
    error: totalError,
  } = useTraceMetricsQuery({
    ...commonParams,
    metricName: TraceMetricKey.TOTAL_TOKENS,
    aggregations: [{ aggregation_type: AggregationType.SUM }, { aggregation_type: AggregationType.AVG }],
    enabled,
  });

  // Merge all data points (only one path is active) and filter by metric_name
  const allTimeSeriesPoints = useMemo(
    () => [...(batchedData?.data_points ?? []), ...(inputData?.data_points ?? []), ...(outputData?.data_points ?? [])],
    [batchedData?.data_points, inputData?.data_points, outputData?.data_points],
  );

  const cacheReadDataPoints = useMemo(() => cacheReadTokensData?.data_points || [], [cacheReadTokensData?.data_points]);
  const cacheCreationDataPoints = useMemo(
    () => cacheCreationTokensData?.data_points || [],
    [cacheCreationTokensData?.data_points],
  );

  const inputDataPoints = useMemo(
    () => allTimeSeriesPoints.filter((dp) => dp.metric_name === TraceMetricKey.INPUT_TOKENS),
    [allTimeSeriesPoints],
  );
  const outputDataPoints = useMemo(
    () => allTimeSeriesPoints.filter((dp) => dp.metric_name === TraceMetricKey.OUTPUT_TOKENS),
    [allTimeSeriesPoints],
  );

  const isLoading = isLoadingTotal || (isBatchingEnabled ? isLoadingBatched : isLoadingInput || isLoadingOutput);
  const error = totalError || (isBatchingEnabled ? batchedError : inputError || outputError);
  const totalTokens = totalData?.data_points?.[0]?.values?.[AggregationType.SUM] || 0;

  const totalInputTokens = useMemo(
    () => inputDataPoints.reduce((sum, dp) => sum + (dp.values?.[AggregationType.SUM] || 0), 0),
    [inputDataPoints],
  );
  const totalOutputTokens = useMemo(
    () => outputDataPoints.reduce((sum, dp) => sum + (dp.values?.[AggregationType.SUM] || 0), 0),
    [outputDataPoints],
  );
  const totalCacheReadTokens = useMemo(
    () => cacheReadDataPoints.reduce((sum, dp) => sum + (dp.values?.[AggregationType.SUM] || 0), 0),
    [cacheReadDataPoints],
  );
  const totalCacheCreationTokens = useMemo(
    () => cacheCreationDataPoints.reduce((sum, dp) => sum + (dp.values?.[AggregationType.SUM] || 0), 0),
    [cacheCreationDataPoints],
  );

  const sumExtractor = useCallback(
    (dp: { values?: Record<string, number> }) => dp.values?.[AggregationType.SUM] || 0,
    [],
  );
  const inputTokensMap = useTimestampValueMap(inputDataPoints, sumExtractor);
  const outputTokensMap = useTimestampValueMap(outputDataPoints, sumExtractor);
  const cacheReadTokensMap = useTimestampValueMap(cacheReadDataPoints, sumExtractor);
  const cacheCreationTokensMap = useTimestampValueMap(cacheCreationDataPoints, sumExtractor);

  const chartData = useMemo(() => {
    return timeBuckets.map((timestampMs) => ({
      name: formatTimestampForTraceMetrics(timestampMs, timeIntervalSeconds),
      inputTokens: inputTokensMap.get(timestampMs) || 0,
      outputTokens: outputTokensMap.get(timestampMs) || 0,
      cacheReadTokens: cacheReadTokensMap.get(timestampMs) || 0,
      cacheCreationTokens: cacheCreationTokensMap.get(timestampMs) || 0,
      timestampMs,
    }));
  }, [timeBuckets, inputTokensMap, outputTokensMap, cacheReadTokensMap, cacheCreationTokensMap, timeIntervalSeconds]);

  return {
    chartData,
    totalTokens,
    totalInputTokens,
    totalOutputTokens,
    totalCacheReadTokens,
    totalCacheCreationTokens,
    isLoading,
    error,
    hasData: inputDataPoints.length > 0 || outputDataPoints.length > 0,
  };
}
