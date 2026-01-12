import { useMemo, useCallback } from 'react';
import {
  MetricViewType,
  AggregationType,
  TraceMetricKey,
  TraceFilterKey,
  TraceStatus,
  createTraceFilter,
} from '@databricks/web-shared/model-trace-explorer';
import { useTraceMetricsQuery } from './useTraceMetricsQuery';
import { formatTimestampForTraceMetrics, useTimestampValueMap } from '../utils/chartUtils';
import { useOverviewChartContext } from '../OverviewChartContext';

// Filter to get only error traces
const ERROR_FILTER = createTraceFilter(TraceFilterKey.STATUS, TraceStatus.ERROR);

export interface ErrorsChartDataPoint {
  name: string;
  errorCount: number;
  errorRate: number;
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
 * Encapsulates all data-fetching and processing logic for the errors chart.
 * Uses OverviewChartContext to get chart props.
 *
 * @returns Processed chart data, loading state, and error state
 */
export function useTraceErrorsChartData(): UseTraceErrorsChartDataResult {
  const { experimentId, startTimeMs, endTimeMs, timeIntervalSeconds, timeBuckets } = useOverviewChartContext();
  // Fetch error count metrics grouped by time bucket
  const {
    data: errorCountData,
    isLoading: isLoadingErrors,
    error: errorCountError,
  } = useTraceMetricsQuery({
    experimentId,
    startTimeMs,
    endTimeMs,
    viewType: MetricViewType.TRACES,
    metricName: TraceMetricKey.TRACE_COUNT,
    aggregations: [{ aggregation_type: AggregationType.COUNT }],
    timeIntervalSeconds,
    filters: [ERROR_FILTER],
  });

  // Fetch total trace count metrics grouped by time bucket (for calculating error rate)
  // This query is also used by TraceRequestsChart, so React Query will dedupe it
  const {
    data: totalCountData,
    isLoading: isLoadingTotal,
    error: totalCountError,
  } = useTraceMetricsQuery({
    experimentId,
    startTimeMs,
    endTimeMs,
    viewType: MetricViewType.TRACES,
    metricName: TraceMetricKey.TRACE_COUNT,
    aggregations: [{ aggregation_type: AggregationType.COUNT }],
    timeIntervalSeconds,
  });

  const errorDataPoints = useMemo(() => errorCountData?.data_points || [], [errorCountData?.data_points]);
  const totalDataPoints = useMemo(() => totalCountData?.data_points || [], [totalCountData?.data_points]);
  const isLoading = isLoadingErrors || isLoadingTotal;
  const error = errorCountError || totalCountError;

  // Calculate totals by summing time-bucketed data
  const totalErrors = useMemo(
    () => errorDataPoints.reduce((sum, dp) => sum + (dp.values?.[AggregationType.COUNT] || 0), 0),
    [errorDataPoints],
  );
  const totalTraces = useMemo(
    () => totalDataPoints.reduce((sum, dp) => sum + (dp.values?.[AggregationType.COUNT] || 0), 0),
    [totalDataPoints],
  );
  const overallErrorRate = totalTraces > 0 ? (totalErrors / totalTraces) * 100 : 0;

  // Create maps by timestamp for easy lookup using shared utility
  const countExtractor = useCallback(
    (dp: { values?: Record<string, number> }) => dp.values?.[AggregationType.COUNT] || 0,
    [],
  );
  const errorCountByTimestamp = useTimestampValueMap(errorDataPoints, countExtractor);
  const totalCountByTimestamp = useTimestampValueMap(totalDataPoints, countExtractor);

  // Prepare chart data - fill in all time buckets with 0 for missing data
  const chartData = useMemo(() => {
    return timeBuckets.map((timestampMs) => {
      const errorCount = errorCountByTimestamp.get(timestampMs) || 0;
      const totalCount = totalCountByTimestamp.get(timestampMs) || 0;
      const errorRate = totalCount > 0 ? (errorCount / totalCount) * 100 : 0;

      return {
        name: formatTimestampForTraceMetrics(timestampMs, timeIntervalSeconds),
        errorCount,
        errorRate: Math.round(errorRate * 100) / 100, // Round to 2 decimal places
      };
    });
  }, [timeBuckets, errorCountByTimestamp, totalCountByTimestamp, timeIntervalSeconds]);

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
    hasData: totalDataPoints.length > 0,
  };
}
