import { useMemo } from 'react';
import {
  MetricViewType,
  AggregationType,
  SpanMetricKey,
  SpanFilterKey,
  SpanType,
  SpanStatus,
  SpanDimensionKey,
  TIME_BUCKET_DIMENSION_KEY,
  createSpanFilter,
} from '@databricks/web-shared/model-trace-explorer';
import { useTraceMetricsQuery } from './useTraceMetricsQuery';
import { formatTimestampForTraceMetrics } from '../utils/chartUtils';
import { useOverviewChartContext } from '../OverviewChartContext';

export interface ToolErrorRateDataPoint {
  name: string;
  errorRate: number;
}

export interface UseToolErrorRateChartDataResult {
  /** Processed chart data with all time buckets filled */
  chartData: ToolErrorRateDataPoint[];
  /** Whether data is currently being fetched */
  isLoading: boolean;
  /** Error if data fetching failed */
  error: unknown;
  /** Whether there are any data points */
  hasData: boolean;
}

interface UseToolErrorRateChartDataProps {
  /** The name of the tool to fetch error rate data for */
  toolName: string;
}

/**
 * Custom hook that fetches and processes tool error rate data over time.
 * Queries span metrics grouped by status and time bucket to calculate error rate.
 * Uses OverviewChartContext to get chart props.
 *
 * @param props - Configuration including toolName
 * @returns Processed chart data, loading state, and error state
 */
export function useToolErrorRateChartData({
  toolName,
}: UseToolErrorRateChartDataProps): UseToolErrorRateChartDataResult {
  const { experimentId, startTimeMs, endTimeMs, timeIntervalSeconds, timeBuckets, filters: contextFilters } = useOverviewChartContext();

  // Filter for TOOL type spans with specific name, combined with context filters
  const toolFilters = useMemo(
    () => [
      createSpanFilter(SpanFilterKey.TYPE, SpanType.TOOL),
      createSpanFilter(SpanFilterKey.NAME, toolName),
      ...(contextFilters || []),
    ],
    [toolName, contextFilters],
  );

  // Query span counts grouped by status and time bucket
  const { data, isLoading, error } = useTraceMetricsQuery({
    experimentId,
    startTimeMs,
    endTimeMs,
    viewType: MetricViewType.SPANS,
    metricName: SpanMetricKey.SPAN_COUNT,
    aggregations: [{ aggregation_type: AggregationType.COUNT }],
    filters: toolFilters,
    dimensions: [SpanDimensionKey.SPAN_STATUS],
    timeIntervalSeconds,
  });

  const dataPoints = useMemo(() => data?.data_points || [], [data?.data_points]);

  // Group data by time bucket and calculate error rate for each bucket
  const errorRateByTimestamp = useMemo(() => {
    const bucketData = new Map<number, { error: number; total: number }>();

    for (const dp of dataPoints) {
      const timeBucket = dp.dimensions?.[TIME_BUCKET_DIMENSION_KEY];
      if (!timeBucket) continue;

      const timestampMs = new Date(timeBucket).getTime();
      const count = dp.values?.[AggregationType.COUNT] || 0;
      const status = dp.dimensions?.[SpanDimensionKey.SPAN_STATUS];

      if (!bucketData.has(timestampMs)) {
        bucketData.set(timestampMs, { error: 0, total: 0 });
      }

      const bucket = bucketData.get(timestampMs)!;
      bucket.total += count;
      if (status === SpanStatus.ERROR) {
        bucket.error += count;
      }
    }

    // Convert to error rate map
    const rateMap = new Map<number, number>();
    for (const [ts, { error, total }] of bucketData) {
      rateMap.set(ts, total > 0 ? (error / total) * 100 : 0);
    }
    return rateMap;
  }, [dataPoints]);

  // Build chart data with all time buckets
  const chartData = useMemo(() => {
    return timeBuckets.map((timestampMs) => ({
      name: formatTimestampForTraceMetrics(timestampMs, timeIntervalSeconds),
      errorRate: errorRateByTimestamp.get(timestampMs) || 0,
    }));
  }, [timeBuckets, errorRateByTimestamp, timeIntervalSeconds]);

  // Check if we have actual data
  const hasData = dataPoints.length > 0;

  return {
    chartData,
    isLoading,
    error,
    hasData,
  };
}
