import { useMemo } from 'react';
import {
  MetricViewType,
  AggregationType,
  SpanMetricKey,
  SpanFilterKey,
  SpanType,
  SpanDimensionKey,
  TIME_BUCKET_DIMENSION_KEY,
  createSpanFilter,
} from '@databricks/web-shared/model-trace-explorer';
import { useTraceMetricsQuery } from './useTraceMetricsQuery';
import { formatTimestampForTraceMetrics } from '../utils/chartUtils';
import { useOverviewChartContext } from '../OverviewChartContext';

export interface ToolLatencyDataPoint {
  timestamp: string;
  [toolName: string]: string | number;
}

export interface UseToolLatencyChartDataResult {
  /** Processed chart data with all time buckets filled */
  chartData: ToolLatencyDataPoint[];
  /** Sorted list of tool names */
  toolNames: string[];
  /** Whether data is currently being fetched */
  isLoading: boolean;
  /** Error if data fetching failed */
  error: unknown;
  /** Whether there are any data points */
  hasData: boolean;
}

/**
 * Custom hook that fetches and processes tool latency data over time.
 * Queries span metrics grouped by tool name and time bucket for average latency.
 * Uses OverviewChartContext to get chart props.
 *
 * @returns Processed chart data, tool names, loading state, and error state
 */
export function useToolLatencyChartData(): UseToolLatencyChartDataResult {
  const { experimentId, startTimeMs, endTimeMs, timeIntervalSeconds, timeBuckets, filters: contextFilters } = useOverviewChartContext();
  // Filter for TOOL type spans, combined with context filters
  const toolFilter = useMemo(
    () => [createSpanFilter(SpanFilterKey.TYPE, SpanType.TOOL), ...(contextFilters || [])],
    [contextFilters],
  );

  // Query average latency grouped by span_name and time bucket
  const { data, isLoading, error } = useTraceMetricsQuery({
    experimentId,
    startTimeMs,
    endTimeMs,
    viewType: MetricViewType.SPANS,
    metricName: SpanMetricKey.LATENCY,
    aggregations: [{ aggregation_type: AggregationType.AVG }],
    filters: toolFilter,
    dimensions: [SpanDimensionKey.SPAN_NAME],
    timeIntervalSeconds,
  });

  // Extract tool names and build chart data
  const { chartData, toolNames } = useMemo(() => {
    const dataPoints = data?.data_points ?? [];
    const toolSet = new Set<string>();
    const dataByTimestamp = new Map<number, Map<string, number>>();

    // Collect tool names and group latencies by timestamp
    for (const dp of dataPoints) {
      const toolName = dp.dimensions?.[SpanDimensionKey.SPAN_NAME];
      const timeBucket = dp.dimensions?.[TIME_BUCKET_DIMENSION_KEY];
      const avgLatency = dp.values?.[AggregationType.AVG];

      if (!toolName || !timeBucket || avgLatency === undefined) continue;

      toolSet.add(toolName);

      const timestampMs = new Date(timeBucket).getTime();
      let toolLatencies = dataByTimestamp.get(timestampMs);
      if (!toolLatencies) {
        toolLatencies = new Map<string, number>();
        dataByTimestamp.set(timestampMs, toolLatencies);
      }
      toolLatencies.set(toolName, avgLatency);
    }

    const sortedToolNames = Array.from(toolSet).sort();

    // Build chart data with all time buckets filled
    const chartDataResult = timeBuckets.map((timestampMs) => {
      const toolLatencies = dataByTimestamp.get(timestampMs);
      const dataPoint: ToolLatencyDataPoint = {
        timestamp: formatTimestampForTraceMetrics(timestampMs, timeIntervalSeconds),
      };

      // Add latency for each tool (0 if not present)
      for (const toolName of sortedToolNames) {
        dataPoint[toolName] = toolLatencies?.get(toolName) ?? 0;
      }

      return dataPoint;
    });

    return {
      chartData: chartDataResult,
      toolNames: sortedToolNames,
    };
  }, [data?.data_points, timeBuckets, timeIntervalSeconds]);

  return {
    chartData,
    toolNames,
    isLoading,
    error,
    hasData: toolNames.length > 0,
  };
}
