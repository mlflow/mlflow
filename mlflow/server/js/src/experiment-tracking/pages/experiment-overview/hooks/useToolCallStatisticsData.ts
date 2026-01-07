import { useMemo } from 'react';
import {
  MetricViewType,
  AggregationType,
  SpanMetricKey,
  SpanFilterKey,
  SpanType,
  SpanStatus,
  SpanDimensionKey,
  createSpanFilter,
} from '@databricks/web-shared/model-trace-explorer';
import { useTraceMetricsQuery } from './useTraceMetricsQuery';
import type { OverviewChartProps } from '../types';

export interface UseToolCallStatisticsDataResult {
  /** Total number of tool calls */
  totalCalls: number;
  /** Number of successful tool calls */
  successCalls: number;
  /** Number of failed tool calls */
  failedCalls: number;
  /** Success rate as a percentage (0-100) */
  successRate: number;
  /** Average latency in milliseconds */
  avgLatency: number;
  /** Whether data is currently being fetched */
  isLoading: boolean;
  /** Error if data fetching failed */
  error: unknown;
}

/**
 * Custom hook that fetches and processes tool call statistics.
 * Queries span metrics for TOOL type spans and calculates counts and latency.
 *
 * @param props - Chart props including experimentId and time range
 * @returns Tool call statistics, loading state, and error state
 */
export function useToolCallStatisticsData({
  experimentId,
  startTimeMs,
  endTimeMs,
}: Pick<OverviewChartProps, 'experimentId' | 'startTimeMs' | 'endTimeMs'>): UseToolCallStatisticsDataResult {
  // Filter for TOOL type spans
  const toolFilter = useMemo(() => [createSpanFilter(SpanFilterKey.TYPE, SpanType.TOOL)], []);

  // Query tool call counts grouped by status (combines total and success/error counts)
  const {
    data: countByStatusData,
    isLoading: isLoadingCounts,
    error: countsError,
  } = useTraceMetricsQuery({
    experimentId,
    startTimeMs,
    endTimeMs,
    viewType: MetricViewType.SPANS,
    metricName: SpanMetricKey.SPAN_COUNT,
    aggregations: [{ aggregation_type: AggregationType.COUNT }],
    filters: toolFilter,
    dimensions: [SpanDimensionKey.SPAN_STATUS],
  });

  // Query average latency for tool calls
  const {
    data: latencyData,
    isLoading: isLoadingLatency,
    error: latencyError,
  } = useTraceMetricsQuery({
    experimentId,
    startTimeMs,
    endTimeMs,
    viewType: MetricViewType.SPANS,
    metricName: SpanMetricKey.LATENCY,
    aggregations: [{ aggregation_type: AggregationType.AVG }],
    filters: toolFilter,
  });

  // Calculate statistics from grouped data
  const { totalCalls, successCalls, failedCalls } = useMemo(() => {
    if (!countByStatusData?.data_points) {
      return { totalCalls: 0, successCalls: 0, failedCalls: 0 };
    }

    let total = 0;
    let success = 0;
    let failed = 0;

    for (const dp of countByStatusData.data_points) {
      const count = dp.values?.[AggregationType.COUNT] || 0;
      const status = dp.dimensions?.[SpanDimensionKey.SPAN_STATUS];
      total += count;
      if (status === SpanStatus.OK) {
        success += count;
      } else if (status === SpanStatus.ERROR) {
        failed += count;
      }
    }

    return { totalCalls: total, successCalls: success, failedCalls: failed };
  }, [countByStatusData?.data_points]);

  const successRate = totalCalls > 0 ? (successCalls / totalCalls) * 100 : 0;
  const avgLatency = latencyData?.data_points?.[0]?.values?.[AggregationType.AVG] || 0;

  const isLoading = isLoadingCounts || isLoadingLatency;
  const error = countsError || latencyError;

  return {
    totalCalls,
    successCalls,
    failedCalls,
    successRate,
    avgLatency,
    isLoading,
    error,
  };
}
