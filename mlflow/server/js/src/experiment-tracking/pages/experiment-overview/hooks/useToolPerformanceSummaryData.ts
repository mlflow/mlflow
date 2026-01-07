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

export interface ToolPerformanceData {
  /** Tool name (span_name) */
  toolName: string;
  /** Total number of calls to this tool */
  totalCalls: number;
  /** Success rate as a percentage (0-100) */
  successRate: number;
  /** Average latency in milliseconds */
  avgLatency: number;
}

export interface UseToolPerformanceSummaryDataResult {
  /** Performance data for each tool */
  toolsData: ToolPerformanceData[];
  /** Whether data is currently being fetched */
  isLoading: boolean;
  /** Error if data fetching failed */
  error: unknown;
  /** Whether there is any data */
  hasData: boolean;
}

/**
 * Custom hook that fetches and processes tool performance summary data.
 * Queries span metrics grouped by tool name to get counts, success rates, and latencies.
 *
 * @param props - Chart props including experimentId and time range
 * @returns Tool performance data, loading state, and error state
 */
export function useToolPerformanceSummaryData({
  experimentId,
  startTimeMs,
  endTimeMs,
}: Pick<OverviewChartProps, 'experimentId' | 'startTimeMs' | 'endTimeMs'>): UseToolPerformanceSummaryDataResult {
  // Filter for TOOL type spans
  const toolFilter = useMemo(() => [createSpanFilter(SpanFilterKey.TYPE, SpanType.TOOL)], []);

  // Query tool call counts grouped by span_name and status
  const {
    data: countByToolAndStatusData,
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
    dimensions: [SpanDimensionKey.SPAN_NAME, SpanDimensionKey.SPAN_STATUS],
  });

  // Query average latency grouped by span_name
  const {
    data: latencyByToolData,
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
    dimensions: [SpanDimensionKey.SPAN_NAME],
  });

  // Process data into per-tool performance metrics
  const toolsData = useMemo(() => {
    const toolCountsMap = new Map<string, { total: number; success: number }>();
    const toolLatencyMap = new Map<string, number>();

    // Process count data grouped by tool and status
    if (countByToolAndStatusData?.data_points) {
      for (const dp of countByToolAndStatusData.data_points) {
        const toolName = dp.dimensions?.[SpanDimensionKey.SPAN_NAME];
        const status = dp.dimensions?.[SpanDimensionKey.SPAN_STATUS];
        const count = dp.values?.[AggregationType.COUNT] || 0;

        if (!toolName) continue;

        const existing = toolCountsMap.get(toolName) || { total: 0, success: 0 };
        existing.total += count;
        if (status === SpanStatus.OK) {
          existing.success += count;
        }
        toolCountsMap.set(toolName, existing);
      }
    }

    // Process latency data grouped by tool
    if (latencyByToolData?.data_points) {
      for (const dp of latencyByToolData.data_points) {
        const toolName = dp.dimensions?.[SpanDimensionKey.SPAN_NAME];
        const avgLatency = dp.values?.[AggregationType.AVG];

        if (toolName && avgLatency !== undefined) {
          toolLatencyMap.set(toolName, avgLatency);
        }
      }
    }

    // Combine into final data structure, sorted by total calls descending
    const result: ToolPerformanceData[] = [];
    for (const [toolName, counts] of toolCountsMap.entries()) {
      const successRate = counts.total > 0 ? (counts.success / counts.total) * 100 : 0;
      result.push({
        toolName,
        totalCalls: counts.total,
        successRate,
        avgLatency: toolLatencyMap.get(toolName) || 0,
      });
    }

    // Sort by total calls descending
    result.sort((a, b) => b.totalCalls - a.totalCalls);

    return result;
  }, [countByToolAndStatusData?.data_points, latencyByToolData?.data_points]);

  const isLoading = isLoadingCounts || isLoadingLatency;
  const error = countsError || latencyError;

  return {
    toolsData,
    isLoading,
    error,
    hasData: toolsData.length > 0,
  };
}
