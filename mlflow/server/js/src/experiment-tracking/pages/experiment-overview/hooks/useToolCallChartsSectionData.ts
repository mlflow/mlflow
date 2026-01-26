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
import { useOverviewChartContext } from '../OverviewChartContext';

export interface UseToolCallChartsSectionDataResult {
  /** Sorted list of tool names */
  toolNames: string[];
  /** Map of tool name to its error rate (0-100) */
  errorRateByTool: Map<string, number>;
  /** Whether data is currently being fetched */
  isLoading: boolean;
  /** Error if data fetching failed */
  error: unknown;
  /** Whether there are any tools */
  hasData: boolean;
}

/**
 * Custom hook that fetches and processes tool call data for the charts section.
 * Queries span metrics grouped by tool name and status to get the list of tools and their error rates.
 * Uses OverviewChartContext to get chart props.
 *
 * @returns Tool names, error rates, loading state, and error state
 */
export function useToolCallChartsSectionData(): UseToolCallChartsSectionDataResult {
  const { experimentIds, startTimeMs, endTimeMs, filters: contextFilters } = useOverviewChartContext();
  // Filter for TOOL type spans, combined with context filters
  const toolFilter = useMemo(
    () => [createSpanFilter(SpanFilterKey.TYPE, SpanType.TOOL), ...(contextFilters || [])],
    [contextFilters],
  );

  // Query span counts grouped by span_name and span_status to get list of tools and their error rates
  const { data, isLoading, error } = useTraceMetricsQuery({
    experimentIds,
    startTimeMs,
    endTimeMs,
    viewType: MetricViewType.SPANS,
    metricName: SpanMetricKey.SPAN_COUNT,
    aggregations: [{ aggregation_type: AggregationType.COUNT }],
    filters: toolFilter,
    dimensions: [SpanDimensionKey.SPAN_NAME, SpanDimensionKey.SPAN_STATUS],
  });

  // Extract tool names and calculate overall error rates
  const { toolNames, errorRateByTool } = useMemo(() => {
    if (!data?.data_points) return { toolNames: [], errorRateByTool: new Map<string, number>() };

    // Group by tool name and aggregate counts by status
    const toolData = new Map<string, { error: number; total: number }>();

    for (const dp of data.data_points) {
      const name = dp.dimensions?.[SpanDimensionKey.SPAN_NAME];
      if (!name) continue;

      const count = dp.values?.[AggregationType.COUNT] || 0;
      const status = dp.dimensions?.[SpanDimensionKey.SPAN_STATUS];

      if (!toolData.has(name)) {
        toolData.set(name, { error: 0, total: 0 });
      }

      const tool = toolData.get(name)!;
      tool.total += count;
      if (status === SpanStatus.ERROR) {
        tool.error += count;
      }
    }

    // Calculate error rates
    const rates = new Map<string, number>();
    for (const [name, { error: errorCount, total }] of toolData) {
      rates.set(name, total > 0 ? (errorCount / total) * 100 : 0);
    }

    return { toolNames: Array.from(toolData.keys()).sort(), errorRateByTool: rates };
  }, [data?.data_points]);

  return {
    toolNames,
    errorRateByTool,
    isLoading,
    error,
    hasData: toolNames.length > 0,
  };
}
