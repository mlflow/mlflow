import React, { useMemo, useCallback } from 'react';
import { useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
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
import { useTraceMetricsQuery } from '../hooks/useTraceMetricsQuery';
import { ChartLoadingState, ChartErrorState, ChartEmptyState } from './ChartCardWrapper';
import { ChartGrid } from './OverviewLayoutComponents';
import { LazyToolErrorRateChart } from './LazyToolErrorRateChart';
import type { OverviewChartProps } from '../types';

/**
 * Component that fetches available tools and renders an error rate chart for each one.
 */
export const ToolCallChartsSection: React.FC<OverviewChartProps> = ({
  experimentId,
  startTimeMs,
  endTimeMs,
  timeIntervalSeconds,
  timeBuckets,
}) => {
  const { theme } = useDesignSystemTheme();

  // Color palette using design system colors
  const toolColors = useMemo(
    () => [
      theme.colors.blue500,
      theme.colors.green500,
      theme.colors.red500,
      theme.colors.yellow500,
      theme.colors.blue300,
      theme.colors.green300,
      theme.colors.red300,
      theme.colors.yellow300,
    ],
    [theme],
  );

  // Get a color for a tool based on its index
  const getToolColor = useCallback((index: number): string => toolColors[index % toolColors.length], [toolColors]);

  // Filter for TOOL type spans
  const toolFilter = useMemo(() => [createSpanFilter(SpanFilterKey.TYPE, SpanType.TOOL)], []);

  // Query span counts grouped by span_name and span_status to get list of tools and their error rates
  const { data, isLoading, error } = useTraceMetricsQuery({
    experimentId,
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
    for (const [name, { error, total }] of toolData) {
      rates.set(name, total > 0 ? (error / total) * 100 : 0);
    }

    return { toolNames: Array.from(toolData.keys()).sort(), errorRateByTool: rates };
  }, [data?.data_points]);

  const chartProps = { experimentId, startTimeMs, endTimeMs, timeIntervalSeconds, timeBuckets };

  if (isLoading) {
    return <ChartLoadingState />;
  }

  if (error) {
    return <ChartErrorState />;
  }

  if (toolNames.length === 0) {
    return (
      <ChartEmptyState
        message={
          <FormattedMessage
            defaultMessage="No tool calls available"
            description="Message shown when there are no tool calls to display"
          />
        }
      />
    );
  }

  return (
    <ChartGrid>
      {toolNames.map((name, index) => (
        <LazyToolErrorRateChart
          key={name}
          {...chartProps}
          toolName={name}
          lineColor={getToolColor(index)}
          overallErrorRate={errorRateByTool.get(name) ?? 0}
        />
      ))}
    </ChartGrid>
  );
};
