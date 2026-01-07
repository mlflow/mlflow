import React, { useMemo, useCallback } from 'react';
import { useDesignSystemTheme } from '@databricks/design-system';
import { useToolCallChartsSectionData } from '../hooks/useToolCallChartsSectionData';
import { OverviewChartLoadingState, OverviewChartErrorState } from './OverviewChartComponents';
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

  // Fetch and process tool call data using the custom hook
  const { toolNames, errorRateByTool, isLoading, error, hasData } = useToolCallChartsSectionData({
    experimentId,
    startTimeMs,
    endTimeMs,
  });

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

  if (isLoading) {
    return <OverviewChartLoadingState />;
  }

  if (error) {
    return <OverviewChartErrorState />;
  }

  // Don't render anything if there are no tools, other charts handle the empty states as well
  if (!hasData) {
    return null;
  }

  return (
    <ChartGrid>
      {toolNames.map((name, index) => (
        <LazyToolErrorRateChart
          key={name}
          experimentId={experimentId}
          startTimeMs={startTimeMs}
          endTimeMs={endTimeMs}
          timeIntervalSeconds={timeIntervalSeconds}
          timeBuckets={timeBuckets}
          toolName={name}
          lineColor={getToolColor(index)}
          overallErrorRate={errorRateByTool.get(name) ?? 0}
        />
      ))}
    </ChartGrid>
  );
};
