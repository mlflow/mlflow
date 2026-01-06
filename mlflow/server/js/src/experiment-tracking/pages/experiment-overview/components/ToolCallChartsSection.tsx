import React, { useMemo, useCallback } from 'react';
import { useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { useToolCallChartsSectionData } from '../hooks/useToolCallChartsSectionData';
import { OverviewChartLoadingState, OverviewChartErrorState, OverviewChartEmptyState } from './OverviewChartComponents';
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

  const chartProps = { experimentId, startTimeMs, endTimeMs, timeIntervalSeconds, timeBuckets };

  if (isLoading) {
    return <OverviewChartLoadingState />;
  }

  if (error) {
    return <OverviewChartErrorState />;
  }

  if (!hasData) {
    return (
      <OverviewChartEmptyState
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
