import React from 'react';
import { useToolCallChartsSectionData } from '../hooks/useToolCallChartsSectionData';
import { OverviewChartLoadingState, OverviewChartErrorState } from './OverviewChartComponents';
import { ChartGrid } from './OverviewLayoutComponents';
import { LazyToolErrorRateChart } from './LazyToolErrorRateChart';
import { useToolColors } from '../utils/chartUtils';
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
  const { getToolColor } = useToolColors();

  // Fetch and process tool call data using the custom hook
  const { toolNames, errorRateByTool, isLoading, error, hasData } = useToolCallChartsSectionData({
    experimentId,
    startTimeMs,
    endTimeMs,
  });

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
