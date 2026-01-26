import React from 'react';
import { useToolCallChartsSectionData } from '../hooks/useToolCallChartsSectionData';
import { OverviewChartLoadingState, OverviewChartErrorState } from './OverviewChartComponents';
import { ChartGrid } from './OverviewLayoutComponents';
import { LazyToolErrorRateChart } from './LazyToolErrorRateChart';
import { useChartColors } from '../utils/chartUtils';

/**
 * Component that fetches available tools and renders an error rate chart for each one.
 */
export const ToolCallChartsSection: React.FC = () => {
  const { getChartColor } = useChartColors();

  // Fetch and process tool call data using the custom hook
  const { toolNames, errorRateByTool, isLoading, error, hasData } = useToolCallChartsSectionData();

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
        <div key={name} id={`tool-chart-${name}`}>
          <LazyToolErrorRateChart
            toolName={name}
            lineColor={getChartColor(index)}
            overallErrorRate={errorRateByTool.get(name) ?? 0}
          />
        </div>
      ))}
    </ChartGrid>
  );
};
