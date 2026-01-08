import React, { useMemo } from 'react';
import { useToolCallChartsSectionData } from '../hooks/useToolCallChartsSectionData';
import { OverviewChartLoadingState, OverviewChartErrorState } from './OverviewChartComponents';
import { ChartGrid } from './OverviewLayoutComponents';
import { LazyToolErrorRateChart } from './LazyToolErrorRateChart';
import { useChartColors } from '../utils/chartUtils';

interface ToolCallChartsSectionProps {
  /** Optional search query to filter tools by name */
  searchQuery?: string;
}

/**
 * Component that fetches available tools and renders an error rate chart for each one.
 */
export const ToolCallChartsSection: React.FC<ToolCallChartsSectionProps> = ({ searchQuery }) => {
  const { getChartColor } = useChartColors();

  // Fetch and process tool call data using the custom hook
  const { toolNames, errorRateByTool, isLoading, error, hasData } = useToolCallChartsSectionData();

  // Filter tool names based on search query (matches tool name which is the chart title)
  const filteredToolNames = useMemo(() => {
    if (!searchQuery?.trim()) return toolNames;
    const query = searchQuery.toLowerCase().trim();
    return toolNames.filter((name) => name.toLowerCase().includes(query));
  }, [toolNames, searchQuery]);

  if (isLoading) {
    return <OverviewChartLoadingState />;
  }

  if (error) {
    return <OverviewChartErrorState />;
  }

  // Don't render anything if there are no tools or no matches
  if (!hasData || filteredToolNames.length === 0) {
    return null;
  }

  return (
    <ChartGrid>
      {filteredToolNames.map((name) => {
        // Use original index for consistent colors
        const originalIndex = toolNames.indexOf(name);
        return (
          <LazyToolErrorRateChart
            key={name}
            toolName={name}
            lineColor={getChartColor(originalIndex)}
            overallErrorRate={errorRateByTool.get(name) ?? 0}
          />
        );
      })}
    </ChartGrid>
  );
};
