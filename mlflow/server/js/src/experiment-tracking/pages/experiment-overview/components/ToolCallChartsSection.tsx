import React from 'react';
import { WrenchIcon } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { useToolCallChartsSectionData } from '../hooks/useToolCallChartsSectionData';
import { useToolSelection } from '../hooks/useToolSelection';
import {
  OverviewChartLoadingState,
  OverviewChartErrorState,
  OverviewChartHeader,
  OverviewChartContainer,
} from './OverviewChartComponents';
import { ChartGrid } from './OverviewLayoutComponents';
import { LazyToolErrorRateChart } from './LazyToolErrorRateChart';
import { ToolSelector } from './ToolSelector';
import { useChartColors } from '../utils/chartUtils';

/**
 * Component that fetches available tools and renders an error rate chart for each one.
 */
export const ToolCallChartsSection: React.FC = () => {
  const { getChartColor } = useChartColors();

  // Fetch and process tool call data using the custom hook
  const { toolNames, errorRateByTool, isLoading, error, hasData } = useToolCallChartsSectionData();

  // Tool selection state
  const { displayedItems, isAllSelected, selectorLabel, handleSelectAllToggle, handleItemToggle } =
    useToolSelection(toolNames);

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
    <OverviewChartContainer componentId="mlflow.charts.tool_error_rate_section">
      <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 16 }}>
        <OverviewChartHeader
          icon={<WrenchIcon />}
          title={
            <FormattedMessage defaultMessage="Tool Error Rates" description="Title for the tool error rates section" />
          }
        />
        <ToolSelector
          componentId="mlflow.charts.tool_error_rate.tool_selector"
          toolNames={toolNames}
          displayedItems={displayedItems}
          isAllSelected={isAllSelected}
          selectorLabel={selectorLabel}
          onSelectAllToggle={handleSelectAllToggle}
          onItemToggle={handleItemToggle}
        />
      </div>
      <ChartGrid>
        {displayedItems.map((name) => {
          const originalIndex = toolNames.indexOf(name);
          return (
            <div key={name} id={`tool-chart-${name}`}>
              <LazyToolErrorRateChart
                toolName={name}
                lineColor={getChartColor(originalIndex)}
                overallErrorRate={errorRateByTool.get(name) ?? 0}
              />
            </div>
          );
        })}
      </ChartGrid>
    </OverviewChartContainer>
  );
};
