import React, { useCallback } from 'react';
import { ChartLineIcon, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { useToolUsageChartData } from '../hooks/useToolUsageChartData';
import { useToolSelection } from '../hooks/useToolSelection';
import {
  OverviewChartLoadingState,
  OverviewChartErrorState,
  OverviewChartEmptyState,
  OverviewChartHeader,
  OverviewChartContainer,
  ScrollableTooltip,
  useChartXAxisProps,
  useChartYAxisProps,
  useScrollableLegendProps,
} from './OverviewChartComponents';
import { ToolSelector } from './ToolSelector';
import { formatCount, useLegendHighlight, useChartColors } from '../utils/chartUtils';

/**
 * Chart showing tool usage over time as a stacked bar chart.
 * Each bar is broken down by tool, with each tool having a different color.
 */
export const ToolUsageChart: React.FC = () => {
  const { theme } = useDesignSystemTheme();
  const xAxisProps = useChartXAxisProps();
  const yAxisProps = useChartYAxisProps();
  const scrollableLegendProps = useScrollableLegendProps();
  const { getOpacity, handleLegendMouseEnter, handleLegendMouseLeave } = useLegendHighlight();
  const { getChartColor } = useChartColors();

  // Fetch and process tool usage chart data
  const { chartData, toolNames, isLoading, error, hasData } = useToolUsageChartData();

  // Tool selection state
  const { displayedItems, isAllSelected, selectorLabel, handleSelectAllToggle, handleItemToggle } =
    useToolSelection(toolNames);

  const tooltipFormatter = useCallback(
    (value: number, name: string) => [formatCount(value), name] as [string, string],
    [],
  );

  if (isLoading) {
    return <OverviewChartLoadingState />;
  }

  if (error) {
    return <OverviewChartErrorState />;
  }

  return (
    <OverviewChartContainer componentId="mlflow.charts.tool_usage">
      <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <OverviewChartHeader
          icon={<ChartLineIcon />}
          title={
            <FormattedMessage defaultMessage="Tool Usage Over Time" description="Title for the tool usage chart" />
          }
        />
        {hasData && (
          <ToolSelector
            componentId="mlflow.charts.tool_usage.tool_selector"
            toolNames={toolNames}
            displayedItems={displayedItems}
            isAllSelected={isAllSelected}
            selectorLabel={selectorLabel}
            onSelectAllToggle={handleSelectAllToggle}
            onItemToggle={handleItemToggle}
          />
        )}
      </div>

      {/* Chart */}
      <div css={{ height: 300, marginTop: theme.spacing.sm }}>
        {hasData && displayedItems.length > 0 ? (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
              <XAxis dataKey="timestamp" {...xAxisProps} />
              <YAxis {...yAxisProps} />
              <Tooltip
                content={<ScrollableTooltip formatter={tooltipFormatter} />}
                cursor={{ fill: theme.colors.actionTertiaryBackgroundHover }}
              />
              {displayedItems.map((toolName) => {
                const originalIndex = toolNames.indexOf(toolName);
                return (
                  <Bar
                    key={toolName}
                    dataKey={toolName}
                    stackId="tools"
                    fill={getChartColor(originalIndex)}
                    fillOpacity={getOpacity(toolName)}
                  />
                );
              })}
              <Legend
                verticalAlign="bottom"
                iconType="square"
                onMouseEnter={handleLegendMouseEnter}
                onMouseLeave={handleLegendMouseLeave}
                {...scrollableLegendProps}
              />
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <OverviewChartEmptyState />
        )}
      </div>
    </OverviewChartContainer>
  );
};
