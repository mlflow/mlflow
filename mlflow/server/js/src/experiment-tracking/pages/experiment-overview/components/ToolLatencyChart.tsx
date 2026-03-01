import React, { useCallback } from 'react';
import { LightningIcon, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { useToolLatencyChartData } from '../hooks/useToolLatencyChartData';
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
import { formatLatency, useLegendHighlight, useChartColors, getLineDotStyle } from '../utils/chartUtils';

/**
 * Chart showing average latency comparison for each tool over time as a line chart.
 * Each line represents a different tool with its average latency.
 */
export const ToolLatencyChart: React.FC = () => {
  const { theme } = useDesignSystemTheme();
  const xAxisProps = useChartXAxisProps();
  const yAxisProps = useChartYAxisProps();
  const scrollableLegendProps = useScrollableLegendProps();
  const { getOpacity, handleLegendMouseEnter, handleLegendMouseLeave } = useLegendHighlight();
  const { getChartColor } = useChartColors();

  // Fetch and process tool latency chart data
  const { chartData, toolNames, isLoading, error, hasData } = useToolLatencyChartData();

  // Tool selection state
  const { displayedItems, isAllSelected, selectorLabel, handleSelectAllToggle, handleItemToggle } =
    useToolSelection(toolNames);

  const tooltipFormatter = useCallback(
    (value: number, name: string) => [formatLatency(value), name] as [string, string],
    [],
  );

  if (isLoading) {
    return <OverviewChartLoadingState />;
  }

  if (error) {
    return <OverviewChartErrorState />;
  }

  return (
    <OverviewChartContainer componentId="mlflow.charts.tool_latency">
      <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <OverviewChartHeader
          icon={<LightningIcon />}
          title={
            <FormattedMessage
              defaultMessage="Latency Comparison"
              description="Title for the tool latency comparison chart"
            />
          }
        />
        {hasData && (
          <ToolSelector
            componentId="mlflow.charts.tool_latency.tool_selector"
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
            <LineChart data={chartData} margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
              <XAxis dataKey="timestamp" {...xAxisProps} />
              <YAxis {...yAxisProps} />
              <Tooltip
                content={<ScrollableTooltip formatter={tooltipFormatter} />}
                cursor={{ stroke: theme.colors.actionTertiaryBackgroundHover }}
              />
              {displayedItems.map((toolName) => {
                const originalIndex = toolNames.indexOf(toolName);
                return (
                  <Line
                    key={toolName}
                    type="monotone"
                    dataKey={toolName}
                    stroke={getChartColor(originalIndex)}
                    strokeWidth={2}
                    strokeOpacity={getOpacity(toolName)}
                    dot={getLineDotStyle(getChartColor(originalIndex))}
                  />
                );
              })}
              <Legend
                verticalAlign="bottom"
                iconType="plainline"
                onMouseEnter={handleLegendMouseEnter}
                onMouseLeave={handleLegendMouseLeave}
                {...scrollableLegendProps}
              />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <OverviewChartEmptyState />
        )}
      </div>
    </OverviewChartContainer>
  );
};
