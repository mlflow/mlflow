import React, { useCallback } from 'react';
import { ChartLineIcon, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { useToolUsageChartData } from '../hooks/useToolUsageChartData';
import {
  OverviewChartLoadingState,
  OverviewChartErrorState,
  OverviewChartEmptyState,
  OverviewChartHeader,
  OverviewChartContainer,
  OverviewChartTimeLabel,
  ScrollableTooltip,
  useChartXAxisProps,
  useChartYAxisProps,
  useScrollableLegendProps,
} from './OverviewChartComponents';
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
    <OverviewChartContainer>
      <OverviewChartHeader
        icon={<ChartLineIcon />}
        title={<FormattedMessage defaultMessage="Tool Usage Over Time" description="Title for the tool usage chart" />}
      />

      <OverviewChartTimeLabel />

      {/* Chart */}
      <div css={{ height: 200, marginTop: theme.spacing.sm }}>
        {hasData ? (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
              <XAxis dataKey="timestamp" {...xAxisProps} />
              <YAxis {...yAxisProps} />
              <Tooltip
                content={<ScrollableTooltip formatter={tooltipFormatter} />}
                cursor={{ fill: theme.colors.actionTertiaryBackgroundHover }}
              />
              {toolNames.map((toolName, index) => (
                <Bar
                  key={toolName}
                  dataKey={toolName}
                  stackId="tools"
                  fill={getChartColor(index)}
                  fillOpacity={getOpacity(toolName)}
                />
              ))}
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
