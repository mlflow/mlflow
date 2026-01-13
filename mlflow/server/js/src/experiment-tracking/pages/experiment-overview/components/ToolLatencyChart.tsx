import React, { useCallback } from 'react';
import { LightningIcon, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { useToolLatencyChartData } from '../hooks/useToolLatencyChartData';
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
  DEFAULT_CHART_CONTENT_HEIGHT,
} from './OverviewChartComponents';
import { formatLatency, useLegendHighlight, useChartColors } from '../utils/chartUtils';

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
      <OverviewChartHeader
        icon={<LightningIcon />}
        title={
          <FormattedMessage
            defaultMessage="Latency Comparison"
            description="Title for the tool latency comparison chart"
          />
        }
      />

      <OverviewChartTimeLabel />

      {/* Chart */}
      <div css={{ height: DEFAULT_CHART_CONTENT_HEIGHT, marginTop: theme.spacing.sm }}>
        {hasData ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
              <XAxis dataKey="timestamp" {...xAxisProps} />
              <YAxis {...yAxisProps} />
              <Tooltip
                content={<ScrollableTooltip formatter={tooltipFormatter} />}
                cursor={{ stroke: theme.colors.actionTertiaryBackgroundHover }}
              />
              {toolNames.map((toolName, index) => (
                <Line
                  key={toolName}
                  type="monotone"
                  dataKey={toolName}
                  stroke={getChartColor(index)}
                  strokeWidth={2}
                  strokeOpacity={getOpacity(toolName)}
                  dot={false}
                />
              ))}
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
