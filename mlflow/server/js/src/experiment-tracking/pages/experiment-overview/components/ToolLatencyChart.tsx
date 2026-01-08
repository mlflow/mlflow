import React from 'react';
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
  useChartTooltipStyle,
  useChartXAxisProps,
  useChartLegendFormatter,
} from './OverviewChartComponents';
import { formatLatency, useLegendHighlight, useToolColors } from '../utils/chartUtils';
import type { OverviewChartProps } from '../types';

/**
 * Chart showing average latency comparison for each tool over time as a line chart.
 * Each line represents a different tool with its average latency.
 */
export const ToolLatencyChart: React.FC<OverviewChartProps> = (props) => {
  const { theme } = useDesignSystemTheme();
  const tooltipStyle = useChartTooltipStyle();
  const xAxisProps = useChartXAxisProps();
  const legendFormatter = useChartLegendFormatter();
  const { getOpacity, handleLegendMouseEnter, handleLegendMouseLeave } = useLegendHighlight();
  const { getToolColor } = useToolColors();

  // Fetch and process tool latency chart data
  const { chartData, toolNames, isLoading, error, hasData } = useToolLatencyChartData(props);

  if (isLoading) {
    return <OverviewChartLoadingState />;
  }

  if (error) {
    return <OverviewChartErrorState />;
  }

  return (
    <OverviewChartContainer>
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
      <div css={{ height: 200, marginTop: theme.spacing.sm }}>
        {hasData ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
              <XAxis dataKey="timestamp" {...xAxisProps} />
              <YAxis hide />
              <Tooltip
                contentStyle={tooltipStyle}
                cursor={{ stroke: theme.colors.actionTertiaryBackgroundHover }}
                formatter={(value: number, name: string) => [formatLatency(value), name]}
              />
              {toolNames.map((toolName, index) => (
                <Line
                  key={toolName}
                  type="monotone"
                  dataKey={toolName}
                  stroke={getToolColor(index)}
                  strokeWidth={2}
                  strokeOpacity={getOpacity(toolName)}
                  dot={false}
                />
              ))}
              <Legend
                verticalAlign="bottom"
                iconType="plainline"
                height={36}
                onMouseEnter={handleLegendMouseEnter}
                onMouseLeave={handleLegendMouseLeave}
                formatter={legendFormatter}
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
