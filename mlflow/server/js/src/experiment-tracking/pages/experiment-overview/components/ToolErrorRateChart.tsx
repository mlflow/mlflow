import React, { useCallback } from 'react';
import { WrenchIcon, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { useToolErrorRateChartData } from '../hooks/useToolErrorRateChartData';
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
  DEFAULT_CHART_CONTENT_HEIGHT,
} from './OverviewChartComponents';

export interface ToolErrorRateChartProps {
  /** The name of the tool to display */
  toolName: string;
  /** Optional color for the line chart */
  lineColor?: string;
  /** Pre-computed overall error rate */
  overallErrorRate: number;
}

export const ToolErrorRateChart: React.FC<ToolErrorRateChartProps> = ({ toolName, lineColor, overallErrorRate }) => {
  const { theme } = useDesignSystemTheme();
  const xAxisProps = useChartXAxisProps();
  const yAxisProps = useChartYAxisProps();
  const scrollableLegendProps = useScrollableLegendProps();

  const chartLineColor = lineColor || theme.colors.red500;

  // Fetch and process error rate chart data
  const { chartData, isLoading, error, hasData } = useToolErrorRateChartData({ toolName });

  const tooltipFormatter = useCallback(
    (value: number) => [`${value.toFixed(2)}%`, 'Error Rate'] as [string, string],
    [],
  );

  if (isLoading) {
    return <OverviewChartLoadingState />;
  }

  if (error) {
    return <OverviewChartErrorState />;
  }

  if (!hasData) {
    return (
      <OverviewChartContainer componentId="mlflow.charts.tool_error_rate" data-testid={`tool-chart-${toolName}`}>
        <OverviewChartHeader icon={<WrenchIcon css={{ color: chartLineColor }} />} title={toolName} />
        <OverviewChartEmptyState />
      </OverviewChartContainer>
    );
  }

  return (
    <OverviewChartContainer componentId="mlflow.charts.tool_error_rate" data-testid={`tool-chart-${toolName}`}>
      <OverviewChartHeader
        icon={<WrenchIcon css={{ color: chartLineColor }} />}
        title={toolName}
        value={`${overallErrorRate.toFixed(2)}%`}
        subtitle={
          <FormattedMessage defaultMessage="overall error rate" description="Subtitle for overall tool error rate" />
        }
      />

      <div css={{ height: DEFAULT_CHART_CONTENT_HEIGHT }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
            <XAxis dataKey="name" {...xAxisProps} />
            <YAxis domain={[0, 100]} tickFormatter={(value) => `${value}%`} {...yAxisProps} />
            <Tooltip
              content={<ScrollableTooltip formatter={tooltipFormatter} />}
              cursor={{ stroke: theme.colors.actionTertiaryBackgroundHover }}
            />
            <Legend iconType="plainline" {...scrollableLegendProps} />
            <Line
              type="monotone"
              dataKey="errorRate"
              name="Error Rate"
              stroke={chartLineColor}
              strokeWidth={2}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </OverviewChartContainer>
  );
};
