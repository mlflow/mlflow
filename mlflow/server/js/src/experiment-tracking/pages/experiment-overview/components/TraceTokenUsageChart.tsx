import React, { useCallback } from 'react';
import { useDesignSystemTheme, LightningIcon } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { useTraceTokenUsageChartData } from '../hooks/useTraceTokenUsageChartData';
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
import { formatCount, useLegendHighlight } from '../utils/chartUtils';

export const TraceTokenUsageChart: React.FC = () => {
  const { theme } = useDesignSystemTheme();
  const xAxisProps = useChartXAxisProps();
  const yAxisProps = useChartYAxisProps();
  const scrollableLegendProps = useScrollableLegendProps();
  const { getOpacity, handleLegendMouseEnter, handleLegendMouseLeave } = useLegendHighlight(0.8, 0.2);

  // Fetch and process token usage chart data
  const { chartData, totalTokens, totalInputTokens, totalOutputTokens, isLoading, error, hasData } =
    useTraceTokenUsageChartData();

  const tooltipFormatter = useCallback(
    (value: number, name: string) => [formatCount(value), name] as [string, string],
    [],
  );

  // Area colors
  const areaColors = {
    inputTokens: theme.colors.blue400,
    outputTokens: theme.colors.green400,
  };

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
        title={<FormattedMessage defaultMessage="Token Usage" description="Title for the token usage chart" />}
        value={formatCount(totalTokens)}
        subtitle={`(${formatCount(totalInputTokens)} input, ${formatCount(totalOutputTokens)} output)`}
      />

      <OverviewChartTimeLabel />

      {/* Chart */}
      <div css={{ height: DEFAULT_CHART_CONTENT_HEIGHT, marginTop: theme.spacing.sm }}>
        {hasData ? (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData} margin={{ top: 10, right: 30, left: 10, bottom: 0 }}>
              <XAxis dataKey="name" {...xAxisProps} />
              <YAxis {...yAxisProps} />
              <Tooltip
                content={<ScrollableTooltip formatter={tooltipFormatter} />}
                cursor={{ fill: theme.colors.actionTertiaryBackgroundHover }}
              />
              <Area
                type="monotone"
                dataKey="inputTokens"
                stackId="1"
                stroke={areaColors.inputTokens}
                fill={areaColors.inputTokens}
                strokeOpacity={getOpacity('Input Tokens')}
                fillOpacity={getOpacity('Input Tokens')}
                strokeWidth={2}
                name="Input Tokens"
              />
              <Area
                type="monotone"
                dataKey="outputTokens"
                stackId="1"
                stroke={areaColors.outputTokens}
                fill={areaColors.outputTokens}
                strokeOpacity={getOpacity('Output Tokens')}
                fillOpacity={getOpacity('Output Tokens')}
                strokeWidth={2}
                name="Output Tokens"
              />
              <Legend
                verticalAlign="bottom"
                iconType="plainline"
                onMouseEnter={handleLegendMouseEnter}
                onMouseLeave={handleLegendMouseLeave}
                {...scrollableLegendProps}
              />
            </AreaChart>
          </ResponsiveContainer>
        ) : (
          <OverviewChartEmptyState />
        )}
      </div>
    </OverviewChartContainer>
  );
};
