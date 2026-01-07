import React from 'react';
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
  useChartTooltipStyle,
  useChartXAxisProps,
  useChartLegendFormatter,
} from './OverviewChartComponents';
import { formatTokenCount, useLegendHighlight } from '../utils/chartUtils';
import type { OverviewChartProps } from '../types';

export const TraceTokenUsageChart: React.FC<OverviewChartProps> = (props) => {
  const { theme } = useDesignSystemTheme();
  const tooltipStyle = useChartTooltipStyle();
  const xAxisProps = useChartXAxisProps();
  const legendFormatter = useChartLegendFormatter();
  const { getOpacity, handleLegendMouseEnter, handleLegendMouseLeave } = useLegendHighlight(0.8, 0.2);

  // Fetch and process token usage chart data
  const { chartData, totalTokens, totalInputTokens, totalOutputTokens, isLoading, error, hasData } =
    useTraceTokenUsageChartData(props);

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
        value={formatTokenCount(totalTokens)}
        subtitle={`(${formatTokenCount(totalInputTokens)} input, ${formatTokenCount(totalOutputTokens)} output)`}
      />

      <OverviewChartTimeLabel />

      {/* Chart */}
      <div css={{ height: 200, marginTop: theme.spacing.sm }}>
        {hasData ? (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData} margin={{ top: 10, right: 30, left: 30, bottom: 0 }}>
              <XAxis dataKey="name" {...xAxisProps} />
              <YAxis hide />
              <Tooltip
                contentStyle={tooltipStyle}
                cursor={{ fill: theme.colors.actionTertiaryBackgroundHover }}
                formatter={(value: number, name: string) => [formatTokenCount(value), name]}
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
                height={36}
                onMouseEnter={handleLegendMouseEnter}
                onMouseLeave={handleLegendMouseLeave}
                formatter={legendFormatter}
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
